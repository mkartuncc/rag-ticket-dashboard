"""
Evaluator Lambda — runs every 15 min via EventBridge, fetches all pending_eval
tickets from DynamoDB, scores each with the LLM judge, updates status.
"""
import json
import os
from datetime import datetime, timezone

import boto3
from langchain_aws import ChatBedrockConverse

from github_helper import update_ticket_pr

dynamodb = boto3.resource("dynamodb", region_name=os.environ["AWS_REGION_NAME"])
table = dynamodb.Table(os.environ["DYNAMODB_TABLE"])

JUDGE_MODEL = "amazon.nova-pro-v1:0"
NEEDS_REVIEW_THRESHOLD = 6  # scores below this get flagged


def build_eval_prompt(question: str, answer: str, chunks: list) -> str:
    ctx = "\n---\n".join(c["content"] for c in chunks) if chunks else "No context provided."
    return (
        "You are evaluating a RAG assistant's answer to a coding question.\n"
        "Score the answer from 1 to 10 based on these criteria:\n"
        "- Grounding: answer uses only the provided Context (not outside knowledge)\n"
        "- Correctness: factually accurate per the Context\n"
        "- Completeness: fully addresses the question using available Context\n"
        "- Clarity: concise and well-structured\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Answer:\n{answer}\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"score": <integer 1-10>, "feedback": "<one sentence explanation>"}'
    )


def score_ticket(ticket: dict) -> dict:
    question = ticket["question"]
    answer   = ticket["answer"]
    chunks   = json.loads(ticket.get("chunks") or "[]")

    prompt = build_eval_prompt(question, answer, chunks)

    llm = ChatBedrockConverse(
        model_id=JUDGE_MODEL,
        region_name=os.environ["AWS_REGION_NAME"],
        temperature=0,
    )
    content = llm.invoke(prompt).content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        result = json.loads(match.group(0)) if match else {}

    score    = int(result.get("score", 7))
    score    = max(1, min(10, score))  # clamp to 1-10
    feedback = result.get("feedback", "")
    status   = "approved" if score >= NEEDS_REVIEW_THRESHOLD else "needs_review"
    return {"score": score, "feedback": feedback, "status": status}


def lambda_handler(event, context):
    # Fetch all pending_eval tickets
    response = table.query(
        IndexName="status-created-index",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("status").eq("pending_eval"),
    )
    tickets = response.get("Items", [])
    print(f"Evaluating {len(tickets)} tickets")

    evaluated = 0
    for ticket in tickets:
        try:
            result = score_ticket(ticket)
            table.update_item(
                Key={"ticket_id": ticket["ticket_id"]},
                UpdateExpression=(
                    "SET #s = :s, eval_score = :sc, eval_feedback = :fb, evaluated_at = :ea"
                ),
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":s": result["status"],
                    ":sc": result["score"],
                    ":fb": result["feedback"],
                    ":ea": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Update GitHub PR with score + label
            pr_number = ticket.get("pr_number")
            if pr_number:
                try:
                    update_ticket_pr(
                        pr_number=int(pr_number),
                        score=result["score"],
                        feedback=result["feedback"],
                        status=result["status"],
                    )
                except Exception as e:
                    print(f"GitHub PR update failed for #{pr_number}: {e}")

            evaluated += 1
            print(f"Ticket {ticket['ticket_id']}: score={result['score']} status={result['status']}")
        except Exception as e:
            print(f"Failed to evaluate {ticket['ticket_id']}: {e}")

    return {"statusCode": 200, "body": json.dumps({"evaluated": evaluated})}
