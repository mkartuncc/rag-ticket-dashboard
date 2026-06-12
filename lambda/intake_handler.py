"""
Intake Lambda — receives email payload from API Gateway, runs RAG pipeline,
stores ticket in DynamoDB, sends auto-reply via SES.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import boto3

from rag_pipeline import run_rag_pipeline
from github_helper import create_ticket_pr

dynamodb = boto3.resource("dynamodb", region_name=os.environ["AWS_REGION_NAME"])
ses = boto3.client("ses", region_name=os.environ["AWS_REGION_NAME"])
table = dynamodb.Table(os.environ["DYNAMODB_TABLE"])

SENDER_EMAIL = os.environ["SENDER_EMAIL"]


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
    except (json.JSONDecodeError, TypeError):
        return {"statusCode": 400, "body": "Invalid JSON"}

    sender = body.get("sender", "").strip()
    question = body.get("question", "").strip()

    if not sender or not question:
        return {"statusCode": 400, "body": "Missing sender or question"}

    ticket_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Run RAG pipeline
    result = run_rag_pipeline(question, response_technique="zero_shot")
    answer = result.get("response") or result.get("error") or "Could not generate an answer."
    route = result.get("routing", {}).get("route", "unknown")
    chunks = result.get("chunks") or []

    # Create GitHub PR for this ticket
    pr_number = None
    try:
        pr_number = create_ticket_pr(
            ticket_id=ticket_id,
            question=question,
            answer=answer,
            route=route,
            sender=sender,
        )
        print(f"Created PR #{pr_number} for ticket {ticket_id}")
    except Exception as e:
        print(f"GitHub PR creation failed: {e}")

    # Store ticket in DynamoDB (chunks saved for evaluator)
    table.put_item(Item={
        "ticket_id": ticket_id,
        "sender_email": sender,
        "question": question,
        "answer": answer,
        "route": route,
        "chunks": json.dumps([{"content": c["content"]} for c in chunks]),
        "pr_number": pr_number,
        "status": "pending_eval",
        "created_at": created_at,
        "evaluated_at": None,
        "eval_score": None,
        "eval_feedback": None,
    })

    # Send auto-reply via SES
    try:
        ses.send_email(
            Source=SENDER_EMAIL,
            Destination={"ToAddresses": [sender]},
            Message={
                "Subject": {"Data": "Re: Your coding question"},
                "Body": {
                    "Text": {
                        "Data": (
                            f"Hi,\n\nThank you for your question:\n\"{question}\"\n\n"
                            f"Here is your answer:\n\n{answer}\n\n"
                            f"— Code Documentation Assistant\n"
                            f"[Ticket ID: {ticket_id}]"
                        )
                    }
                },
            },
        )
    except Exception as e:
        print(f"SES send failed: {e}")

    return {
        "statusCode": 200,
        "body": json.dumps({"ticket_id": ticket_id, "status": "pending_eval"}),
        "headers": {"Content-Type": "application/json"},
    }
