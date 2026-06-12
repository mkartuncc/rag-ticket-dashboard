"""
RAG Pipeline for Code Documentation Q&A

Flow:
1. Route question to correct collection using selected prompting technique
2. Query Pinecone namespace
3. Generate response using an LLM with retrieved context
4. Optional comparisons (models or techniques) and LLM-as-a-Judge evaluation
"""

import json
import os
import re
from dotenv import load_dotenv
import json
import boto3
from langchain_aws import ChatBedrockConverse
from pinecone import Pinecone

from prompt_renderer import (
    render_routing_prompt,
    render_answer_prompt,
    get_answer_techniques,
)

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is required. Add it to your .env file.")

INDEX_NAME = "code-docs"
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(INDEX_NAME)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_embedding(text: str) -> list:
    response = bedrock_runtime.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": text[:8000]}),
    )
    return json.loads(response["body"].read())["embedding"]

COLLECTIONS = ["python_book", "java_book", "javascript_book"]

COMPARISON_MODELS = [
    {"id": "amazon.nova-pro-v1:0", "name": "Amazon Nova Pro"},
    {"id": "amazon.nova-lite-v1:0", "name": "Amazon Nova Lite"},
    {"id": "amazon.nova-micro-v1:0", "name": "Amazon Nova Micro"},
]
ROUTER_MODEL = "amazon.nova-micro-v1:0"
JUDGE_MODEL = "amazon.nova-pro-v1:0"


def extract_json(text: str) -> dict:
    """Extract JSON object from LLM response."""
    json_pattern = r"\{[^{}]*\}"
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            return json.loads(match)#TODO load the JSON formatted string, 'matches' into a python dictionary
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(match)#TODO load the JSON formatted string, 'matches' into a python dictionary
    except json.JSONDecodeError:
        return {"route": "CLARIFY", "message": "Could not parse routing response"}


def route_question(question: str) -> dict:
    """Route the question to the appropriate collection."""

    # use the prompt_renderer.py file to build the prompt using the route.poml prompt
    # TODO use the corrert function from prompt_renderer.py and pass it the question
    prompt = render_routing_prompt(question)

    llm = ChatBedrockConverse(
        model_id=ROUTER_MODEL,
        region_name=AWS_REGION,
        temperature=0,
    )
    response = llm.invoke(prompt)

    llm_response = response.content.strip()
    result = extract_json(llm_response)
    route = result.get("route") or result.get("collection", "CLARIFY")

    # Fallback: if LLM returned CLARIFY but question has an explicit language keyword, override
    if route == "CLARIFY":
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["java ", "java?", "java.", "in java", "java class", "javadoc"]):
            route = "java_book"
        elif any(kw in q_lower for kw in ["python", "django", "flask", "pip ", "def "]):
            route = "python_book"
        elif any(kw in q_lower for kw in ["javascript", "js ", "node", "npm", "react", "typescript"]):
            route = "javascript_book"

    return {
        "route": route,
        "message": result.get("message", ""),
        "raw_response": llm_response,
    }


def query_collection(collection_name: str, question: str, top_k: int = 5) -> list:
    """Query Pinecone namespace to look for similar chunks."""
    try:
        query_vector = get_embedding(question)
        results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=collection_name,
            include_metadata=True,
        )
        chunks = []
        for match in results["matches"]:
            chunks.append(
                {
                    "content": match["metadata"].get("content", ""),
                    "metadata": match["metadata"],
                    "distance": 1 - match["score"],  # cosine similarity → distance
                }
            )
        return chunks

    except Exception as e:
        print(f"Error querying collection: {e}")
        return []


def generate_response(question: str, context_chunks: list, collection_name: str,
                      conversation_context: str = "", response_technique: str = "zero_shot",
                      model_id: str = "amazon.nova-lite-v1:0") -> str:
    """Generate a response using an LLM with retrieved context."""

    language_map = {
        "python_book": "Python",
        "java_book": "Java",
        "javascript_book": "JavaScript",
    }
    language = language_map.get(collection_name, "programming")

    #TODO Call the correct function from prompt_renderer.py
    prompt = render_answer_prompt(
        technique=response_technique,
        question=question,
        context=context_chunks,
        language=language,
        conversation_context=conversation_context,
    )

    llm = ChatBedrockConverse(
        model_id=model_id,
        region_name=AWS_REGION,
        temperature=0.2,
    )
    # TODO Invoke the LLM with the prompt
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_multi_model_responses(question: str, context_chunks: list, collection_name: str,
                                   conversation_context: str = "",
                                   response_technique: str = "zero_shot") -> list:
    """Generate responses from multiple models for comparison."""
    responses = []

    for model in COMPARISON_MODELS:
        try:
            answer = generate_response(
                question=question,
                context_chunks=context_chunks,
                collection_name=collection_name,
                conversation_context=conversation_context,
                response_technique=response_technique,
                model_id=model["id"],
            )
            responses.append({"model_name": model["name"], "response": answer, "error": None})
        except Exception as e:
            responses.append({"model_name": model["name"], "response": None, "error": str(e)})

    return responses


def generate_multi_technique_responses(question: str, context_chunks: list, collection_name: str,
                                       conversation_context: str = "",
                                       model_id: str = "amazon.nova-pro-v1:0") -> list:
    """Generate responses from multiple prompting techniques for comparison."""
    responses = []
    techniques = get_answer_techniques()

    for technique_key, technique_name in techniques.items():
        try:
            answer = generate_response(
                question=question,
                context_chunks=context_chunks,
                collection_name=collection_name,
                conversation_context=conversation_context,
                response_technique=technique_key,
                model_id=model_id,
            )
            responses.append(
                {
                    "technique_key": technique_key,
                    "technique_name": technique_name,
                    "response": answer,
                    "error": None,
                }
            )
        except Exception as e:
            responses.append(
                {
                    "technique_key": technique_key,
                    "technique_name": technique_name,
                    "response": None,
                    "error": str(e),
                }
            )

    return responses


def run_rag_pipeline(question: str, response_technique: str,
                     conversation_context: str = "") -> dict:
    """Run the full RAG pipeline (route -> query -> generate)."""
    result = {
        "routing": None,
        "chunks": None,
        "response": None,
        "error": None,
    }

    try:
        routing = route_question(question)
        result["routing"] = routing

        if routing["route"] == "CLARIFY":
            result["response"] = routing.get("message") or "Please specify the programming language."
            return result

        if routing["route"] not in COLLECTIONS:
            result["error"] = f"Invalid route: {routing['route']}"
            return result

        chunks = query_collection(routing["route"], question)
        result["chunks"] = chunks

        response = generate_response(
            question=question,
            context_chunks=chunks,
            collection_name=routing["route"],
            conversation_context=conversation_context,
            response_technique=response_technique,
        )
        result["response"] = response

    except Exception as e:
        result["error"] = str(e)

    return result


def run_model_comparison(question: str, response_technique: str) -> dict:
    """Route once, then compare multiple models on the same question."""

    routing = route_question(question)

    if routing["route"] == "CLARIFY":
        return {"error": routing.get("message") or "Please specify the programming language."}

    if routing["route"] not in COLLECTIONS:
        return {"error": f"Invalid route: {routing['route']}"}

    chunks = query_collection(routing["route"], question)

    model_responses = generate_multi_model_responses(
        question=question,
        context_chunks=chunks,
        collection_name=routing["route"],
        response_technique=response_technique,
    )

    return {
        "routing": routing,
        "chunks": chunks,
        "model_responses": model_responses,
    }


def run_technique_comparison(question: str, model_id: str) -> dict:
    """Route once, then compare multiple prompting techniques using one model."""
    routing = route_question(question)

    if routing["route"] == "CLARIFY":
        return {"error": routing.get("message") or "Please specify the programming language."}

    if routing["route"] not in COLLECTIONS:
        return {"error": f"Invalid route: {routing['route']}"}

    chunks = query_collection(routing["route"], question)

    technique_responses = generate_multi_technique_responses(
        question=question,
        context_chunks=chunks,
        collection_name=routing["route"],
        model_id=model_id,
    )

    return {
        "routing": routing,
        "chunks": chunks,
        "technique_responses": technique_responses,
    }
