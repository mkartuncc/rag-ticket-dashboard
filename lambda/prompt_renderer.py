"""
Lightweight prompt renderer for Lambda — replaces poml with plain Python strings.
Faithfully reproduces the same prompts as the .poml templates.
"""

ANSWER_TECHNIQUES = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "cot": "Chain of Thought",
    "advanced": "Advanced (Constrained Generation)",
}


def render_routing_prompt(question: str) -> str:
    return (
        "You are a router for a RAG code-documentation chatbot.\n"
        "Your job is to look at the user's question and choose exactly ONE target collection:\n"
        '"python_book", "java_book", "javascript_book", or "CLARIFY" if the language is unclear.\n'
        "Return JSON only (no extra text).\n\n"
        "Collections:\n"
        "- python_book: Python syntax, standard library, Python programming concepts.\n"
        "- java_book: Java syntax, core libraries, Java programming concepts.\n"
        "- javascript_book: JavaScript syntax, browser/Node concepts, JS programming concepts.\n"
        "- CLARIFY: Use when the question does not clearly indicate one language.\n\n"
        "Chain of Thought (internal only — output JSON only):\n"
        "1. Detect explicit language mentions (Python, Java, JavaScript, JS, Node).\n"
        "2. Detect language-specific keywords (pip/venv/def → Python; JVM/ArrayList → Java; npm/async/await → JS).\n"
        "3. If exactly one language is strongly indicated, route to that collection.\n"
        "4. If multiple or none, route to CLARIFY.\n\n"
        "Examples:\n"
        '  Input: how do python lists work? → {"collection":"python_book","message":""}\n'
        '  Input: ArrayList vs LinkedList? → {"collection":"java_book","message":""}\n'
        '  Input: how do I install packages? → {"collection":"CLARIFY","message":"Which language: Python, Java, or JavaScript?"}\n\n'
        f"Input: {question}\n"
        'Output (JSON only): {"collection":"python_book|java_book|javascript_book|CLARIFY","message":"string"}'
    )


def render_answer_prompt(technique: str, question: str, context: list,
                         language: str, conversation_context: str = "") -> str:
    ctx_text = "\n---\n".join(c["content"] for c in context) if context else "No context available."

    if technique == "zero_shot":
        return (
            "You are a code-documentation Q&A assistant.\n"
            "Use ONLY the provided Context to answer the Question.\n"
            "If the answer is not in the Context, say you don't know.\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {question}"
        )
    elif technique == "few_shot":
        return (
            "You are a code-documentation Q&A assistant.\n"
            "Answer using ONLY the provided Context. If Context doesn't contain the answer, say you don't know.\n\n"
            "Examples:\n"
            "Q: What does a list do in Python?\n"
            "A: A Python list stores multiple values and supports indexing and modification.\n\n"
            "Q: How do I read a line from standard input in Java?\n"
            "A: Use Scanner or BufferedReader as shown in the Java docs.\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {question}"
        )
    elif technique == "cot":
        return (
            "You are a code-documentation Q&A assistant.\n"
            "Use ONLY the provided Context. Think through internally, output only the final answer.\n\n"
            "Reasoning steps (internal):\n"
            "1. Identify language/topic and locate relevant Context.\n"
            "2. Extract exact facts from Context.\n"
            "3. Write a concise answer with a minimal example only if supported.\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {question}"
        )
    elif technique == "advanced":
        return (
            "You are a code-documentation Q&A assistant using Constrained Generation.\n"
            "Answer ONLY from the provided Context. Follow the output format exactly.\n\n"
            "Rules:\n"
            "- If you cannot quote evidence from Context, say 'I don't know from the provided docs'.\n"
            "- Do NOT write code unless Context includes that pattern.\n"
            "- If Context contains conflicts, mention the conflict and ask for clarification.\n\n"
            "Required output format:\n"
            "Answer (1-3 sentences)\n\n"
            "Evidence:\n- Evidence 1: \"...\"\n- Evidence 2: \"...\"\n\n"
            "Example (only if Context supports it)\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Question: {question}"
        )
    else:
        raise ValueError(f"Unknown technique: {technique}")


def render_judge_prompt(question: str, responses: list, comparison_type: str, chunks: list) -> str:
    ctx_text = "\n---\n".join(c["content"] for c in chunks) if chunks else "No context provided."
    responses_text = ""
    for i, r in enumerate(responses, 1):
        label = r.get("model_name") or r.get("technique_name") or f"Response {i}"
        responses_text += f"Response {i} ({label}):\n{r.get('response', '')}\n---\n"

    return (
        "You are an impartial judge evaluating AI-generated answers for a code documentation Q&A task.\n"
        "Penalize hallucinations. Reward grounding in the provided Context.\n\n"
        "Criteria:\n"
        "1. Grounding: Answer uses ONLY Context. Penalize outside knowledge.\n"
        "2. Correctness: Factually accurate per Context.\n"
        "3. Completeness: Fully answers the Question using Context.\n"
        "4. Clarity: Concise and well-structured.\n"
        "5. Refusal: If Context is missing info, model correctly says 'I don't know'.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Comparison type: {comparison_type}\n\n"
        f"Responses:\n{responses_text}\n"
        "Output ONLY valid JSON with exactly these keys:\n"
        '{"rankings": ["Response 1 (label)", ...], '
        '"feedback": {"Response 1 (label)": "..."}, '
        '"summary": "Winner: ... Reason: ..."}'
    )


def get_answer_techniques() -> dict:
    return ANSWER_TECHNIQUES
