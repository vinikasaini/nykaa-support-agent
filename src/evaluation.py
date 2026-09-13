import json
import re
from pathlib import Path

from src.rag_pipeline import (
    retrieve_context,
    mock_llm,
    THRESHOLD,
    MOCK_LLM
)


EVALUATION_LOG = Path("logs/rag_evaluation.json")


TEST_SET = [
    {
        "id": 1,
        "topic": "Cancellation",
        "query": "Can I cancel my order before it is dispatched?"
    },
    {
        "id": 2,
        "topic": "Cash on Delivery",
        "query": "When is Cash on Delivery available?"
    },
    {
        "id": 3,
        "topic": "Damaged Product",
        "query": "What should I do if my product arrives damaged?"
    },
    {
        "id": 4,
        "topic": "Delayed Shipment",
        "query": "What should I do if my order delivery is delayed?"
    },
    {
        "id": 5,
        "topic": "Delivery",
        "query": "How can I track my order after it has been shipped?"
    },
    {
        "id": 6,
        "topic": "Escalation",
        "query": "When should I contact customer support for an unresolved issue?"
    },
    {
        "id": 7,
        "topic": "Exchange",
        "query": "What are the conditions for exchanging a product?"
    },
    {
        "id": 8,
        "topic": "Loyalty and Rewards",
        "query": "What conditions can apply to loyalty rewards and promotional offers?"
    },
    {
        "id": 9,
        "topic": "Missing Product",
        "query": "What should I do if an item is missing from my order?"
    },
    {
        "id": 10,
        "topic": "Payment",
        "query": "What should I do if my payment failed but money was deducted?"
    },
    {
        "id": 11,
        "topic": "Refund",
        "query": "When is a refund processed after an approved return?"
    },
    {
        "id": 12,
        "topic": "Return",
        "query": "How many days do I have to request a return?"
    },
    {
        "id": 13,
        "topic": "Out of Scope",
        "query": "What is the weather today?"
    },
    {
        "id": 14,
        "topic": "Out of Scope",
        "query": "Can you recommend a laptop for gaming?"
    },
    {
        "id": 15,
        "topic": "Edge Case",
        "query": "Can you tell me my exact bank account balance?"
    }
]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def tokenize(text):
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "what", "when", "where", "how", "can", "i",
        "my", "me", "to", "for", "of", "and", "or",
        "if", "it", "do", "does", "should", "be",
        "in", "on", "after", "before", "from", "with",
        "this", "that", "your", "you"
    }

    words = normalize_text(text).split()

    return {
        word
        for word in words
        if len(word) > 2 and word not in stop_words
    }


def keyword_overlap(text_a, text_b):
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a.intersection(tokens_b)

    return len(intersection) / len(tokens_a)


def mock_judge(prompt, query, context, answer, similarity):
    """
    MOCK_LLM-based evaluation judge.

    The prompt represents the LLM-as-judge instruction.
    Since MOCK_LLM=True in this project, deterministic
    scoring is used instead of calling an external LLM.
    """

    if not MOCK_LLM:
        raise RuntimeError(
            "Task 13 requires MOCK_LLM=True for this evaluation."
        )

    query_tokens = tokenize(query)
    context_tokens = tokenize(context)
    answer_tokens = tokenize(answer)

    if not query_tokens:
        query_tokens = {"support"}

    context_overlap = (
        len(query_tokens.intersection(context_tokens))
        / len(query_tokens)
    )

    answer_overlap = (
        len(query_tokens.intersection(answer_tokens))
        / len(query_tokens)
    )

    context_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]\s+",
            context
        )
        if sentence.strip()
    ]

    context_relevance = 1

    if similarity >= 0.65:
        context_relevance = 5
    elif similarity >= 0.55:
        context_relevance = 4
    elif similarity >= THRESHOLD:
        context_relevance = 3
    elif similarity >= 0.35:
        context_relevance = 2
    else:
        context_relevance = 1

    if context_overlap >= 0.65:
        context_relevance = min(5, context_relevance + 1)

    if (
        "i don't know" in answer.lower()
        or "could not find sufficiently relevant" in answer.lower()
    ):
        groundedness = 5 if similarity < THRESHOLD else 2
        answer_relevance = 4 if similarity < THRESHOLD else 2
    else:
        if similarity >= THRESHOLD:
            groundedness = 5
        elif similarity >= 0.40:
            groundedness = 3
        else:
            groundedness = 1

        if answer_overlap >= 0.65:
            answer_relevance = 5
        elif answer_overlap >= 0.45:
            answer_relevance = 4
        elif answer_overlap >= 0.25:
            answer_relevance = 3
        elif answer_overlap > 0:
            answer_relevance = 2
        else:
            answer_relevance = 1

    judge_prompt = f"""
You are an evaluation judge for a customer-support RAG system.

Evaluate the system response using three metrics.

1. Context Relevance:
Does the retrieved context contain information relevant to the
customer question?

2. Groundedness:
Is the answer supported by the retrieved context?
The answer must not introduce unsupported information.

3. Answer Relevance:
Does the answer directly address the customer's question?

Give each metric a score from 1 to 5.

Customer Question:
{query}

Retrieved Context:
{context}

System Answer:
{answer}

Top Similarity:
{similarity}

Return JSON with:
context_relevance
groundedness
answer_relevance
"""

    return {
        "judge_prompt": judge_prompt.strip(),
        "context_relevance": context_relevance,
        "groundedness": groundedness,
        "answer_relevance": answer_relevance
    }


def evaluate_query(item):
    query = item["query"]

    retrieved_chunks = retrieve_context(query)

    if not retrieved_chunks:
        context = ""
        top_similarity = 0.0
        answer = (
            "I don't know. "
            "I could not find sufficiently relevant "
            "information in the knowledge base."
        )
        sources = []
    else:
        context = "\n\n".join(
            chunk["content"]
            for chunk in retrieved_chunks
        )

        top_similarity = retrieved_chunks[0]["similarity"]

        sources = [
            chunk["source"]
            for chunk in retrieved_chunks
        ]

        if top_similarity < THRESHOLD:
            answer = (
                "I don't know. "
                "I could not find sufficiently relevant "
                "information in the knowledge base."
            )
        else:
            answer = mock_llm(
                query,
                retrieved_chunks
            )

    judge_result = mock_judge(
        prompt="",
        query=query,
        context=context,
        answer=answer,
        similarity=top_similarity
    )

    return {
        "id": item["id"],
        "topic": item["topic"],
        "query": query,
        "top_similarity": round(
            float(top_similarity),
            4
        ),
        "threshold": THRESHOLD,
        "retrieved_sources": sources,
        "answer": answer,
        "context_relevance": judge_result[
            "context_relevance"
        ],
        "groundedness": judge_result[
            "groundedness"
        ],
        "answer_relevance": judge_result[
            "answer_relevance"
        ],
        "judge_prompt": judge_result[
            "judge_prompt"
        ]
    }


def calculate_average(results, metric):
    if not results:
        return 0.0

    return round(
        sum(
            result[metric]
            for result in results
        ) / len(results),
        2
    )


def save_results(results, averages):
    EVALUATION_LOG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "evaluation_type": "RAG Triad Evaluation",
        "mock_llm": MOCK_LLM,
        "total_queries": len(results),
        "metrics": {
            "context_relevance": {
                "scale": "1-5",
                "average": averages[
                    "context_relevance"
                ]
            },
            "groundedness": {
                "scale": "1-5",
                "average": averages[
                    "groundedness"
                ]
            },
            "answer_relevance": {
                "scale": "1-5",
                "average": averages[
                    "answer_relevance"
                ]
            }
        },
        "results": results
    }

    with open(
        EVALUATION_LOG,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    print("\n" + "=" * 70)
    print("RAG TRIAD EVALUATION")
    print("=" * 70)

    print("\nEvaluation configuration:")
    print("Total queries:", len(TEST_SET))
    print("MOCK_LLM:", MOCK_LLM)
    print("Similarity threshold:", THRESHOLD)

    results = []

    for item in TEST_SET:

        print("\n" + "-" * 70)
        print(
            f"Query {item['id']} "
            f"| Topic: {item['topic']}"
        )
        print("-" * 70)

        print("Question:")
        print(item["query"])

        result = evaluate_query(item)

        results.append(result)

        print(
            "\nTop-1 Similarity:",
            result["top_similarity"]
        )

        print("Retrieved Sources:")

        for source in result["retrieved_sources"]:
            print(" -", source)

        print("\nRAG Triad Scores:")
        print(
            "Context Relevance:",
            result["context_relevance"],
            "/ 5"
        )
        print(
            "Groundedness:",
            result["groundedness"],
            "/ 5"
        )
        print(
            "Answer Relevance:",
            result["answer_relevance"],
            "/ 5"
        )

    averages = {
        "context_relevance": calculate_average(
            results,
            "context_relevance"
        ),
        "groundedness": calculate_average(
            results,
            "groundedness"
        ),
        "answer_relevance": calculate_average(
            results,
            "answer_relevance"
        )
    }

    print("\n" + "=" * 70)
    print("AVERAGE RAG TRIAD SCORES")
    print("=" * 70)

    print(
        "Context Relevance:",
        averages["context_relevance"],
        "/ 5"
    )

    print(
        "Groundedness:",
        averages["groundedness"],
        "/ 5"
    )

    print(
        "Answer Relevance:",
        averages["answer_relevance"],
        "/ 5"
    )

    save_results(
        results,
        averages
    )

    print("\nEvaluation results saved to:")
    print(EVALUATION_LOG)


if __name__ == "__main__":
    main()