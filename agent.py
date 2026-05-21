"""
agent.py
--------
Simple CLI agent for TelecomGPT RAG pipeline.

Usage:
    python agent.py            # runs interactive loop, reads GROQ_API_KEY from .env
    python agent.py --force    # force rebuild index before running
    python agent.py --query "What is X?"  # single-shot query

The agent uses functions from `rag_pipeline.py`: `get_vectorstore`, `build_llm`, and `answer_question`.
"""

import argparse
import os
from dotenv import load_dotenv

from rag_pipeline import get_vectorstore, build_llm, answer_question, TOP_K_RETRIEVE


def main():
    parser = argparse.ArgumentParser(description="TelecomGPT CLI agent")
    parser.add_argument("--force", action="store_true", help="Force rebuild index before running")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY missing from .env")

    print("\n[Agent] Initializing vectorstore (this may take a minute)...")
    vs = get_vectorstore(force_rebuild=args.force)
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
    llm = build_llm(api_key)

    history = ""

    def ask(q: str):
        nonlocal history
        answer, best_docs, all_docs = answer_question(q, retriever, llm, history)
        print("\n=== Answer ===\n")
        print(answer)
        print("\n=== Sources ===\n")
        for i, d in enumerate(best_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            print(f"{i}. {src} (p.{page})")
        history += f"\nUser: {q}\nAssistant: {answer}\n"

    if args.query:
        ask(args.query)
        return

    print("\nTelecomGPT agent ready. Type a question or 'quit' to exit.")
    while True:
        try:
            q = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            break
        ask(q)


if __name__ == "__main__":
    main()
