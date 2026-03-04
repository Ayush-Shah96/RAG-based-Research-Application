"""
main.py — CLI entry point for the RAG Research Agent.

Run:
    python -m src.main
or:
    python src/main.py
"""

import sys

from src.agent.agent_core import ResearchAgent


def main() -> None:
    agent = ResearchAgent()

    print("=" * 60)
    print("  RAG Research Agent")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            query = input("\nAsk your research question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Agent] Goodbye!")
            sys.exit(0)

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("[Agent] Goodbye!")
            sys.exit(0)

        print()
        answer = agent.run(query)
        print("\n" + "-" * 60)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()