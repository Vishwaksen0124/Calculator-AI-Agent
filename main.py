"""main.py

CLI entry point for the Strict Calculation Agent.

Usage:
    # Interactive mode:
    python main.py

    # Single query execution:
    python main.py "Calculate 45 * 87 + sqrt(144)"
    python main.py "What is the capital of France?"

    # Use Google Gemini (free tier) instead of local Ollama:
    python main.py --provider gemini --model gemini-2.0-flash "Calculate sin(pi/4)"
"""

import argparse
import sys
from agent import CalculationAgent, get_default_llm


def main():
    parser = argparse.ArgumentParser(
        description="Strict Calculation AI Agent (Stateless, Tool-Enforced, Free Models)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="The mathematical question to calculate. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini"],
        default=None,
        help="LLM provider: 'ollama' (local free open-source) or 'gemini' (free-tier API)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., 'qwen3.5:4b' for Ollama, 'gemini-2.0-flash' for Gemini)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display tool call parameters and execution details",
    )

    args = parser.parse_args()

    # Initialize model and agent
    try:
        llm = get_default_llm(provider=args.provider, model_name=args.model)
        agent = CalculationAgent(llm=llm)
    except Exception as exc:
        print(f"\n[Error Initializing Agent]: {exc}", file=sys.stderr)
        sys.exit(1)

    # Single query mode
    if args.query:
        result = agent.run(args.query)
        if args.verbose:
            print("\n" + "=" * 50)
            print(f"[Query]       : {result['query']}")
            print(f"[Tool Called] : {result['tool_called']}")
            if result["tool_called"]:
                print(f"[Expression]  : {result['expression']}")
                print(f"[Tool Result] : {result['tool_result']}")
            print("=" * 50)
        print(f"\n{result['response']}\n")
        return

    # Interactive mode
    print("\n=======================================================")
    print("      Strict Calculation AI Agent (Stateless)          ")
    print("=======================================================")
    print("Features:")
    print(" • Strictly performs calculations via the calculation tool.")
    print(" • Rejects all non-mathematical questions.")
    print(" • Stateless: zero conversation history retained.")
    print(" • Minimal inputs & outputs to prevent hallucinations.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("CalcAgent> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        result = agent.run(user_input)

        if args.verbose:
            print("-" * 40)
            print(f"Tool Called : {result['tool_called']}")
            if result["tool_called"]:
                print(f"Expression  : {result['expression']}")
                print(f"Tool Output : {result['tool_result']}")
            print("-" * 40)

        print(f"\n{result['response']}\n")


if __name__ == "__main__":
    main()
