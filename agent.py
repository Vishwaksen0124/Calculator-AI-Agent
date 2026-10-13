"""agent.py

Open-source LangChain agent strictly specialized for calculations.
Guarantees:
1. Strictly invokes the calculate tool for all calculations; never outputs mental math.
2. Refuses to answer any questions not regarding calculations.
3. Stateless execution: no conversation history maintained or passed.
4. Minimal inputs and outputs to prevent hallucinations.
5. Uses free AI models (Ollama local / Gemini free tier).
"""

import os
import re
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel

from calculator_tool import calculate, evaluate_math

# Load environment variables (e.g., GEMINI_API_KEY, OLLAMA_MODEL)
load_dotenv()

REFUSAL_MESSAGE = (
    "I can only assist with mathematical calculations. Please provide a math or calculation query."
)

SYSTEM_PROMPT = """You are a strict, specialized Calculation Agent.

MANDATORY RULES:
1. DOMAIN RESTRICTION: You answer ONLY requests regarding mathematical calculations.
2. NON-CALCULATION REFUSAL: If the user's prompt is NOT asking for a mathematical calculation (e.g., general knowledge, chatting, history, coding, creative writing, opinions), you MUST NOT answer it. You must refuse immediately with EXACTLY:
"I can only assist with mathematical calculations. Please provide a math or calculation query."
3. STRICT TOOL USE: You are strictly forbidden from performing calculations mentally or estimating numbers. You MUST call the `calculate` tool for ANY mathematical calculation.
4. MINIMAL DATA: Pass ONLY the raw mathematical expression string to the `calculate` tool (parameter: `expression`).
5. GROUNDED RESULTS: Report the final answer strictly derived from the tool result. Do not alter or guess numbers.
"""

# Keywords that indicate mathematical calculation intent
MATH_KEYWORDS: Set[str] = {
    "add", "sum", "plus", "subtract", "minus", "difference",
    "multiply", "times", "product", "divide", "division", "quotient",
    "calculate", "compute", "evaluate", "solve", "equation",
    "sqrt", "cbrt", "root", "power", "pow", "exponent",
    "sin", "cos", "tan", "asin", "acos", "atan", "log", "ln", "exp",
    "factorial", "derivative", "differentiate", "integral", "integrate",
    "percentage", "fraction", "modulo", "mod", "pi"
}


def has_calculation_intent(text: str) -> bool:
    """Fast guardrail check to determine if text contains mathematical intent."""
    clean = text.lower()
    # Check for arithmetic operators
    if re.search(r"[+\-*/%^=<>]", clean):
        return True
    # Check for whole math words
    tokens = set(re.findall(r"\b[a-z]+\b", clean))
    if tokens & MATH_KEYWORDS:
        return True
    # Digits accompanied by math context or basic expressions
    if re.search(r"\d", clean) and any(
        kw in clean for kw in ["is", "value", "what", "find", "how much", "equal", "to the power"]
    ):
        return True
    return False


def get_default_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Initializes a free open-source or free-tier AI model.
    
    Supported providers:
    - 'ollama': Local free open-source models (default: qwen3.5:4b or lfm2.5:8b-a1b-q4_K_M).
    - 'gemini': Google Gemini free-tier API (e.g. gemini-1.5-flash / gemini-2.0-flash).
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "")).lower()

    if not provider:
        # Auto-detect: if GEMINI_API_KEY is present and Ollama isn't explicitly set, use gemini, else ollama
        if os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            provider = "ollama"

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini provider. "
                "Or use Ollama (local free AI) by setting LLM_PROVIDER=ollama."
            )
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=temperature,
            api_key=api_key,
        )

    # Default to Ollama (100% free, local, open-source)
    from langchain_ollama import ChatOllama
    ollama_model = model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=temperature,
    )


class CalculationAgent:
    """Stateless calculation agent with strict tool-calling and domain guardrails."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.raw_llm = llm or get_default_llm()
        # Bind calculate tool to the model
        self.llm_with_tools = self.raw_llm.bind_tools([calculate])

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Processes a single query with optional formatted history from client storage.
        
        Args:
            query: The user's input question.
            history: Optional list of past messages formatted as:
                     [{'role': 'human'|'ai', 'content': '...'}]
            
        Returns:
            Dictionary with query, tool_called, expression, tool_result, and response.
        """
        clean_query = query.strip()
        if not clean_query:
            return {
                "query": clean_query,
                "tool_called": False,
                "expression": None,
                "tool_result": None,
                "response": "Please provide a mathematical calculation request.",
            }

        # Guardrail: Check calculation intent
        if not has_calculation_intent(clean_query):
            return {
                "query": clean_query,
                "tool_called": False,
                "expression": None,
                "tool_result": None,
                "response": REFUSAL_MESSAGE,
            }

        # Construct message payload: system prompt + formatted history + current query
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Append formatted conversation history with distinct AI vs Human roles
        if history:
            for item in history:
                role = item.get("role", "").lower()
                content = item.get("content", "").strip()
                if not content:
                    continue
                if role in ("human", "user"):
                    messages.append(HumanMessage(content=content))
                elif role in ("ai", "assistant"):
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=clean_query))

        # Invoke model
        response_msg = self.llm_with_tools.invoke(messages)

        tool_calls = getattr(response_msg, "tool_calls", None)

        # 1. Happy path: LLM invokes calculate tool
        if tool_calls:
            first_call = tool_calls[0]
            call_id = first_call.get("id", "call_1")
            raw_args = first_call.get("args", {})
            expression = raw_args.get("expression", "")

            # Execute tool strictly
            tool_output = calculate.invoke({"expression": str(expression)})

            # Pass minimal feedback to format final response
            followup_messages = messages + [
                response_msg,
                ToolMessage(tool_call_id=call_id, content=str(tool_output)),
            ]
            final_response = self.llm_with_tools.invoke(followup_messages)

            return {
                "query": clean_query,
                "tool_called": True,
                "expression": expression,
                "tool_result": tool_output,
                "response": final_response.content.strip(),
            }

        # 2. LLM did not call tool
        content = response_msg.content.strip()

        # Check if the LLM refused
        if "mathematical calculation" in content.lower() or "only assist" in content.lower():
            return {
                "query": clean_query,
                "tool_called": False,
                "expression": None,
                "tool_result": None,
                "response": REFUSAL_MESSAGE,
            }

        # 3. Fallback enforcement: If query had calculation intent but LLM attempted mental math,
        # enforce tool execution directly to guarantee that unverified calculations are NEVER output.
        extracted_expr = self._extract_math_expression(clean_query)
        tool_output = calculate.invoke({"expression": extracted_expr})

        return {
            "query": clean_query,
            "tool_called": True,
            "expression": extracted_expr,
            "tool_result": tool_output,
            "response": f"The calculated result is: {tool_output}",
        }

    @staticmethod
    def _extract_math_expression(text: str) -> str:
        """Extracts a mathematical expression from text when fallback is needed."""
        cleaned = re.sub(
            r"^(calculate|compute|evaluate|what is the value of|what is|find)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip().rstrip("?.")
        return cleaned or text
