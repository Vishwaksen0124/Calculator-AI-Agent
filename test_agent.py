"""test_agent.py

Test suite for the Calculation Agent and Calculator Tool.
Verifies:
1. Tool accuracy across arithmetic, advanced math, and symbolic expressions.
2. Security protection against arbitrary code execution.
3. Domain guardrails: strict refusal of non-calculation queries.
4. Tool enforcement: calculation tool is strictly invoked for math requests.
5. Statelessness: no memory or history is retained between queries.
"""

import pytest
from calculator_tool import calculate, evaluate_math
from agent import CalculationAgent, REFUSAL_MESSAGE, has_calculation_intent


# ---------------------------------------------------------------------------
# 1. Calculator Tool Tests
# ---------------------------------------------------------------------------

def test_tool_basic_arithmetic():
    assert evaluate_math("25 * 48") == "1200"
    assert evaluate_math("100 / 4 + 15") == "40"
    assert evaluate_math("2^10") == "1024"
    assert evaluate_math("20% * 150") == "30"


def test_tool_advanced_math():
    assert evaluate_math("sqrt(144)") == "12"
    assert evaluate_math("cbrt(27)") == "3"
    assert evaluate_math("factorial(5)") == "120"
    assert evaluate_math("log10(1000)") == "3"
    assert evaluate_math("sin(pi/2)") == "1"


def test_tool_algebra_and_calculus():
    assert evaluate_math("solve(x**2 - 25, x)") == "[-5, 5]"
    assert evaluate_math("diff(3*x**2 + 2*x, x)") == "6*x + 2"
    assert evaluate_math("integrate(x**2, x)") == "x**3/3"


def test_tool_security():
    malicious = '__import__("os").system("echo hacked")'
    result = evaluate_math(malicious)
    assert "Error" in result
    assert "hacked" not in result


def test_tool_minimal_schema():
    # Verify calculate tool expects only 'expression' argument
    assert "expression" in calculate.args
    assert len(calculate.args) == 1
    # Verify output is a string
    res = calculate.invoke({"expression": "10 + 20"})
    assert res == "30"


# ---------------------------------------------------------------------------
# 2. Guardrail & Intent Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_intent", [
    ("What is 15 * 12?", True),
    ("Calculate the square root of 64", True),
    ("Solve x + 5 = 10", True),
    ("sin(pi / 3)", True),
    ("What is the capital of France?", False),
    ("Write a poem about nature", False),
    ("How do I make pancakes?", False),
    ("Tell me a funny joke", False),
    ("Who was Napoleon Bonaparte?", False),
])
def test_calculation_intent_detection(query, expected_intent):
    assert has_calculation_intent(query) is expected_intent


# ---------------------------------------------------------------------------
# 3. Agent Integration Tests (Using Free Local LLM)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent():
    return CalculationAgent()


def test_agent_refuses_non_calculation_query(agent):
    res = agent.run("What is the capital of France?")
    assert res["tool_called"] is False
    assert res["tool_result"] is None
    assert res["response"] == REFUSAL_MESSAGE


def test_agent_refuses_general_conversation(agent):
    res = agent.run("Write a python script to download files")
    assert res["tool_called"] is False
    assert res["response"] == REFUSAL_MESSAGE


def test_agent_executes_calculation_via_tool(agent):
    res = agent.run("Calculate (120 / 4) + 15 * 2")
    assert res["tool_called"] is True
    assert res["tool_result"] == "60"
    assert "60" in res["response"]


def test_agent_statelessness(agent):
    """Verifies that subsequent calls do not carry forward prior conversation state."""
    # First query
    res1 = agent.run("Calculate 100 * 2")
    assert res1["tool_result"] == "200"

    # Second query asks for a completely different calculation
    res2 = agent.run("Calculate 50 + 50")
    assert res2["tool_result"] == "100"

    # Non-math query immediately refused without referring to past calculations
    res3 = agent.run("What was the result of our previous calculation?")
    assert res3["tool_called"] is False
    assert res3["response"] == REFUSAL_MESSAGE


def test_agent_with_formatted_history(agent):
    """Verifies that formatted client history differentiating human vs AI is processed correctly."""
    history = [
        {"role": "human", "content": "Calculate 20 * 5"},
        {"role": "ai", "content": "The result is 100."}
    ]
    # Follow-up that references the history
    res = agent.run("Now multiply that by 4", history=history)
    assert res["tool_called"] is True
    assert res["tool_result"] == "400"

