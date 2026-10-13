"""calculator_tool.py

High-precision, safe mathematical calculation engine and LangChain tool.
Evaluates arithmetic, algebra, trigonometry, calculus, and functions using SymPy.
Passes only the necessary mathematical expression and returns only the calculated result.
"""

import re
from typing import Any, Dict
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
from langchain_core.tools import tool

# Standard parsing transformations to support natural math (e.g., 2x, 2^3)
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Whitelist safe mathematical functions and symbols from SymPy
SAFE_GLOBALS: Dict[str, Any] = {
    k: v for k, v in sympy.__dict__.items() if not k.startswith("_")
}
SAFE_GLOBALS["__builtins__"] = None

# Additional mathematical aliases
SAFE_GLOBALS.update({
    "ln": sympy.log,
    "log10": lambda x: sympy.log(x, 10),
    "log2": lambda x: sympy.log(x, 2),
    "cbrt": sympy.cbrt,
    "deg2rad": lambda deg: deg * sympy.pi / 180,
    "rad2deg": lambda rad: rad * 180 / sympy.pi,
})


def clean_expression(raw_expr: str) -> str:
    """Sanitizes and normalizes the mathematical expression string."""
    expr = raw_expr.strip()
    
    # Block private/dunder attribute access
    if "__" in expr:
        raise ValueError("Disallowed expression syntax")

    # Strip common enclosing characters or markdown code ticks
    if expr.startswith("`") and expr.endswith("`"):
        expr = expr.strip("`")
    
    # Replace common symbols with Python/SymPy operators
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    
    # Handle percentage: e.g. "20% * 50" -> "(20/100) * 50"
    expr = re.sub(r"(\d+(\.\d+)?)\s*%", r"(\1/100)", expr)
    
    return expr


def evaluate_math(expression: str) -> str:
    """Evaluates any mathematical expression safely and accurately.
    
    Args:
        expression: The raw mathematical expression string.
        
    Returns:
        The exact evaluated result as a concise string.
    """
    try:
        cleaned = clean_expression(expression)
        if not cleaned:
            return "Error: Empty expression"

        parsed = parse_expr(
            cleaned,
            global_dict=SAFE_GLOBALS,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

        # If it's a constant numerical expression, evaluate it numerically
        if hasattr(parsed, "free_symbols") and not parsed.free_symbols and hasattr(parsed, "evalf"):
            val = parsed.evalf()
            if val.is_integer:
                return str(int(val))
            # Format floating point numbers to clean precision without trailing zeroes
            formatted = f"{float(val):.10g}"
            return formatted

        return str(parsed)
    except Exception as exc:
        return f"Error: Unable to calculate expression ({str(exc)})"


@tool
def calculate(expression: str) -> str:
    """Evaluates any mathematical expression accurately.
    
    Use this tool for ANY and ALL calculations (arithmetic, algebra, roots, powers,
    trigonometry, logarithms, calculus, equations). 
    Pass ONLY the pure mathematical expression string to 'expression'.
    """
    return evaluate_math(expression)
