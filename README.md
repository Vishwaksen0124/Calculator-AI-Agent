# Strict Calculation AI Agent

A robust, domain-restricted AI Calculation Agent built in Python using **LangChain** (open-source framework) and powered by **free AI models** (Local Ollama by default, or Google Gemini Free Tier).

---

## 🎯 Key Design Principles & Guarantees

1. **Strict Tool Execution (No Mental Math)**:
   The agent is strictly forbidden from estimating, hallucinating, or calculating numbers mentally. It **must invoke the calculation tool** for any and all mathematical queries. Even if an LLM attempts to answer without calling the tool, a built-in enforcement layer intercepts and evaluates the expression via the tool so unverified answers are never returned.

2. **Domain Guardrail (Refuses Non-Math Questions)**:
   The agent refuses to answer any question that is not regarding mathematical calculations (e.g., general knowledge, coding, writing, chatting, recipes, history). It responds exclusively with:
   > *"I can only assist with mathematical calculations. Please provide a math or calculation query."*

3. **Stateless Backend with Client LocalStorage History**:
   To completely eliminate server memory bloat and hallucinations while allowing follow-up calculations (e.g. *"Now multiply that by 3"*):
   - The frontend stores conversation history in browser `localStorage`.
   - Before sending, messages are formatted with distinct roles (`human` vs `ai`).
   - The agent strictly processes this formatted history, groundings follow-up calculations via the calculation tool, and does not hallucinate.

4. **Minimal Inputs & Minimal Outputs**:
   - **Agent Input**: Clean user query string + lean history turns.
   - **Tool Input**: Exactly `{"expression": "<math expression>"}`.
   - **Tool Output**: Only the pure evaluated numerical/symbolic result string.
   - No unnecessary payloads, logs, or intermediate metadata are passed to prevent hallucination.

5. **Universal Math Capability**:
   Powered by SymPy with safe parsing, the `calculate` tool handles:
   - Arithmetic: `+`, `-`, `*`, `/`, `^`, `%`, `//`
   - Roots & Powers: `sqrt(x)`, `cbrt(x)`, `x**y`
   - Trigonometry: `sin`, `cos`, `tan`, `asin`, `acos`, `atan` (in radians or degrees)
   - Logarithms & Exponentials: `log(x, base)`, `ln(x)`, `exp(x)`
   - Combinatorics: `factorial(n)`, `gamma(x)`
   - Algebra: `solve(equation, var)`, equation simplification
   - Calculus: `diff(expr, var)`, `integrate(expr, var)`, `limit(expr, var, to)`
   - Security: Restricts dangerous builtins and code execution (`__import__`, `eval`).

6. **Web Frontend & Localhost Deployment**:
   - Built-in responsive UI served at `http://localhost:8000`.
   - Visually differentiates Human vs AI messages with role badges.
   - Renders interactive Tool Execution cards showing exact expression evaluated and output returned.
   - "Clear History" button to manage browser `localStorage`.

6. **Free AI Models**:
   - **Ollama (Default & 100% Free/Local)**: Runs locally with models like `qwen3.5:4b` or `lfm2.5:8b-a1b-q4_K_M`. Requires no API key, zero costs, and runs completely private.
   - **Google Gemini (Free-Tier API)**: Optionally switch to `gemini-2.0-flash` or `gemini-1.5-flash` with a free Gemini API key.

---

## 📁 Repository Structure

```
Calculator/
├── calculator_tool.py   # High-precision calculation engine (SymPy) & LangChain tool
├── agent.py             # CalculationAgent with strict tool enforcement & guardrails
├── main.py              # CLI interface (Interactive mode & single query mode)
├── test_agent.py        # Pytest test suite (18 unit & integration tests)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment configuration template
└── README.md            # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Requirements

Ensure you have Python 3.10+ installed.

Dependencies can be installed via:
```bash
pip install -r requirements.txt
```

### 2. Running Locally with Ollama (Free)

Make sure Ollama is running on your machine (default port `11434`):
```bash
ollama run qwen3.5:4b
```

### 3. Usage

#### Interactive Mode:
```bash
python main.py
```
```
CalcAgent> Calculate (50 * 20) + (144 / 12)
The result of (50 * 20) + (144 / 12) is **1012**.

CalcAgent> What is the capital of France?
I can only assist with mathematical calculations. Please provide a math or calculation query.

CalcAgent> exit
```

#### Single-Query CLI Mode:
```bash
python main.py "Calculate sqrt(256) + 2^4"
```

#### Verbose Mode (Shows Exact Tool Parameters & Output):
```bash
python main.py -v "Calculate (15 * 8) + log10(1000)"
```
Output:
```
==================================================
[Query]       : Calculate (15 * 8) + log10(1000)
[Tool Called] : True
[Expression]  : (15 * 8) + log10(1000)
[Tool Result] : 123
==================================================

The result is **123**.
```

#### Non-Math Rejection:
```bash
python main.py "Tell me a story about space"
```
Output:
```
I can only assist with mathematical calculations. Please provide a math or calculation query.
```

---

## 🧪 Running the Automated Test Suite

Run pytest to verify all 18 unit, security, guardrail, and statelessness tests:

```bash
pytest test_agent.py -v
```

All 18 tests will verify:
- Accurate calculation across basic arithmetic, powers, roots, trigonometry, and calculus.
- Security protection against code injection.
- Rejection of general questions (geography, history, creative writing, chat).
- Verification that the tool is strictly called.
- Verification that no conversation state is preserved across queries.

---

## ⚙️ Configuration (.env)

Copy `.env.example` to `.env` if you wish to customize options:

```bash
cp .env.example .env
```

To switch to Google Gemini (Free Tier):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```
