import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import CalculationAgent, get_default_llm

# Configure structured logging to both app.log and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("CalculationAgent")

app = FastAPI(title="Calculation AI Agent API", version="1.0.0")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the calculation agent singleton
agent = CalculationAgent()


class ChatMessage(BaseModel):
    role: str = Field(..., description="'human' or 'ai'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    message: str = Field(..., description="Current user input")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Formatted conversation history stored in client localStorage",
    )


class ChatResponse(BaseModel):
    query: str
    tool_called: bool
    expression: Optional[str] = None
    tool_result: Optional[str] = None
    response: str


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Calculation AI Agent"}


class EvaluateRequest(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate")


@app.post("/api/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    """Direct execution endpoint for WebMCP client tools."""
    try:
        logger.info(f"WebMCP Tool Evaluation: expression='{request.expression}'")
        res = calculate.invoke({"expression": request.expression})
        logger.info(f"WebMCP Tool Output: '{res}'")
        return {"result": str(res), "expression": request.expression}
    except Exception as exc:
        logger.error(f"WebMCP Evaluation Error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # Convert Pydantic history objects to dicts for agent
        formatted_history = None
        if request.history:
            formatted_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.history
            ]

        logger.info(f"Incoming Request: '{request.message}'")
        result = agent.run(query=request.message, history=formatted_history)
        if result.get("tool_called"):
            logger.info(
                f"Tool Executed: calculate(expression='{result['expression']}') -> Result: '{result['tool_result']}'"
            )
        else:
            logger.info("No tool called (Guardrail Refusal / Non-calculation Query)")
        logger.info(f"Response: '{result['response']}'")
        return ChatResponse(**result)
    except Exception as exc:
        logger.error(f"Error processing request: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
