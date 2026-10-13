"""app.py

FastAPI backend server for the Strict Calculation Agent.
Serves static frontend and API endpoints for chat with local storage history support.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import CalculationAgent, get_default_llm

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

        result = agent.run(query=request.message, history=formatted_history)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
