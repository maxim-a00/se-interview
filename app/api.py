from functools import lru_cache

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agent import build_agent
from app.config import get_config
from app.observability import configure_phoenix


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@lru_cache(maxsize=1)
def get_agent():
    configure_phoenix()
    return build_agent()


def create_app() -> FastAPI:
    config = get_config()
    app = FastAPI(
        title=config.app_title,
        description=config.app_description,
        version=config.app_version,
    )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        """Send a message to the agent and get a response."""
        result = get_agent().invoke({"messages": [HumanMessage(content=request.message)]})
        return ChatResponse(response=result["messages"][-1].content)

    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()
