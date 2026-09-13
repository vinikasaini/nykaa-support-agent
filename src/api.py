from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import run_agent


app = FastAPI(
    title="Nykaa Support Agent API",
    description="FastAPI deployment for the Nykaa customer support LangGraph agent",
    version="1.0.0"
)


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base")


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Customer question"
    )


class AgentResponse(BaseModel):
    query: str
    intent: str
    answer: str
    source: Optional[str] = None
    similarity: Optional[float] = None
    order_status: Optional[str] = None
    order_value_inr: Optional[int] = None
    escalation_score: Optional[float] = None
    recommended_escalation: Optional[bool] = None


class AddDocumentRequest(BaseModel):
    filename: str = Field(
        ...,
        min_length=1,
        description="Name of the knowledge-base text file"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Content of the knowledge-base document"
    )


class AddDocumentResponse(BaseModel):
    message: str
    filename: str


@app.get("/")
def root():
    return {
        "message": "Nykaa Support Agent API is running",
        "endpoints": [
            "POST /ask",
            "POST /add-document"
        ]
    }


@app.post("/ask", response_model=AgentResponse)
def ask_agent(request: AskRequest):

    try:
        result = run_agent(request.query)

        structured_response = result["structured_response"]

        return AgentResponse(
            query=structured_response["query"],
            intent=structured_response["intent"],
            answer=structured_response["answer"],
            source=structured_response.get("source"),
            similarity=structured_response.get("similarity"),
            order_status=structured_response.get("order_status"),
            order_value_inr=structured_response.get("order_value_inr"),
            escalation_score=structured_response.get("escalation_score"),
            recommended_escalation=structured_response.get(
                "recommended_escalation"
            )
        )

    except Exception as error:
        print("\nAPI Error:")
        print(str(error))

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@app.post(
    "/add-document",
    response_model=AddDocumentResponse
)
def add_document(request: AddDocumentRequest):

    filename = Path(request.filename).name

    if not filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt knowledge-base files are allowed."
        )

    file_path = KNOWLEDGE_BASE_PATH / filename

    try:
        KNOWLEDGE_BASE_PATH.mkdir(
            parents=True,
            exist_ok=True
        )

        file_path.write_text(
            request.content,
            encoding="utf-8"
        )

        return AddDocumentResponse(
            message="Document added successfully.",
            filename=filename
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )