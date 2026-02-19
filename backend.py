from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from chatbot import get_response_from_ai_agents


# Define request body schema
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    messages: List[str]
    allow_scarch: bool  # consider renaming to `allow_search`


ALLOWED_MODEL_NAMES = ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"]

app = FastAPI(title="Langgraph AI Agent")


@app.post("/chat")
def chat_endpoint(request: RequestState):  # <-- lowercase 'request' here
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name"}

    # Get the last user query
    query = request.messages[-1] if request.messages else "Hello"

    # Call your AI agent
    response = get_response_from_ai_agents(
        llm_id=request.model_name,
        allow_search=request.allow_scarch,  # typo in the original field name
        query=query,
        provider=request.model_provider
    )

    return {"response": response}
if __name__ =="__main__":
    import uvicorn
    uvicorn.run("backend:app",host="127.0.0.1",port=8000,reload=True)

