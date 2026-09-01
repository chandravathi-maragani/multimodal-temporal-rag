import asyncio  
import json
import logging
import sqlite3
import time
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag_backend import (
    get_cached_metadata,     # Used for GET /metadata route
    query_rag_system_async   # Used for POST /query route
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(title="Multimodal Temporal RAG API")


# --- Helper Function to Stop 'TextAccessor' & Type Errors ---
def to_plain_string(val) -> str:
    """Safely converts strings, lists, dicts, or LangChain objects into pure text."""
    if val is None:
        return ""
    if hasattr(val, "content"):  # Handles LangChain AIMessage / BaseMessage objects
        return str(val.content)
    if isinstance(val, list):
        return "\n\n".join([to_plain_string(item) for item in val])
    return str(val)


# --- Database Logging ---
def init_db():
    conn = sqlite3.connect("rag_metrics.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            latency_seconds REAL,
            context_relevance INTEGER,
            faithfulness INTEGER,
            answer_relevance INTEGER,
            eval_details TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


def save_log_to_db(latency: float, eval_data: dict):
    try:
        conn = sqlite3.connect("rag_metrics.db", timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluation_logs (
                latency_seconds, context_relevance, faithfulness, answer_relevance, eval_details
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            round(latency, 4),
            eval_data["context_relevance"]["score"],
            eval_data["faithfulness"]["score"],
            eval_data["answer_relevance"]["score"],
            json.dumps(eval_data)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save metrics log: {str(e)}")


# --- Global Exception Handler (Hides Raw Stack Traces) ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal error at {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An internal server error occurred."}
    )


# --- Schemas ---
class QueryRequest(BaseModel):
    question: str
    filename: str | None = None
    topic: str | None = None
    date: str | None = None

class MetricEval(BaseModel):
    score: int = Field(description="Rating from 1 to 5")
    reason: str = Field(description="Brief reason")

class RAGTriadEvaluation(BaseModel):
    context_relevance: MetricEval
    faithfulness: MetricEval
    answer_relevance: MetricEval


# --- LLM Evaluator Setup ---
judge_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0
).with_structured_output(RAGTriadEvaluation)

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI Evaluator specializing in the RAG Triad framework. Evaluate strictly based on the provided inputs."),
    ("human", """User Question: {question}

Retrieved Context: {context}

Generated Answer: {answer}

Evaluate across Context Relevance, Faithfulness, and Answer Relevance (scores 1-5).""")
])


async def run_llm_judge(question: str, context: str, answer: str, latency: float):
    try:
        eval_chain = EVAL_PROMPT | judge_llm
        result: RAGTriadEvaluation = await eval_chain.ainvoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        save_log_to_db(latency, result.model_dump())
        logger.info(f"Evaluation finished and saved ({round(latency, 4)}s)")
    except Exception as e:
        logger.error(f"LLM Judge execution failed: {str(e)}")


# --- API Endpoint ---
@app.post("/query")
async def handle_query(request: QueryRequest, background_tasks: BackgroundTasks):
    # Sanitize inputs: strip whitespace and ignore default Swagger placeholder strings ("string")
    def clean_filter(val: str | None) -> str | None:
        if not val:
            return None
        cleaned = str(val).strip()
        if cleaned.lower() in ("", "string", "none", "null"):
            return None
        return cleaned

    filename_filter = clean_filter(request.filename)
    topic_filter = clean_filter(request.topic)
    date_filter = clean_filter(request.date)
    
    start_time = time.perf_counter()
    
    try:
        pipeline_output = await asyncio.wait_for(
            query_rag_system_async(
                user_query=request.question,
                filename_filter=filename_filter,
                topic_filter=topic_filter,
                date_filter=date_filter
            ), 
            timeout=10.0
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"status": "error", "message": "Query processing timed out."})
    except ConnectionError:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Knowledge store unreachable."})
    
    latency = time.perf_counter() - start_time
    
    if pipeline_output == "FILTER_MISMATCH":
        return {"status": "error", "message": "No matching notes found.", "latency_seconds": round(latency, 4)}
    
    # Safely convert output fields into clean strings
    if isinstance(pipeline_output, dict):
        answer_str = to_plain_string(pipeline_output.get("answer", ""))
        context_str = to_plain_string(pipeline_output.get("context", ""))
    else:
        answer_str = to_plain_string(pipeline_output)
        context_str = ""

    # Run judge asynchronously in the background
    background_tasks.add_task(
        run_llm_judge,
        question=to_plain_string(request.question),
        context=context_str,
        answer=answer_str,
        latency=latency
    )
    
    return {
        "status": "success", 
        "answer": answer_str,
        "latency_seconds": round(latency, 4)
    }

@app.get("/metadata")
def get_metadata():
    return get_cached_metadata()
