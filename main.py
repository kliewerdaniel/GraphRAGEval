"""
FastAPI backend for the Research Assistant application
Following the architecture outlined in the README
"""
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from datetime import datetime
import json

# Import our custom modules
from scripts.reasoning_agent import PersonaReasoningAgent
from scripts.hybrid_retriever import HybridRetriever
from scripts.ingest_research_data import ResearchGraphBuilder
from evaluation.run_evaluation import Evaluator

# Initialize components
app = FastAPI(title="Research Assistant API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Allow Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized on startup)
reasoning_agent = None
retriever = None
evaluator = None

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []
    persona_override: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    context_used: List[Dict[str, Any]]
    quality_grade: float
    retrieval_method: Optional[str]
    retrieval_performed: bool
    sources: List[Dict[str, str]]

class IngestionRequest(BaseModel):
    directory: str = "data/research_papers"
    recreate_indexes: bool = False

class EvaluationRequest(BaseModel):
    dataset_path: Optional[str] = None
    output_path: Optional[str] = "evaluation/results/api_evaluation.json"

class SystemStatus(BaseModel):
    neo4j_connected: bool
    ollama_ready: bool
    redis_connected: bool
    paper_count: int
    evaluation_count: int

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global reasoning_agent, retriever, evaluator

    try:
        reasoning_agent = PersonaReasoningAgent()
        retriever = HybridRetriever()
        evaluator = Evaluator(
            test_dataset_path=Path("evaluation/datasets/research_assistant_v1.json"),
            trace_db_path=Path("evaluation/trace.db")
        )
        print("✓ All components initialized")
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")

@app.get("/api/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat")
async def chat(request: QueryRequest) -> QueryResponse:
    """Main chat endpoint with GraphRAG"""
    if not reasoning_agent:
        raise HTTPException(status_code=503, detail="Reasoning agent not initialized")

    try:
        # Generate response
        result = reasoning_agent.generate_response(
            request.query,
            request.chat_history
        )

        # Format sources for frontend
        sources = []
        for doc in result['context_used']:
            sources.append({
                'title': str(doc.get('title', 'Unknown')),
                'authors': ', '.join(doc.get('authors', [])) if doc.get('authors') else 'Unknown',
                'year': str(doc.get('year', 'Unknown')),
                'relevance_score': f"{doc.get('relevance_score', 0.0):.3f}",
                'retrieval_method': str(doc.get('retrieval_method', 'unknown'))
            })

        return QueryResponse(
            response=result['response'],
            context_used=result['context_used'],
            quality_grade=result['quality_grade'],
            retrieval_method=result.get('retrieval_method'),
            retrieval_performed=result.get('retrieval_performed', False),
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/api/search")
async def search_papers(query: str, limit: int = 10):
    """Direct paper search endpoint"""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        results = retriever.retrieve_context(query, limit=limit)
        return {"results": results, "query": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/ingest")
async def ingest_papers(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Ingest research papers"""
    try:
        # Run ingestion in background
        background_tasks.add_task(run_ingestion, request.directory, request.recreate_indexes)
        return {"message": f"Started ingestion from {request.directory}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/evaluate")
async def run_evaluation_endpoint(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Run evaluation"""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")

    try:
        # Run evaluation in background
        background_tasks.add_task(run_evaluation_task, request.dataset_path, request.output_path)
        return {"message": "Started evaluation"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/status")
async def get_system_status() -> SystemStatus:
    """Get comprehensive system status"""
    try:
        # Check Neo4j connection
        neo4j_connected = False
        paper_count = 0
        try:
            if retriever and retriever.driver:
                with retriever.driver.session() as session:
                    result = session.run("MATCH (p:Paper) RETURN count(p) as count")
                    paper_count = result.single()["count"]
                    neo4j_connected = True
        except:
            pass

        # Check Redis
        redis_connected = False
        try:
            import redis
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            r.ping()
            redis_connected = True
        except:
            pass

        # Check Ollama (simplified)
        ollama_ready = True  # Assume ready if service started

        # Get evaluation count
        evaluation_count = 0
        try:
            trace_file = Path("evaluation/trace.db")
            if trace_file.exists():
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                    evaluation_count = len(data)
        except:
            pass

        return SystemStatus(
            neo4j_connected=neo4j_connected,
            ollama_ready=ollama_ready,
            redis_connected=redis_connected,
            paper_count=paper_count,
            evaluation_count=evaluation_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/api/evaluation-results")
async def get_evaluation_results():
    """Get latest evaluation results"""
    try:
        results_path = Path("evaluation/results/evaluation_output.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            return {"error": "No evaluation results found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")

# Background tasks
def run_ingestion(directory: str, recreate_indexes: bool):
    """Run paper ingestion in background"""
    print(f"Starting background ingestion from {directory}")

    try:
        builder = ResearchGraphBuilder()
        papers_dir = Path(directory)

        if recreate_indexes:
            # Drop existing indexes first
            try:
                with builder.driver.session() as session:
                    session.run("DROP INDEX paper_abstracts IF EXISTS")
            except:
                pass
            builder.create_vector_indexes()

        builder.ingest_directory(papers_dir)

        if recreate_indexes:
            builder.create_vector_indexes()

        print("✓ Background ingestion completed")

    except Exception as e:
        print(f"✗ Background ingestion failed: {e}")

def run_evaluation_task(dataset_path: str = None, output_path: str = "evaluation/results/api_evaluation.json"):
    """Run evaluation in background"""
    print("Starting background evaluation")

    try:
        if not evaluator:
            print("✗ Evaluator not initialized")
            return

        # Use default sample queries if no dataset provided
        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
                queries = dataset.get('queries', [])
        else:
            queries = [
                {
                    'query': 'What are the main approaches to attention mechanisms in deep learning?',
                    'persona': 'researcher',
                    'ground_truth_chunk_ids': ['attention paper'],
                    'reference_answer': 'Attention mechanisms in deep learning...',
                    'complexity_score': 0.7
                },
                {
                    'query': 'How do transformer models handle long-range dependencies?',
                    'persona': 'student',
                    'ground_truth_chunk_ids': ['transformer paper'],
                    'reference_answer': 'Transformer models use...',
                    'complexity_score': 0.6
                }
            ]

        results = evaluator.run_evaluation(
            queries=queries,
            output_path=Path(output_path)
        )

        print("✓ Background evaluation completed")

    except Exception as e:
        print(f"✗ Background evaluation failed: {e}")

# Mount static files if they exist (for production)
frontend_path = Path("frontend/build")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
