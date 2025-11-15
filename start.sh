#!/bin/bash

# GraphRAG Research Assistant - Complete Startup Script
# This script handles the complete setup and startup of the research assistant

set -e  # Exit on any error

echo "🚀 Starting GraphRAG Research Assistant - Complete Setup"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        exit 1
    fi
}

# Check prerequisites
log_info "Checking prerequisites..."
check_command "python3"
check_command "node"
check_command "npm"
check_command "ollama"

if command -v docker-compose &> /dev/null || command -v docker &> /dev/null && docker compose version &> /dev/null; then
    log_success "Docker/Docker Compose found"
else
    log_warning "Docker Compose not found - will skip service startup"
    SKIP_DOCKER=true
fi

# Set up Python environment
log_info "Setting up Python environment..."
if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
else
    log_success "Virtual environment already exists"
fi

source venv/bin/activate
log_info "Activating virtual environment and installing dependencies..."
pip install --quiet -r requirements.txt

# Start Docker services (if available)
if [ "$SKIP_DOCKER" != true ]; then
    log_info "Starting Docker services (Neo4j + Redis)..."
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    else
        docker compose up -d
    fi

    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 10

    # Test Neo4j connection
    log_info "Testing Neo4j connection..."
    python3 -c "
import sys
from neo4j import GraphDatabase
import os
import time

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'research2025')

for attempt in range(10):
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run('RETURN 1 as num')
        driver.close()
        print('Neo4j connection successful')
        sys.exit(0)
    except Exception as e:
        print(f'Neo4j connection attempt {attempt + 1} failed: {e}')
        time.sleep(5)

print('Neo4j connection failed after 10 attempts')
sys.exit(1)
    " || {
        log_error "Neo4j connection failed"
        exit 1
    }
else
    log_warning "Skipping Docker services - Neo4j and Redis must be running manually"
fi

# Test Ollama models
log_info "Checking Ollama models..."
ollama list | grep -q "mistral" || {
    log_info "Pulling Mistral model..."
    ollama pull mistral
}

ollama list | grep -q "nomic-embed-text" || {
    log_info "Pulling nomic-embed-text model..."
    ollama pull nomic-embed-text
}

log_success "Ollama models ready"

# Set up database indexes
log_info "Setting up Neo4j schema and indexes..."
python3 scripts/ingest_research_data.py --setup-indexes 2>/dev/null || log_warning "Index setup may have issues - check Neo4j connection"

# Generate test datasets
log_info "Generating vero-eval test datasets..."
python3 evaluation/generate_test_dataset.py --queries 25 --include-stress-tests

# Ingest sample data
log_info "Ingesting sample research papers..."
mkdir -p data/sample_papers
python3 scripts/ingest_research_data.py --directory data/research_papers

# Run initial evaluation
log_info "Running initial system evaluation..."
python3 evaluation/run_evaluation.py --dataset evaluation/datasets/research_assistant_v1.json --output evaluation/results/startup_evaluation.json

# Start FastAPI backend in background
log_info "Starting FastAPI backend..."
python3 main.py &
BACKEND_PID=$!
log_success "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
sleep 5
curl -s http://localhost:8000/api/health > /dev/null || {
    log_warning "Backend health check failed - it may still be starting"
}

# Build and start frontend
log_info "Building and starting Next.js frontend..."
cd frontend
npm install --silent

# Build for production
log_info "Building frontend for production..."
npm run build

# Start frontend in background
log_info "Starting frontend server..."
npm start &
FRONTEND_PID=$!
log_success "Frontend started (PID: $FRONTEND_PID)"
cd ..

echo ""
echo "🎉 GraphRAG Research Assistant is now running!"
echo "=========================================================="
echo ""
echo "📊 System Status:"
echo "   ✅ Python environment: Active"
echo "   ✅ Neo4j database: $(docker ps | grep -q neo4j && echo "Running" || echo "Not detected")"
echo "   ✅ Redis cache: $(docker ps | grep -q redis && echo "Running" || echo "Not detected")"
echo "   ✅ Ollama LLM: Ready"
echo "   ✅ Research papers: $(python3 -c "
from neo4j import GraphDatabase
import os
try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=('neo4j', 'research2025')
    )
    with driver.session() as session:
        result = session.run('MATCH (p:Paper) RETURN count(p) as count')
        count = result.single()['count']
        print(f'{count} indexed')
except:
    print('Connection failed')
driver.close() 2>/dev/null
")"
echo ""
echo "🌐 Access Points:"
echo "   🖥️  Frontend:     http://localhost:3000"
echo "   🔌 Backend API:  http://localhost:8000"
echo "   📊 Neo4j Browser: http://localhost:7474"
echo ""
echo "📈 Monitoring:"
echo "   📊 Evaluation Results: evaluation/results/"
echo "   📋 System Logs: Available in terminal"
echo ""
echo "🛑 To stop all services:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo "   docker-compose down (if using Docker)"
echo ""
echo "💡 Next steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Ask questions about research in the chat interface"
echo "   3. Add more PDF papers to data/research_papers/"
echo "   4. Run evaluations: python evaluation/run_evaluation.py"
echo ""
log_success "Setup complete! System is ready for research queries."

# Keep the script running to show logs
wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
