#!/bin/bash

echo "🔬 Setting up Research Assistant GraphRAG System with vero-eval"

# 1. Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing Ollama..."
    # On macOS, install via Homebrew
    if command -v brew &> /dev/null; then
        brew install ollama
    else
        echo "Please install Ollama from https://ollama.ai"
        exit 1
    fi
fi

# 2. Activate virtual environment and install dependencies
echo "📦 Installing Python dependencies..."
source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Start Docker services
echo "🐳 Starting Docker services..."
if command -v docker-compose &> /dev/null || command -v docker &> /dev/null && docker compose version &> /dev/null; then
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    else
        docker compose up -d
    fi
    echo "⏳ Waiting for Neo4j to start (this may take a minute)..."
    sleep 30
else
    echo "⚠️  Docker Compose not found. Please start Neo4j manually:"
    echo "   docker-compose up -d"
    echo "   Or install Docker Desktop"
fi

# 4. Check Neo4j connection
echo "🔗 Checking Neo4j connection..."
python3 -c "
from neo4j import GraphDatabase
import os
try:
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USERNAME', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'research2025'))
    )
    with driver.session() as session:
        result = session.run('RETURN 1 as num')
        print('✅ Neo4j connection successful')
    driver.close()
except Exception as e:
    print(f'⚠️  Neo4j connection failed: {e}')
    print('   Please ensure Neo4j is running')
"

# 5. Pull Ollama models
echo "🤖 Setting up Ollama models..."
ollama pull mistral
ollama pull nomic-embed-text

# 6. Test Ollama
echo "🧪 Testing Ollama models..."
python3 -c "
import ollama
try:
    response = ollama.generate(model='mistral', prompt='Hello', options={'num_predict': 10})
    print('✅ Ollama mistral model ready')
except Exception as e:
    print(f'⚠️  Ollama test failed: {e}')
"

# 7. Create vector indexes (requires Neo4j connection)
echo "📊 Creating Neo4j graph schema and indexes..."
python3 scripts/ingest_research_data.py --setup-indexes 2>/dev/null || echo "⚠️  Index setup skipped (Neo4j may not be ready)"

# 8. Generate sample data
echo "📚 Creating sample data..."
mkdir -p data/sample_papers
# Create a sample PDF or skip if no PDFs exist
echo "⚠️  Add PDF files to data/research_papers/ to test ingestion"

# 9. Run initial evaluation
echo "🧪 Running initial evaluation..."
python3 evaluation/run_evaluation.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Start the system: python scripts/reasoning_agent.py --query 'your test query'"
echo "   2. Add research papers: cp your_papers/*.pdf data/research_papers/"
echo "   3. Ingest papers: python scripts/ingest_research_data.py"
echo "   4. Run full evaluation: python evaluation/run_evaluation.py"
echo ""
echo "🔗 Access points (after starting web frontend):"
echo "   - Frontend: http://localhost:3000 (when Next.js is set up)"
echo "   - Neo4j Browser: http://localhost:7474"
echo ""
echo "📊 Evaluation results: evaluation/results/"
echo "📈 Performance dashboard: evaluation/dashboard.html"
echo ""
echo "🚀 Ready to build and evaluate your research assistant!"
