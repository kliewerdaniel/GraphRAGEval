#!/bin/bash
# Start Reddit GraphRAG Application
# This script sets up and starts the Reddit GraphRAG application

echo "🚀 Starting Reddit GraphRAG setup..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python3 is required but not installed. Please install Python 3.8+ first."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed. Please install Node.js 16+ first."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "❌ pip is required but not installed. Please install pip first."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm is required but not installed. Please install npm first."; exit 1; }

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $port is already in use. Please stop the service using that port or choose a different port."
        return 1
    fi
    return 0
}

# Check if required ports are available
check_port 3001 || exit 1
# Frontend typically runs on 3000 by default
check_port 3000 || check_port 3001 || exit 1
check_port 8000 || exit 1

# Backend setup
echo "🐍 Setting up Python backend..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install neo4j-graphrag[ollama] ollama PyPDF2 python-dotenv fastapi uvicorn pydantic httpx numpy matplotlib plotly pandas redis pinecone-client python-multipart scikit-learn sentence-transformers nltk rouge_score || {
    echo "❌ Failed to install Python dependencies. Please check your Python/pip installation."
    exit 1
}

# Check if required services are running
echo "🔍 Checking required services..."

# Check if Neo4j is running (optional - user may need to start it)
if ! nc -z localhost 7687 2>/dev/null; then
    echo "⚠️ Neo4j does not appear to be running on port 7687"
    echo "   Please start Neo4j before running this script:"
    echo "   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest"
    echo "   Or use your existing Neo4j installation"
    echo ""
fi

# Check if Ollama is running
if ! nc -z localhost 11434 2>/dev/null; then
    echo "⚠️ Ollama does not appear to be running on port 11434"
    echo "   Please start Ollama and ensure models are pulled:"
    echo "   ollama serve &"
    echo "   ollama pull llama3.2:3b"
    echo "   ollama pull mxbai-embed-large:latest"
    echo ""
fi

# Data ingestion check
echo "📊 Checking data ingestion..."
if [ ! -d "../reddit_export" ]; then
    echo "❌ reddit_export directory not found in parent directory"
    echo "   Please ensure the reddit_export folder exists with Reddit markdown files"
    exit 1
fi

# Check if database is already populated
echo "🔍 Checking database status..."
python3 -c "
import sys
try:
    from scripts.reddit_retriever import RedditRetriever
    retriever = RedditRetriever()
    result = retriever.retriever.driver.session().run('MATCH (r:RedditContent) RETURN count(r) as count').single()
    count = result['count'] if result else 0
    print(f'Found {count} Reddit content nodes in database')
    if count == 0:
        print('Database appears empty - will run ingestion')
        sys.exit(1)
    else:
        print('Database already populated - skipping ingestion')
        sys.exit(0)
except Exception as e:
    print(f'Error checking database: {e}')
    print('Will attempt to run ingestion')
    sys.exit(1)
" 2>/dev/null

INGESTION_NEEDED=$?

if [ $INGESTION_NEEDED -eq 1 ]; then
    echo "📥 Running Reddit data ingestion..."
    if ! python3 scripts/ingest_reddit_data.py --directory ../reddit_export --setup-indexes; then
        echo "❌ Data ingestion failed. Please check your Neo4j connection and try again."
        exit 1
    fi
    echo "✅ Data ingestion completed successfully"
else
    echo "✅ Database already contains data - skipping ingestion"
fi

# Frontend setup
echo "⚛️ Setting up Next.js frontend..."

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found. Please ensure the frontend folder exists."
    exit 1
fi

cd frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
if ! npm install --force; then
    echo "❌ Failed to install Node.js dependencies. Please check your Node.js/npm installation."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found in frontend directory"
    echo "   Please create frontend/.env with appropriate configuration"
    echo "   You can use the .env.example as a template"
fi

cd ..

# Start services in background
echo "🌐 Starting services..."

# Start FastAPI backend
echo "🐍 Starting FastAPI backend on port 8000..."
python3 main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
BACKEND_READY=false
for i in {1..30}; do
    echo "   Checking backend (attempt $i/30)..."
    if curl -s --max-time 5 http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        BACKEND_READY=true
        break
    fi
    sleep 2
done

if [ "$BACKEND_READY" = false ]; then
    echo "❌ Backend failed to start within expected time"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start Next.js frontend
echo "⚛️ Starting Next.js frontend on port 3001..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
FRONTEND_READY=false
for i in {1..20}; do
    echo "   Checking frontend (attempt $i/20)..."
    if curl -s --max-time 5 http://localhost:3001 > /dev/null 2>&1; then
        echo "✅ Frontend is ready!"
        FRONTEND_READY=true
        break
    fi
    sleep 2
done

if [ "$FRONTEND_READY" = false ]; then
    echo "❌ Frontend failed to start within expected time"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "🎉 Reddit GraphRAG application is running successfully!"
echo ""
echo "🌐 Frontend (Next.js): http://localhost:3000 (or 3001)"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "📊 Database Status:"
python3 -c "
try:
    from scripts.reddit_retriever import RedditRetriever
    retriever = RedditRetriever()
    result = retriever.retriever.driver.session().run('MATCH (r:RedditContent) RETURN count(r) as count').single()
    count = result['count'] if result else 0
    print(f'   • Reddit contents: {count}')
    result = retriever.retriever.driver.session().run('MATCH (u:RedditUser) RETURN count(u) as count').single()
    count = result['count'] if result else 0
    print(f'   • Users indexed: {count}')
    result = retriever.retriever.driver.session().run('MATCH (s:Subreddit) RETURN count(s) as count').single()
    count = result['count'] if result else 0
    print(f'   • Subreddits: {count}')
    result = retriever.retriever.driver.session().run('MATCH (t:Topic) RETURN count(t) as count').single()
    count = result['count'] if result else 0
    print(f'   • Topics identified: {count}')
except Exception as e:
    print(f'   • Error checking database: {e}')
" 2>/dev/null
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo ""
echo "💡 Try asking questions like:"
echo "   • 'What do people think about AI safety?'"
echo "   • 'Show me discussions about machine learning in r/MachineLearning'"
echo "   • 'What are the main opinions on GPT models?'"

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ Services stopped. Goodbye!"
    exit 0
}

# Set trap to cleanup on script termination
trap cleanup INT TERM

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
