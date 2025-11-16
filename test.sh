#!/bin/bash
# Test script for Reddit GraphRAG application

echo "🧪 Testing Reddit GraphRAG Application"

# Check if services are running
echo "🔍 Checking if services are running..."

if ! curl -s --max-time 5 http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "❌ Backend API is not running on port 8000"
    echo "   Please start the application with ./start.sh first"
    exit 1
else
    echo "✅ Backend API is running"
fi

if ! curl -s --max-time 5 http://localhost:3001 > /dev/null 2>&1; then
    echo "❌ Frontend is not running on port 3001"
    echo "   Please start the application with ./start.sh first"
    exit 1
else
    echo "✅ Frontend is running"
fi

# Test API endpoints
echo ""
echo "🧪 Testing API endpoints..."

# Health check
echo "📡 Testing /api/health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/health)
if [[ $HEALTH_RESPONSE == *'"healthy"'* ]]; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed: $HEALTH_RESPONSE"
fi

# Status check
echo "📊 Testing /api/status..."
STATUS_RESPONSE=$(curl -s http://localhost:8000/api/status)
if [[ $STATUS_RESPONSE == *'"reddit_count"'* ]]; then
    REDDIT_COUNT=$(echo $STATUS_RESPONSE | grep -o '"reddit_count":[0-9]*' | cut -d':' -f2)
    echo "✅ Status check passed - $REDDIT_COUNT Reddit items indexed"
else
    echo "❌ Status check failed: $STATUS_RESPONSE"
fi

# Test chat endpoint
echo "💬 Testing /api/chat..."
CHAT_RESPONSE=$(curl -s -X POST \
  http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the most discussed topic?", "chat_history": []}')

if [[ $CHAT_RESPONSE == *'"response"'* ]]; then
    echo "✅ Chat endpoint working"
    RESPONSE_PREVIEW=$(echo $CHAT_RESPONSE | grep -o '"response":"[^"]*' | cut -d'"' -f4 | cut -c1-50)
    echo "   Response preview: $RESPONSE_PREVIEW..."
else
    echo "❌ Chat endpoint failed: $CHAT_RESPONSE"
fi

# Test search endpoint
echo "🔍 Testing /api/search..."
SEARCH_RESPONSE=$(curl -s "http://localhost:8000/api/search?query=AI&limit=3")
if [[ $SEARCH_RESPONSE == *'"results"'* ]]; then
    echo "✅ Search endpoint working"
else
    echo "❌ Search endpoint failed: $SEARCH_RESPONSE"
fi

echo ""
echo "🎯 Test Results Summary:"
echo "=========================="

if [[ $STATUS_RESPONSE == *'"reddit_count":[1-9]'* ]]; then
    echo "✅ Database populated with content"
else
    echo "⚠️  Database appears empty - run ingestion first:"
    echo "   python3 scripts/ingest_reddit_data.py --directory ../reddit_export --setup-indexes"
fi

if [[ $CHAT_RESPONSE == *'"sources"'* ]]; then
    SOURCE_COUNT=$(echo $CHAT_RESPONSE | grep -o '"sources":\[[^]]*\]' | grep -o '"title"' | wc -l)
    echo "✅ Chat responses include $SOURCE_COUNT sources"
else
    echo "❌ Chat responses missing source citations"
fi

echo ""
echo "🌟 Application URLs:"
echo "• Frontend: http://localhost:3001"
echo "• Backend API: http://localhost:8000"
echo "• API Docs: http://localhost:8000/docs"

echo ""
echo "💡 Try these sample queries:"
echo "• 'What do people think about AI safety?'"
echo "• 'Show me discussions about machine learning'"
echo "• 'What are common opinions about GPT models?'"

echo ""
echo "✅ Testing complete!"
