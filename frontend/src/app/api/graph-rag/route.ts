import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

interface GraphRAGRequest {
  query: string
  limit?: number
}

export async function POST(request: NextRequest) {
  try {
    const { query, limit = 5 }: GraphRAGRequest = await request.json()

    if (!query || typeof query !== 'string') {
      return NextResponse.json(
        { error: 'Query is required and must be a string' },
        { status: 400 }
      )
    }

    // Use the new chat history manager for RAG queries
    const scriptPath = path.join(process.cwd(), 'scripts', 'chat_history_manager.py')

    // Run the Python script to search chat history
    const result = await runPythonScript(scriptPath, query, limit)

    if (result.error) {
      console.error('Chat history RAG query error:', result.error)
      return NextResponse.json(
        { error: 'Failed to retrieve chat context' },
        { status: 500 }
      )
    }

    return NextResponse.json({
      context: result.context,
      query: query,
      limit: limit
    })

  } catch (error) {
    console.error('Chat history RAG API error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

async function runPythonScript(scriptPath: string, query: string, limit: number): Promise<{context?: any[], error?: string, metadata?: any[]}> {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      resolve({ error: 'Python script timed out after 30 seconds' })
    }, 30000)

    const pythonProcess = spawn('python3', [scriptPath, 'search', query, limit.toString()])

    let stdout = ''
    let stderr = ''

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    pythonProcess.on('close', (code) => {
      clearTimeout(timeout)
      if (code === 0) {
        try {
          const results = JSON.parse(stdout)

          // Format results for the existing API contract
          const context = []
          const metadata = []

          for (const result of results) {
            // Format context for LLM consumption
            const formatted_context = `From conversation "${result.session_title}":

Role: ${result.role}
Content: ${result.content.substring(0, 400)}${result.content.length > 400 ? '...' : ''}

Relevance: ${(result.relevance_score * 100).toFixed(1)}%`

            context.push(formatted_context)

            // Enhanced metadata
            metadata.push({
              'session_id': result.session_id,
              'session_title': result.session_title,
              'message_index': result.message_index,
              'role': result.role,
              'content_preview': result.content.substring(0, 100) + "...",
              'relevance_score': result.relevance_score,
              'conversation_created': result.conversation_created
            })
          }

          resolve({
            context: context,
            metadata: metadata
          })

        } catch (parseError) {
          resolve({ error: `Failed to parse Python script output: ${String(parseError)}` })
        }
      } else {
        resolve({ error: stderr || `Python script failed with code ${code}. stdout: ${stdout}` })
      }
    })

    pythonProcess.on('error', (error) => {
      clearTimeout(timeout)
      resolve({ error: error.message })
    })
  })
}
