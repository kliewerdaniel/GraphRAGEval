import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import { existsSync, writeFileSync, unlinkSync } from 'fs'
import path from 'path'

interface ChatSession {
  id: string
  title: string
  messages: Array<{
    role: 'user' | 'assistant'
    content: string
    error?: string
    retryCount?: number
    isThinking?: boolean
  }>
  model: string
  promptId: string
  createdAt: number
  updatedAt: number
}

async function runPythonScript(scriptPath: string, command: string, data?: any): Promise<{processed_count?: number, error?: string, message?: string}> {
  return new Promise((resolve) => {
    // Set timeout to prevent hanging
    const timeout = setTimeout(() => {
      resolve({ error: 'Python script timed out after 60 seconds' })
    }, 60000)

    const args = [scriptPath, command]
    if (data) {
      args.push(JSON.stringify(data))
    }

    const pythonProcess = spawn('python3', args)

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
          // Try to parse JSON output
          const output = stdout.trim()
          if (output) {
            const result = JSON.parse(output)
            resolve(result)
          } else {
            resolve({ processed_count: 0 })
          }
        } catch {
          // If not JSON, treat as plain text success message
          resolve({
            processed_count: parseInt(stdout) || 0,
            message: stdout.trim()
          })
        }
      } else {
        resolve({
          error: stderr || `Python script failed with code ${code}. stdout: ${stdout}`
        })
      }
    })

    pythonProcess.on('error', (error) => {
      clearTimeout(timeout)
      resolve({ error: error.message })
    })
  })
}

export async function POST(request: NextRequest) {
  try {
    console.log('ingest-chat-history API called')

    const requestBody = await request.json()
    console.log('Received request body keys:', Object.keys(requestBody))

    const { chatSessions }: { chatSessions: ChatSession[] } = requestBody

    if (!chatSessions || !Array.isArray(chatSessions)) {
      console.log('Invalid chatSessions:', chatSessions)
      return NextResponse.json(
        { error: 'chatSessions array is required' },
        { status: 400 }
      )
    }

    // Filter and validate chat sessions structure
    const validSessions = []
    for (const session of chatSessions) {
      if (!session.id || !session.messages || !Array.isArray(session.messages)) {
        console.log(`Skipping invalid session structure: missing id or messages array`)
        continue
      }

      // Skip sessions with no messages at all
      if (session.messages.length === 0) {
        console.log(`Skipping session ${session.id} - session has no messages`)
        continue
      }

      // Filter out empty or whitespace-only messages
      const validMessages = session.messages.filter(message =>
        message.role &&
        message.content &&
        message.content.trim().length > 0
      )

      if (validMessages.length === 0) {
        console.log(`Skipping session ${session.id} - no valid messages after filtering`)
        continue
      }

      validSessions.push({
        ...session,
        messages: validMessages
      })
    }

    if (validSessions.length === 0) {
      console.log('No valid sessions found after filtering')
      return NextResponse.json({
        error: 'No valid chat history to ingest - please ensure chat sessions include their message arrays for proper persistence',
        details: 'All provided sessions contained empty or invalid message arrays',
        processed_count: 0,
        sessions_processed: chatSessions.length
      }, { status: 400 })
    }

    console.log(`Starting ingestion of ${validSessions.length} valid chat sessions (filtered from ${chatSessions.length} total)...`)

    // Run the new chat history manager
    const scriptPath = path.join(process.cwd(), 'scripts', 'chat_history_manager.py')
    console.log(`Using script: ${scriptPath}`)

    // Check if script exists
    if (!existsSync(scriptPath)) {
      console.error(`Script not found: ${scriptPath}`)
      return NextResponse.json(
        { error: 'Chat history manager script not found' },
        { status: 500 }
      )
    }

    // For ingestion, we'll call the CLI with 'ingest' command
    // Since the CLI doesn't support stdin ingestion, we'll simulate it differently
    // Let's create a temporary approach - save to a temp file and process

    const tempDataPath = path.join(process.cwd(), 'temp_chat_sessions.json')

    try {
      // Write sessions to temp file
      writeFileSync(tempDataPath, JSON.stringify(validSessions))

      // Run the script to process the temp file
      const result = await runPythonScript(scriptPath, 'ingest', tempDataPath)

      // Clean up temp file
      if (existsSync(tempDataPath)) {
        unlinkSync(tempDataPath)
      }

      if (result.error) {
        console.error('Chat history ingestion error:', result.error)
        return NextResponse.json(
          { error: 'Failed to ingest chat history', details: result.error },
          { status: 500 }
        )
      }

    } catch (fsError) {
      console.error('Error handling temp file:', fsError)
      return NextResponse.json(
        { error: 'Failed to process chat sessions', details: String(fsError) },
        { status: 500 }
      )
    }

    return NextResponse.json({
      success: true,
      message: 'Chat history stored as JSON and embeddings generated successfully',
      processed_count: validSessions.length,
      sessions_processed: chatSessions.length
    })

  } catch (error) {
    console.error('Chat history ingestion API error:', error)
    return NextResponse.json(
      { error: 'Internal server error', details: String(error) },
      { status: 500 }
    )
  }
}
