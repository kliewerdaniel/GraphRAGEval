import { NextRequest } from 'next/server'
import { getPromptById } from '@/lib/prompts-data'
import { spawn } from 'child_process'
import path from 'path'

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

// API route for chat functionality with Ollama integration
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { message, messages, model, promptId, graphRAG = false } = body

    // Support both single message and conversation messages
    let conversationMessages: ChatMessage[]
    if (messages && Array.isArray(messages)) {
      // Full conversation provided
      conversationMessages = messages
    } else if (message) {
      // Single message, create conversation
      conversationMessages = [{ role: 'user', content: message }]
    } else {
      return new Response('Either message or messages array is required', { status: 400 })
    }

    // Get the selected system prompt
    const selectedPromptId = promptId || 'default'
    const systemPromptData = getPromptById(selectedPromptId)

    if (!systemPromptData) {
      console.error('System prompt not found:', selectedPromptId)
      return new Response('System prompt not found', { status: 500 })
    }

    let enhancedSystemPrompt = systemPromptData.content

    // Find the last user message for graph RAG query
    const lastUserMessage = [...conversationMessages].reverse().find(m => m.role === 'user')?.content || ''

    // If Graph RAG is enabled, retrieve relevant context from chat history
    if (graphRAG) {
      try {
        // Call the new chat history manager for retrieving relevant conversation context
        const chatHistoryScriptPath = path.join(process.cwd(), 'scripts', 'chat_history_manager.py')

        const graphResult = await new Promise<{context?: any[], error?: string}>((resolve) => {
          const pythonProcess = spawn('python3', [chatHistoryScriptPath, 'search', lastUserMessage, '3'])

          let stdout = ''
          let stderr = ''

          pythonProcess.stdout.on('data', (data: Buffer) => {
            stdout += data.toString()
          })

          pythonProcess.stderr.on('data', (data: Buffer) => {
            stderr += data.toString()
          })

          pythonProcess.on('close', (code: number | null) => {
            if (code === 0) {
              try {
                const results = JSON.parse(stdout)
                // Format results similar to the old system for compatibility
                const context = []
                for (const result of results) {
                  const formatted_context = `From conversation "${result.session_title}":

Role: ${result.role}
Content: ${result.content.substring(0, 400)}${result.content.length > 400 ? '...' : ''}

Relevance: ${(result.relevance_score * 100).toFixed(1)}%`
                  context.push(formatted_context)
                }
                resolve({ context })
              } catch {
                resolve({ error: 'Failed to parse chat history search output' })
              }
            } else {
              resolve({ error: stderr || 'Chat history search script failed' })
            }
          })

          pythonProcess.on('error', (error: Error) => {
            resolve({ error: error.message })
          })
        })

        if (graphResult.context && graphResult.context.length > 0) {
          // Enhanced system prompt with conversation context
          enhancedSystemPrompt = `${systemPromptData.content}

You have access to relevant context from previous conversations:

${graphResult.context.map((ctx: string, i: number) => `[${i+1}] ${ctx}`).join('\n\n')}

Use this context to maintain continuity and reference previous discussions when relevant.`
        } else {
          // No context found, still indicate we tried
          enhancedSystemPrompt = `${systemPromptData.content}

(Note: No relevant conversation context found for this query.)`
        }
      } catch (error) {
        console.warn('Chat history search error:', error)
        enhancedSystemPrompt = `${systemPromptData.content}

(Note: Conversation context retrieval failed, proceeding without additional context.)`
      }
    }

    // Determine if we should stream or return JSON
    const isConversationMode = messages && Array.isArray(messages)

    // Prepare Ollama messages
    let ollamaMessages
    if (isConversationMode) {
      // Conversation mode: system + all conversation messages
      ollamaMessages = [
        { role: 'system', content: enhancedSystemPrompt },
        ...conversationMessages
      ]
    } else {
      // Single message mode: system + single user message
      ollamaMessages = [
        { role: 'system', content: enhancedSystemPrompt },
        { role: 'user', content: lastUserMessage }
      ]
    }

    // Prepare the request body for Ollama
    const ollamaRequestBody = {
      model: model || 'mistral-small3.2:latest',
      messages: ollamaMessages,
      stream: true  // Always stream for both single message and conversation modes
    }

    // Make request to Ollama API
    const ollamaResponse = await fetch('http://localhost:11434/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(ollamaRequestBody),
    })

    if (!ollamaResponse.ok) {
      console.error('Ollama API error:', ollamaResponse.statusText)
      return new Response('Failed to get response from Ollama', { status: 500 })
    }

    // Always stream the response for both single message and conversation modes
    const stream = new ReadableStream({
      start(controller) {
        const reader = ollamaResponse.body?.getReader()

        if (!reader) {
          controller.close()
          return
        }

        function pump(): Promise<void> {
          return reader!.read().then(({ done, value }) => {
            if (done) {
              controller.close()
              return
            }

            controller.enqueue(value)
            return pump()
          })
        }

        return pump()
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
      },
    })

  } catch (error) {
    console.error('Chat API error:', error)
    return new Response('Internal server error', { status: 500 })
  }
}
