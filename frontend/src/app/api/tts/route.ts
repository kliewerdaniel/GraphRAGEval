import { NextRequest } from 'next/server'
import { TTSService } from '@/lib/tts-service'

const ttsService = new TTSService()

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { text, model, outputPath } = body

    if (!text) {
      return new Response('Text is required', { status: 400 })
    }

    // Generate unique filename if not provided
    const filename = outputPath || `tts_${Date.now()}.wav`

    // Generate speech using TTS service (now uses Coqui TTS server)
    const generatedFilename = await ttsService.generateSpeech({
      text,
      model: model || 'tts_models/en/ljspeech/tacotron2-DDC',
      outputPath: filename
    })

    // Return the public URL path
    const publicPath = `/audio/${generatedFilename}`

    return new Response(JSON.stringify({
      success: true,
      audioUrl: publicPath,
      message: 'TTS audio generated successfully'
    }), {
      headers: {
        'Content-Type': 'application/json',
      },
    })

  } catch (error) {
    console.error('TTS API error:', error)
    return new Response(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : 'Internal server error'
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }
}

export async function GET() {
  try {
    // Check if TTS server is available
    const isServerAvailable = await ttsService.checkServerHealth()

    if (isServerAvailable) {
      // List available models from the server
      const models = await ttsService.listAvailableModels()

      return new Response(JSON.stringify({
        success: true,
        models: models
      }), {
        headers: {
          'Content-Type': 'application/json',
        },
      })
    } else {
      // Return fallback models when server is not available
      return new Response(JSON.stringify({
        success: true,
        models: [
          {
            name: 'browser-tts',
            description: 'Browser TTS (Default)',
            language: 'en'
          }
        ]
      }), {
        headers: {
          'Content-Type': 'application/json',
        },
      })
    }

  } catch (error) {
    console.error('TTS models list error:', error)
    return new Response(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : 'Internal server error'
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }
}
