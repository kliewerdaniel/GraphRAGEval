import { NextRequest, NextResponse } from 'next/server'
import { readPrompts, createPrompt } from '@/lib/prompts-data'

// GET /api/system-prompts - List all system prompts
export async function GET() {
  try {
    const prompts = readPrompts()
    return NextResponse.json({ prompts })
  } catch (error) {
    console.error('Error reading system prompts:', error)
    return NextResponse.json(
      { error: 'Failed to read system prompts' },
      { status: 500 }
    )
  }
}

// POST /api/system-prompts - Create a new system prompt
export async function POST(request: NextRequest) {
  try {
    const { name, content } = await request.json()

    if (!name || typeof name !== 'string') {
      return NextResponse.json(
        { error: 'Name is required and must be a string' },
        { status: 400 }
      )
    }

    if (!content || typeof content !== 'string') {
      return NextResponse.json(
        { error: 'Content is required and must be a string' },
        { status: 400 }
      )
    }

    const newPrompt = createPrompt(name, content)
    return NextResponse.json({ prompt: newPrompt }, { status: 201 })
  } catch (error) {
    console.error('Error creating system prompt:', error)
    return NextResponse.json(
      { error: 'Failed to create system prompt' },
      { status: 500 }
    )
  }
}
