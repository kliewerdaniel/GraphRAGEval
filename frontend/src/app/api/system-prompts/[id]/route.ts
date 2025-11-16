import { NextRequest, NextResponse } from 'next/server'
import { getPromptById, updatePrompt, deletePrompt } from '@/lib/prompts-data'

interface Params {
  params: Promise<{
    id: string
  }>
}

// GET /api/system-prompts/[id] - Get a specific system prompt
export async function GET(request: NextRequest, { params }: Params) {
  try {
    const { id } = await params
    const prompt = getPromptById(id)

    if (!prompt) {
      return NextResponse.json(
        { error: 'System prompt not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ prompt })
  } catch (error) {
    console.error('Error reading system prompt:', error)
    return NextResponse.json(
      { error: 'Failed to read system prompt' },
      { status: 500 }
    )
  }
}

// PUT /api/system-prompts/[id] - Update a specific system prompt
export async function PUT(request: NextRequest, { params }: Params) {
  try {
    const { id } = await params
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

    const updatedPrompt = updatePrompt(id, name, content)
    if (!updatedPrompt) {
      return NextResponse.json(
        { error: 'System prompt not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ prompt: updatedPrompt })
  } catch (error) {
    console.error('Error updating system prompt:', error)
    return NextResponse.json(
      { error: 'Failed to update system prompt' },
      { status: 500 }
    )
  }
}

// DELETE /api/system-prompts/[id] - Delete a specific system prompt
export async function DELETE(request: NextRequest, { params }: Params) {
  try {
    const { id } = await params

    // Prevent deletion of default prompt
    if (id === 'default') {
      return NextResponse.json(
        { error: 'Cannot delete the default system prompt' },
        { status: 400 }
      )
    }

    const success = deletePrompt(id)
    if (!success) {
      return NextResponse.json(
        { error: 'System prompt not found' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error deleting system prompt:', error)
    return NextResponse.json(
      { error: 'Failed to delete system prompt' },
      { status: 500 }
    )
  }
}
