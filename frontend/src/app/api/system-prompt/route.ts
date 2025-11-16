import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const SYSTEM_PROMPT_PATH = path.join(process.cwd(), 'system_prompt.md')

// GET: Read the current system prompt
export async function GET() {
  try {
    const content = fs.readFileSync(SYSTEM_PROMPT_PATH, 'utf8')
    return NextResponse.json({ content })
  } catch (error) {
    console.error('Error reading system prompt:', error)
    return NextResponse.json(
      { error: 'Failed to read system prompt' },
      { status: 500 }
    )
  }
}

// PUT: Update the system prompt
export async function PUT(request: NextRequest) {
  try {
    const { content } = await request.json()

    if (typeof content !== 'string') {
      return NextResponse.json(
        { error: 'Content must be a string' },
        { status: 400 }
      )
    }

    fs.writeFileSync(SYSTEM_PROMPT_PATH, content, 'utf8')
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Error updating system prompt:', error)
    return NextResponse.json(
      { error: 'Failed to update system prompt' },
      { status: 500 }
    )
  }
}
