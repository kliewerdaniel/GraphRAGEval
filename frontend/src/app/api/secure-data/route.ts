import { NextResponse } from 'next/server'

// This demonstrates server-side data fetching
// In a real app, this might connect to a database with sensitive credentials

export async function GET() {
  try {
    // Validate API secret (server-side only - never exposed to client)
    const apiSecret = process.env.API_SECRET_KEY
    if (!apiSecret) {
      return NextResponse.json(
        { error: 'Server configuration error' },
        { status: 500 }
      )
    }

    // Server-side data access
    // This data can be safely accessed because:
    // 1. API_SECRET_KEY is never exposed to the client
    // 2. Database credentials are server-side only
    // 3. Sensitive operations happen in Server Components/API routes

    const data = {
      message: 'This data was fetched server-side',
      timestamp: new Date().toISOString(),
      publicData: {
        // Only public-safe data sent to client
        userCount: 42,
        serverStatus: 'healthy',
        lastUpdated: new Date().toLocaleTimeString(),
      },
    }

    // Log operation (server-side only)
    console.log('[API] Data accessed at', new Date().toISOString())

    return NextResponse.json(data)

  } catch {
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
