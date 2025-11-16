'use client'

import dynamic from 'next/dynamic'

// Dynamically import Chat with SSR disabled to prevent hydration mismatches
const Chat = dynamic(() => import('@/components/Chat'), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex flex-col h-screen max-w-4xl mx-auto p-2 sm:p-4">
      <div className="flex-1 flex flex-col bg-card/80 backdrop-blur-sm border rounded-xl shadow-sm">
        <div className="flex-1 flex flex-col space-y-4 px-6 pb-6">
          <div className="flex-1 overflow-y-auto space-y-4 p-4 border rounded-lg bg-muted relative">
            <div className="text-center text-muted-foreground mt-8">
              Loading chat interface...
            </div>
          </div>
        </div>
      </div>
    </div>
  )
})

export default function Home() {
  return (
    <div className="min-h-screen">
      <div className="container mx-auto px-4 py-8">
        <Chat />
      </div>
    </div>
  )
}
