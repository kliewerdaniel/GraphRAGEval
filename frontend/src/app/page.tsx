"use client";

import { useState } from "react";
import ChatInterface from "@/components/ChatInterface";
import SystemStatus from "@/components/SystemStatus";

export default function Home() {
  const [showStatus, setShowStatus] = useState(false);

  return (
    <div className="flex min-h-screen bg-gray-50">
      <div className="flex-1 max-w-4xl mx-auto p-4">
        <header className="mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Research Assistant
              </h1>
              <p className="text-gray-600 mt-1">
                GraphRAG-powered academic research assistant with vero-eval evaluation
              </p>
            </div>
            <button
              onClick={() => setShowStatus(!showStatus)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              System Status
            </button>
          </div>
        </header>

        {showStatus && (
          <div className="mb-6">
            <SystemStatus />
          </div>
        )}

        <div className="bg-white rounded-lg shadow-sm border">
          <ChatInterface />
        </div>

        <footer className="mt-8 text-center text-sm text-gray-500">
          <p>
            Powered by Neo4j GraphRAG, Ollama, and vero-eval evaluation framework
          </p>
        </footer>
      </div>
    </div>
  );
}
