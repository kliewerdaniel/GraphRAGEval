"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: {
    title: string;
    authors: string[];
    year: string;
    relevance_score: number;
    retrieval_method: string;
  }[];
  quality_grade?: number;
  retrieval_method?: string;
  retrieval_performed?: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showSources, setShowSources] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userMessage.content,
          chat_history: messages.slice(-4).map(m => ({
            role: m.role,
            content: m.content,
          })), // Last 4 messages for context
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        sources: data.sources || [],
        quality_grade: data.quality_grade,
        retrieval_method: data.retrieval_method,
        retrieval_performed: data.retrieval_performed,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I encountered an error while processing your request. Please try again.\n\nError: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleSources = (messageId: string) => {
    setShowSources(prev => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  return (
    <div className="flex flex-col h-[600px]">
      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <div className="text-4xl mb-4">📚</div>
            <h3 className="text-lg font-medium mb-2">
              Welcome to your Research Assistant
            </h3>
            <p className="text-sm">
              Ask questions about research papers, methodologies, or get help with your academic work.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-900"
              }`}
            >
              {/* Message Quality Indicator (for assistant messages) */}
              {message.role === "assistant" && message.quality_grade !== undefined && (
                <div className="flex items-center mb-2 text-xs">
                  <div className="flex items-center space-x-1">
                    <span>Quality:</span>
                    <div className="flex space-x-1">
                      {[...Array(5)].map((_, i) => (
                        <div
                          key={i}
                          className={`w-2 h-2 rounded-full ${
                            i < Math.round((message.quality_grade || 0) * 5)
                              ? "bg-green-500"
                              : "bg-gray-300"
                          }`}
                        />
                      ))}
                    </div>
                    <span className="text-gray-600">
                      {((message.quality_grade || 0) * 100).toFixed(0)}%
                    </span>

                    {message.retrieval_performed && (
                      <>
                        <span className="mx-2">•</span>
                        <span className="text-blue-600">
                          🔍 {message.retrieval_method || "Retrieved context"}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Message Content */}
              <div className="whitespace-pre-wrap">
                {message.content}
              </div>

              {/* Sources Toggle */}
              {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                <div className="mt-3 pt-2 border-t border-gray-200">
                  <button
                    onClick={() => toggleSources(message.id)}
                    className="text-xs text-blue-600 hover:text-blue-800 flex items-center"
                  >
                    📚 {message.sources.length} sources retrieved
                    <svg
                      className={`ml-1 w-3 h-3 transition-transform ${
                        showSources[message.id] ? "rotate-180" : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {showSources[message.id] && (
                    <div className="mt-2 space-y-1">
                      {message.sources.map((source, idx) => (
                        <div key={idx} className="text-xs bg-gray-50 p-2 rounded">
                          <div className="flex items-center justify-between">
                            <span className="font-medium text-gray-900">
                              {source.title}
                            </span>
                            <div className="flex items-center space-x-2">
                              <span className="text-green-600">
                                Score: {(source.relevance_score * 100).toFixed(0)}%
                              </span>
                              <span className={`px-1 py-0.5 rounded text-xs ${
                                source.retrieval_method === 'vector_search'
                                  ? 'bg-blue-100 text-blue-800'
                                  : 'bg-purple-100 text-purple-800'
                              }`}>
                                {source.retrieval_method === 'vector_search' ? '🔍' : '🔗'}
                              </span>
                            </div>
                          </div>
                          <div className="text-gray-600 mt-1">
                            {Array.isArray(source.authors) && source.authors.length > 0 && (
                              <span>Authors: {source.authors.join(", ")} • </span>
                            )}
                            {source.year !== "Unknown" && (
                              <span>Year: {source.year}</span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 p-3 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                <span className="text-sm text-gray-600">Researching...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="p-4 border-t bg-white">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about research papers, methodologies, or get help with your work..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                <span>Searching...</span>
              </div>
            ) : (
              "Ask"
            )}
          </button>
        </form>

        <div className="mt-2 text-xs text-gray-500 text-center">
          Powered by GraphRAG with Neo4j, Ollama, and vero-eval evaluation
        </div>
      </div>
    </div>
  );
}
