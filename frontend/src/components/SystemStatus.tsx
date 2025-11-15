"use client";

import { useState, useEffect } from "react";

interface SystemStatusData {
  neo4j_connected: boolean;
  ollama_ready: boolean;
  redis_connected: boolean;
  paper_count: number;
  evaluation_count: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function SystemStatus() {
  const [status, setStatus] = useState<SystemStatusData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkSystemStatus();
    // Refresh status every 30 seconds
    const interval = setInterval(checkSystemStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkSystemStatus = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/api/status`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setStatus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check system status');
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusIcon = (isConnected: boolean) => (
    <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
  );

  const getStatusText = (isConnected: boolean) => (
    <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
      {isConnected ? 'Connected' : 'Disconnected'}
    </span>
  );

  return (
    <div className="bg-white border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">System Status</h3>
        <button
          onClick={checkSystemStatus}
          disabled={isLoading}
          className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded text-sm disabled:opacity-50"
        >
          {isLoading ? 'Checking...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          Error: {error}
        </div>
      )}

      {isLoading && !status && (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Checking system status...</span>
        </div>
      )}

      {status && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Core Services */}
          <div className="space-y-3">
            <h4 className="font-medium text-gray-900">Core Services</h4>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {getStatusIcon(status.neo4j_connected)}
                <span className="text-sm text-gray-700">Neo4j Database</span>
              </div>
              {getStatusText(status.neo4j_connected)}
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {getStatusIcon(status.ollama_ready)}
                <span className="text-sm text-gray-700">Ollama LLM</span>
              </div>
              {getStatusText(status.ollama_ready)}
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {getStatusIcon(status.redis_connected)}
                <span className="text-sm text-gray-700">Redis Cache</span>
              </div>
              {getStatusText(status.redis_connected)}
            </div>
          </div>

          {/* Data Statistics */}
          <div className="space-y-3">
            <h4 className="font-medium text-gray-900">Data Statistics</h4>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">Research Papers</span>
              <span className="text-sm font-medium text-gray-900">
                {status.paper_count.toLocaleString()}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">Evaluations Run</span>
              <span className="text-sm font-medium text-gray-900">
                {status.evaluation_count.toLocaleString()}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700">System Health</span>
              <span className={`text-sm font-medium ${
                status.neo4j_connected && status.ollama_ready
                  ? 'text-green-600'
                  : 'text-red-600'
              }`}>
                {status.neo4j_connected && status.ollama_ready ? 'Healthy' : 'Issues Detected'}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 pt-3 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          Powered by GraphRAG architecture with Neo4j knowledge graphs,
          Ollama for local inference, and vero-eval for systematic testing.
        </div>
      </div>
    </div>
  );
}
