'use client'

import { useRef, useEffect, ReactNode, useCallback } from 'react'
// TODO: Fix chat ingestion - it's not processing messages correctly
import ReactMarkdown from 'react-markdown'
import { Highlight, themes } from 'prism-react-renderer'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import SystemPromptManager from './SystemPromptManager'
import { ThemeToggle } from './ThemeToggle'
import { Settings, Copy, Volume2, Play, AudioWaveform, ChevronDown, MessageSquare, Plus, Speaker, Trash2, Mic, Loader2 } from 'lucide-react'
import { voiceProcessor } from '@/utils/VoiceProcessor'
import { TTSService } from '@/lib/tts-service'
import { useChatStore, useUIStore, type Message } from '@/lib/stores'



const CodeBlock = ({ className, children }: { className?: string; children?: ReactNode }) => {
  const language = className?.replace('language-', '') || 'text'
  const copyCode = async () => {
    await navigator.clipboard.writeText(String(children || ''))
  }
  return (
    <div className="relative bg-slate-900 rounded-md p-4 my-2">
      <Button
        variant="ghost"
        size="sm"
        className="absolute top-2 right-2 h-6 w-6 p-0 text-white hover:bg-slate-700"
        onClick={copyCode}
        title="Copy code"
      >
        <Copy className="h-3 w-3" />
      </Button>
      <Highlight
        theme={themes.oneDark}
        code={String(children).replace(/\n$/, '')}
        language={language as any}
      >
        {({ style, className: className2, tokens, getLineProps, getTokenProps }) => (
          <pre style={style} className={className2}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  )
}

const components = {
  code: (props: any) => {
    const { inline, className, children } = props
    return !inline ? (
      <CodeBlock className={className}>
        {children}
      </CodeBlock>
    ) : (
      <code className="bg-slate-200 dark:bg-slate-800 px-1 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    );
  },
};

export default function Chat() {
  // Store selectors
  const {
    messages,
    input,
    isLoading,
    models,
    selectedModel,
    selectedPromptId,
    inputRows,
    isNearBottom,
    isUserScrolling,
    sessions,
    currentSessionId,
    systemPrompts,
    setInput,
    setIsLoading,
    setSelectedModel,
    setSelectedPromptId,
    setInputRows,
    setIsNearBottom,
    setIsUserScrolling,
    addMessage,
    updateLastMessage,
    loadSessions,
    saveSession,
    createNewSession,
    clearCurrentChat,
    switchSession,
    renameSession,
    deleteSession,
    clearAllSessions,
    loadModels,
    fetchSystemPrompts
  } = useChatStore()

  const {
    isEditorOpen,
    isHistoryOpen,
    editingSessionId,
    editingTitle,
    voices,
    selectedVoice,
    playingMessageId,
    voiceEnhancement,
    autoPlayEnabled,
    ttsModels,
    selectedTTSModel,
    isTTSLoading,
    ttsLoadingMessages,
    setIsEditorOpen,
    setIsHistoryOpen,
    setEditingSessionId,
    setEditingTitle,
    loadVoices,
    setSelectedVoice,
    setPlayingMessageId,
    setVoiceEnhancement,
    setAutoPlayEnabled,
    loadTTSModels,
    setSelectedTTSModel,
    setTTSLoading,
    addTTSLoadingMessage,
    removeTTSLoadingMessage,
    isMessageTTSLoading
  } = useUIStore()

  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const wasNearBottomRef = useRef(true)

  const scrollToBottomForced = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const lines = e.target.value.split('\n').length
    setInputRows(Math.min(Math.max(lines, 3), 5))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(e as any)
    }
  }

  const stopGeneration = () => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
  }

  const handleScroll = useCallback(() => {
    const element = messagesContainerRef.current
    if (!element) return
    const { scrollTop, scrollHeight, clientHeight } = element
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight
    const isNear = distanceFromBottom < 100
    if (wasNearBottomRef.current && !isNear) {
      setIsUserScrolling(true)
    }
    wasNearBottomRef.current = isNear
    if (isNear) {
      setIsUserScrolling(false)
    }
    setIsNearBottom(isNear)
  }, [])

  const autoScrollToBottom = useCallback(() => {
    if (isNearBottom && !isUserScrolling) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [isNearBottom, isUserScrolling])

  const startRenaming = (sessionId: string, currentTitle: string) => {
    setEditingSessionId(sessionId)
    setEditingTitle(currentTitle)
  }

  const saveRenaming = () => {
    if (!editingSessionId || !editingTitle.trim()) return
    renameSession(editingSessionId, editingTitle.trim())
    setEditingSessionId(null)
    setEditingTitle('')
  }

  const cancelRenaming = () => {
    setEditingSessionId(null)
    setEditingTitle('')
  }

  const copyToClipboard = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      // Could add toast notification here if desired
    } catch (error) {
      console.error('Failed to copy to clipboard:', error)
    }
  }

  const speakMessage = async (content: string, messageIndex: number) => {
    // Stop any currently playing TTS
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel()
    }

    if (playingMessageId === messageIndex) {
      setPlayingMessageId(null)
      return
    }

    try {
      // Show loading state for this message
      addTTSLoadingMessage(messageIndex)

      // Check if using browser TTS or if Coqui TTS is available
      const isBrowserTTS = selectedTTSModel === 'browser-tts'

      if (isBrowserTTS) {
        // Use browser TTS directly
        removeTTSLoadingMessage(messageIndex)

        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(content)

          // Set the selected voice
          const selectedVoiceObj = voices.find(voice => voice.voiceURI === selectedVoice)
          if (selectedVoiceObj) {
            utterance.voice = selectedVoiceObj
          }

          // Apply voice enhancement if enabled
          if (voiceEnhancement) {
            await voiceProcessor.enhanceSpeech(utterance, content)
          }

          utterance.onstart = () => {
            setPlayingMessageId(messageIndex)
          }

          utterance.onend = () => {
            setPlayingMessageId(null)
          }

          utterance.onerror = () => {
            setPlayingMessageId(null)
          }

          speechSynthesis.speak(utterance)
        } else {
          alert('Text-to-speech is not supported.')
        }
        return
      }

      // Try Coqui TTS service
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: content,
          model: selectedTTSModel,
          outputPath: `tts_${Date.now()}_${messageIndex}.wav`
        }),
      })

      if (!response.ok) {
        throw new Error(`TTS API responded with status: ${response.status}`)
      }

      const data = await response.json()
      if (data.success && data.audioUrl) {
        // Remove loading state
        removeTTSLoadingMessage(messageIndex)

        // Create audio element and play
        const audio = new Audio(data.audioUrl)
        audio.onended = () => {
          setPlayingMessageId(null)
        }
        audio.onerror = () => {
          setPlayingMessageId(null)
        }
        await audio.play()
        setPlayingMessageId(messageIndex)
      } else {
        throw new Error(data.error || 'TTS generation failed')
      }
    } catch (error) {
      console.error('TTS error:', error)
      removeTTSLoadingMessage(messageIndex)
      setPlayingMessageId(null)

      // Fallback to browser TTS if Coqui TTS fails
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(content)

        // Set the selected voice
        const selectedVoiceObj = voices.find(voice => voice.voiceURI === selectedVoice)
        if (selectedVoiceObj) {
          utterance.voice = selectedVoiceObj
        }

        // Apply voice enhancement if enabled
        if (voiceEnhancement) {
          await voiceProcessor.enhanceSpeech(utterance, content)
        }

        utterance.onstart = () => {
          setPlayingMessageId(messageIndex)
        }

        utterance.onend = () => {
          setPlayingMessageId(null)
        }

        utterance.onerror = () => {
          setPlayingMessageId(null)
        }

        speechSynthesis.speak(utterance)
      } else {
        alert('Text-to-speech is not supported and TTS generation failed.')
      }
    }
  }



  useEffect(() => {
    setTimeout(() => autoScrollToBottom(), 0)
  }, [messages, autoScrollToBottom])

  useEffect(() => {
    loadModels()
  }, [loadModels])

  useEffect(() => {
    loadVoices()
    const handleVoicesChanged = () => loadVoices()
    speechSynthesis.addEventListener('voiceschanged', handleVoicesChanged)

    return () => {
      speechSynthesis.removeEventListener('voiceschanged', handleVoicesChanged)
    }
  }, [loadVoices, selectedVoice])

  useEffect(() => {
    const element = messagesContainerRef.current
    if (!element) return
    element.addEventListener('scroll', handleScroll)
    return () => element.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    fetchSystemPrompts()
  }, [fetchSystemPrompts])

  useEffect(() => {
    loadTTSModels()
  }, [loadTTSModels])

  // Save session when model or prompt changes (but not on every message update)
  useEffect(() => {
    if (currentSessionId && sessions.length > 0 && !isLoading) {
      saveSession()
    }
  }, [selectedModel, selectedPromptId])

  // Save session when message thread is complete (not during streaming)
  useEffect(() => {
    if (currentSessionId && sessions.length > 0 && !isLoading && messages.length >= 2) {
      saveSession()
    }
  }, [messages])

  const sendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault()

    if (!input.trim() || isLoading) return

    const abortController = new AbortController()
    abortControllerRef.current = abortController

    const userMessage: Message = { id: `user-${Date.now()}`, role: 'user', content: input.trim() }
    addMessage(userMessage)
    setInput('')
    setInputRows(3)
    setIsLoading(true)
    setTimeout(() => inputRef.current?.focus(), 0)

    // Send full conversation history to preserve context
    const currentMessages = [...messages, userMessage].map(msg => ({
      role: msg.role,
      content: msg.content
    }))

    try {
      // For now, we'll just send the latest user message as the query
      // and the previous messages as chat_history
      const latestUserMessage = currentMessages.filter(m => m.role === 'user').slice(-1)[0]
      const chatHistory = currentMessages.slice(0, -1) // All messages except the last user message

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          chat_history: chatHistory
        }),
        signal: abortController.signal
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      // Our API returns JSON, not streaming
      const data = await response.json()
      const assistantMessage = data.response

      addMessage({ id: `assistant-${Date.now()}`, role: 'assistant', content: assistantMessage })

      // Auto-play if enabled
      if (autoPlayEnabled && assistantMessage.trim()) {
        speakMessage(assistantMessage, messages.length - 1)
      }

    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Error sending message:', error)
        addMessage({
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please make sure Ollama is running and try again.'
        })
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-2 sm:p-4">
      <div className="flex-1 flex flex-col bg-card/80 backdrop-blur-sm border rounded-xl shadow-sm">
        <div className="px-6 py-6 border-b">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="hidden sm:inline">BOT</span>
                <span className="sm:hidden">B</span>
              </div>
              <ThemeToggle />
            </div>
            <div className="flex items-center gap-1 flex-wrap">
              <Button
                variant="ghost"
                size="sm"
                onClick={createNewSession}
                title="New Chat"
                className="h-8 px-2 sm:px-3 btn-hover-lift focus-glow"
              >
                <Plus className="h-4 w-4 mr-1" />
                <span className="hidden sm:inline">New</span>
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={clearCurrentChat}
                title="Clear Current Chat"
                className="h-8 px-2 sm:px-3 btn-hover-lift focus-glow"
                disabled={messages.length === 0 || isLoading}
              >
                <Trash2 className="h-4 w-4 mr-1" />
                <span className="hidden sm:inline">Clear</span>
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsHistoryOpen(true)}
                title="Chat History"
                className="h-8 px-2 sm:px-3 btn-hover-lift focus-glow"
              >
                <MessageSquare className="h-4 w-4 mr-1" />
                <span className="hidden sm:inline">History</span>
              </Button>
            </div>
          </div>
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3 mt-4">
            {/* Row 1: Prompt and Model */}
            <div className="flex flex-col xs:flex-row gap-2 sm:gap-3 flex-1">
              <div className="flex items-center gap-2 flex-1">
                <span className="text-sm text-muted-foreground hidden sm:inline whitespace-nowrap">Prompt:</span>
                <Select value={selectedPromptId} onValueChange={setSelectedPromptId}>
                  <SelectTrigger className="w-full sm:w-48 h-8">
                    <SelectValue placeholder="Select prompt" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    {systemPrompts.map((prompt) => (
                      <SelectItem key={prompt.id} value={prompt.id}>
                        {prompt.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-2 flex-1">
                <span className="text-sm text-muted-foreground hidden sm:inline whitespace-nowrap">Model:</span>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-full sm:w-52 h-8">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent position="popper">
                    {models.map((model) => (
                      <SelectItem key={model.name} value={model.name}>
                        {model.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Settings Button */}
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsEditorOpen(true)}
                title="Edit system prompt"
                className="h-8 w-8 shrink-0"
              >
                <Settings className="h-4 w-4" />
              </Button>
            </div>

            {/* Row 2: Voice Controls */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex items-center gap-1">
                  <Button
                    variant={voiceEnhancement ? "default" : "outline"}
                    size="sm"
                    onClick={() => setVoiceEnhancement(!voiceEnhancement)}
                    title={voiceEnhancement ? "Disable voice enhancement" : "Enable voice enhancement"}
                    className={`h-8 px-2 sm:px-3 ${voiceEnhancement ? "bg-teal-100 text-teal-800" : ""}`}
                  >
                    <AudioWaveform className="h-4 w-4" />
                    <span className="hidden sm:inline ml-1">Enhanced</span>
                  </Button>
                  <Button
                    variant={autoPlayEnabled ? "default" : "outline"}
                    size="sm"
                    onClick={() => setAutoPlayEnabled(!autoPlayEnabled)}
                    title={autoPlayEnabled ? "Disable auto-play" : "Enable auto-play"}
                    className={`h-8 px-2 sm:px-3 ${autoPlayEnabled ? "bg-teal-100 text-teal-800" : ""}`}
                  >
                    <Speaker className="h-4 w-4" />
                    <span className="hidden sm:inline ml-1">Auto-play</span>
                  </Button>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground hidden sm:inline whitespace-nowrap">TTS:</span>
                  <Select value={selectedTTSModel} onValueChange={setSelectedTTSModel} disabled={isTTSLoading}>
                    <SelectTrigger className="w-48 sm:w-56 h-8">
                      <SelectValue placeholder="Select TTS model" />
                    </SelectTrigger>
                    <SelectContent position="popper">
                      {ttsModels.map((model) => (
                        <SelectItem key={model.name} value={model.name}>
                          <div className="flex flex-col">
                            <span className="font-medium">{model.description}</span>
                            {model.language && (
                              <span className="text-xs text-muted-foreground">{model.language}</span>
                            )}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground hidden sm:inline whitespace-nowrap">Voice:</span>
                  <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                    <SelectTrigger className="w-32 sm:w-40 h-8">
                      <SelectValue placeholder="Select voice" />
                    </SelectTrigger>
                    <SelectContent position="popper">
                      {voices.map((voice) => (
                        <SelectItem key={voice.voiceURI} value={voice.voiceURI}>
                          {voice.name} ({voice.lang})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="flex-1 flex flex-col space-y-4 px-6 pb-6">
          {/* Messages */}
          <div
            ref={messagesContainerRef}
            className="flex-1 overflow-y-auto space-y-4 p-4 border rounded-lg bg-muted/80 relative"
          >
            {messages.length === 0 ? (
              <div className="text-center text-muted-foreground mt-8">
                <p>Start a conversation by typing a message below.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  } message-enter`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg relative group transition-all duration-200 hover:shadow-md ${
                      message.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-card border hover:bg-accent/5'
                    }`}
                  >
                    {message.role === 'assistant' && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className={`absolute top-1 left-1 h-6 w-6 p-0 transition-all duration-200 bg-background/50 hover:bg-background ${
                          isMessageTTSLoading(index) ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                        }`}
                        onClick={() => speakMessage(message.content, index)}
                        title={
                          isMessageTTSLoading(index)
                            ? "Generating TTS..."
                            : playingMessageId === index
                              ? "Stop TTS playback"
                              : "Generate and play TTS"
                        }
                        disabled={isMessageTTSLoading(index)}
                      >
                        {isMessageTTSLoading(index) ? (
                          <Loader2 className="h-3 w-3 text-blue-500 animate-spin" />
                        ) : playingMessageId === index ? (
                          <Volume2 className="h-3 w-3 text-red-500" />
                        ) : (
                          <Mic className="h-3 w-3" />
                        )}
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute top-1 right-1 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-all duration-200 bg-background/80 hover:bg-background btn-hover-lift focus-glow"
                      onClick={() => copyToClipboard(message.content)}
                      title="Copy message"
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                    {message.role === 'assistant' ? (
                      <ReactMarkdown components={components}>{message.content}</ReactMarkdown>
                    ) : (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-card border p-3 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-current rounded-full opacity-40 typing-bounce" />
                      <div className="w-2 h-2 bg-current rounded-full opacity-40 typing-bounce" />
                      <div className="w-2 h-2 bg-current rounded-full opacity-40 typing-bounce" />
                    </div>
                    <span className="text-sm text-muted-foreground">BOT is thinking</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
            {!isNearBottom && (
              <Button
                className="absolute bottom-4 right-4 rounded-full shadow-lg z-10"
                size="icon"
                onClick={scrollToBottomForced}
                title="Scroll to bottom"
              >
                <ChevronDown className="h-4 w-4" />
              </Button>
            )}
          </div>

          {/* Input Form */}
          <form onSubmit={sendMessage} className="flex space-x-2">
            <div className="flex-1">
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Type your message here..."
                className="w-full p-3 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                rows={inputRows}
                disabled={isLoading}
              />
              <div className="text-xs text-muted text-right mt-1">
                {input.trim().split(/\s+/).filter(w => w).length} words / {input.length} chars
              </div>
            </div>
            <Button
              type="submit"
              onClick={isLoading ? stopGeneration : undefined}
              disabled={!input.trim() && !isLoading}
              className="self-end"
            >
              {isLoading ? 'Stop' : 'Send'}
            </Button>
          </form>
        </div>
      </div>
      <SystemPromptManager
        isOpen={isEditorOpen}
        onClose={() => setIsEditorOpen(false)}
        selectedPromptId={selectedPromptId}
        onPromptSelect={setSelectedPromptId}
      />
      <Dialog open={isHistoryOpen} onOpenChange={setIsHistoryOpen}>
        <DialogContent key={`sessions-${sessions.length}`}>
          <DialogHeader>
            <DialogTitle>Chat History</DialogTitle>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => {
                if (confirm('Are you sure you want to clear all chat history? This action cannot be undone.')) {
                  clearAllSessions()
                  setIsHistoryOpen(false)
                }
              }}
              className="self-end"
            >
              Clear All
            </Button>
          </DialogHeader>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {sessions.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                No saved chats
              </div>
            ) : (
              sessions.map(session => {
                const firstUserMessage = session.messages.find(m => m.role === 'user')?.content
                const preview = firstUserMessage ?
                  firstUserMessage.substring(0, 100) + (firstUserMessage.length > 100 ? '...' : '') :
                  'No messages yet'
                const isEditingTitle = editingSessionId === session.id

                return (
                  <div key={session.id} className="p-3 border rounded-lg hover:bg-accent/50 transition-colors">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        {isEditingTitle ? (
                          <div className="flex items-center gap-2 mb-2">
                            <input
                              type="text"
                              value={editingTitle}
                              onChange={(e) => setEditingTitle(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') saveRenaming()
                                if (e.key === 'Escape') cancelRenaming()
                              }}
                              className="flex-1 px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-ring"
                              autoFocus
                            />
                            <Button size="sm" variant="outline" onClick={saveRenaming}>
                              ✓
                            </Button>
                            <Button size="sm" variant="outline" onClick={cancelRenaming}>
                              ✕
                            </Button>
                          </div>
                        ) : (
                          <p
                            className="font-medium truncate cursor-pointer hover:text-blue-600"
                            onClick={() => startRenaming(session.id, session.title)}
                            title="Click to rename"
                          >
                            {session.title}
                          </p>
                        )}
                        <p className="text-xs text-muted-foreground mb-1">
                          {new Date(session.updatedAt).toLocaleDateString()} • {session.messages.length} messages • {session.model}
                        </p>
                        {firstUserMessage && (
                          <p className="text-sm text-muted-foreground truncate">
                            {preview}
                          </p>
                        )}
                      </div>
                      <div className="flex space-x-1 ml-2">
                        <Button
                          variant={currentSessionId === session.id ? "default" : "outline"}
                          size="sm"
                          onClick={() => switchSession(session.id)}
                        >
                          {currentSessionId === session.id ? "Current" : "Load"}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          title="Rename"
                          onClick={() => startRenaming(session.id, session.title)}
                        >
                          ✏️
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            if (confirm('Are you sure you want to delete this chat?')) {
                              deleteSession(session.id)
                            }
                          }}
                          className="text-destructive hover:text-destructive"
                        >
                          ✕
                        </Button>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
