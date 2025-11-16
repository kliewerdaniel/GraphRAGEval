'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Settings, Plus, Edit, Trash2, Check, X } from 'lucide-react'
import { usePromptStore, useChatStore } from '@/lib/stores'
import type { SystemPrompt } from '@/lib/stores'

interface SystemPromptManagerProps {
  isOpen: boolean
  onClose: () => void
  selectedPromptId: string
  onPromptSelect: (promptId: string) => void
}

export default function SystemPromptManager({
  isOpen,
  onClose,
  selectedPromptId,
  onPromptSelect
}: SystemPromptManagerProps) {
  // Store selectors
  const { prompts, isLoading, loadPrompts, addPrompt, updatePrompt: storeUpdatePrompt, deletePrompt: storeDeletePrompt } = usePromptStore()
  const { fetchSystemPrompts } = useChatStore()

  // Local form state for creation/editing (not stored in Zustand since it's UI-only)
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editingPrompt, setEditingPrompt] = useState<SystemPrompt | null>(null)
  const [newPromptName, setNewPromptName] = useState('')
  const [newPromptContent, setNewPromptContent] = useState('')

  useEffect(() => {
    if (isOpen) {
      loadPrompts()
    }
  }, [isOpen, loadPrompts])

  const createPrompt = async () => {
    if (!newPromptName.trim() || !newPromptContent.trim()) return
    await addPrompt({ name: newPromptName, content: newPromptContent })
    await fetchSystemPrompts() // Sync with chat store
    setNewPromptName('')
    setNewPromptContent('')
    setIsCreateOpen(false)
  }

  const updatePrompt = async () => {
    if (!editingPrompt || !newPromptName.trim() || !newPromptContent.trim()) return
    await storeUpdatePrompt(editingPrompt.id, { name: newPromptName, content: newPromptContent })
    await fetchSystemPrompts() // Sync with chat store
    setEditingPrompt(null)
    setNewPromptName('')
    setNewPromptContent('')
    setIsEditOpen(false)
  }

  const deletePromptFromStore = async (id: string) => {
    if (id === 'default') return // Prevent deleting default
    await storeDeletePrompt(id)
    await fetchSystemPrompts() // Sync with chat store
    if (selectedPromptId === id) {
      onPromptSelect('default')
    }
  }

  const openEditDialog = (prompt: SystemPrompt) => {
    setEditingPrompt(prompt)
    setNewPromptName(prompt.name)
    setNewPromptContent(prompt.content)
    setIsEditOpen(true)
  }

  const handlePromptSelect = (promptId: string) => {
    onPromptSelect(promptId)
  }

  const truncateContent = (content: string, maxLength: number = 100) => {
    if (content.length <= maxLength) return content
    return content.substring(0, maxLength) + '...'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <Card className="w-full max-w-4xl h-[80vh] flex flex-col">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              System Prompt Manager
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
              <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
                <DialogTrigger asChild>
                  <Button size="sm">
                    <Plus className="h-4 w-4 mr-2" />
                    New Prompt
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Create New System Prompt</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">Name</label>
                      <Input
                        value={newPromptName}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewPromptName(e.target.value)}
                        placeholder="Enter prompt name..."
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Content</label>
                      <Textarea
                        value={newPromptContent}
                        onChange={(e) => setNewPromptContent(e.target.value)}
                        placeholder="Enter prompt content..."
                        className="min-h-[200px]"
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" onClick={() => setIsCreateOpen(false)}>
                        Cancel
                      </Button>
                      <Button onClick={createPrompt}>
                        Create
                      </Button>
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground">Current Prompt:</span>
            <Select value={selectedPromptId} onValueChange={handlePromptSelect}>
              <SelectTrigger className="w-80">
                <SelectValue placeholder="Select a system prompt" />
              </SelectTrigger>
              <SelectContent>
                {prompts.map((prompt) => (
                  <SelectItem key={prompt.id} value={prompt.id}>
                    {prompt.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden">
          <div className="h-full overflow-y-auto space-y-3">
            {isLoading ? (
              <div className="text-center py-8">Loading prompts...</div>
            ) : prompts.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No system prompts found. Create your first prompt.
              </div>
            ) : (
              prompts.map((prompt) => (
                <Card key={prompt.id} className={`p-4 ${selectedPromptId === prompt.id ? 'ring-2 ring-primary' : ''}`}>
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <h3 className="font-medium">{prompt.name}</h3>
                        {selectedPromptId === prompt.id && <Check className="h-4 w-4 text-primary" />}
                        {prompt.id === 'default' && (
                          <span className="text-xs bg-muted px-2 py-1 rounded">Default</span>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {truncateContent(prompt.content)}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Updated: {new Date(prompt.updatedAt).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 ml-4">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openEditDialog(prompt)}
                      >
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => deletePromptFromStore(prompt.id)}
                        disabled={prompt.id === 'default'}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))
            )}
          </div>
        </CardContent>

        <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit System Prompt</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium">Name</label>
                <Input
                  value={newPromptName}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewPromptName(e.target.value)}
                  placeholder="Enter prompt name..."
                />
              </div>
              <div>
                <label className="text-sm font-medium">Content</label>
                <Textarea
                  value={newPromptContent}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setNewPromptContent(e.target.value)}
                  placeholder="Enter prompt content..."
                  className="min-h-[200px]"
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setIsEditOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={updatePrompt}>
                  Save Changes
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </Card>
    </div>
  )
}
