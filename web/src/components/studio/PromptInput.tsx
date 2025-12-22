"use client"

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, ChevronDown, Paperclip } from "lucide-react"
import { AVAILABLE_MODELS, DEFAULT_MODEL, ModelConfig } from "@/lib/models"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface PromptInputProps {
    onSubmit: (prompt: string, model: string) => void
    isLoading?: boolean
}

export function PromptInput({ onSubmit, isLoading }: PromptInputProps) {
    const [input, setInput] = useState('')
    const [selectedModel, setSelectedModel] = useState<ModelConfig>(
        AVAILABLE_MODELS.find(m => m.id === DEFAULT_MODEL) || AVAILABLE_MODELS[0]
    )

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (input.trim() && !isLoading) {
            onSubmit(input.trim(), selectedModel.id)
            setInput('')
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
        }
    }

    return (
        <div className="border-t bg-background p-4">
            <form onSubmit={handleSubmit} className="space-y-3">
                <div className="flex gap-2">
                    <Textarea
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Describe what you want to build... (Shift+Enter for new line)"
                        className="min-h-[20px] max-h-[200px] resize-none py-3"
                        disabled={isLoading}
                        rows={1}
                        style={{ height: 'auto', minHeight: '44px' }}
                        onInput={(e) => {
                            const target = e.target as HTMLTextAreaElement;
                            target.style.height = 'auto';
                            target.style.height = `${target.scrollHeight}px`;
                        }}
                    />
                    <Button
                        type="submit"
                        size="icon"
                        disabled={!input.trim() || isLoading}
                        className="h-[44px] w-[44px] shrink-0"
                    >
                        <Send className="h-4 w-4" />
                    </Button>
                </div>

                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button variant="outline" size="sm" className="text-xs h-8 gap-1">
                                    {selectedModel.name}
                                    <ChevronDown className="h-3 w-3" />
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="start" className="w-64">
                                {AVAILABLE_MODELS.map((model) => (
                                    <DropdownMenuItem
                                        key={model.id}
                                        onClick={() => setSelectedModel(model)}
                                        className="flex flex-col items-start gap-0.5"
                                    >
                                        <span className="font-medium">{model.name}</span>
                                        {model.description && (
                                            <span className="text-xs text-muted-foreground">{model.description}</span>
                                        )}
                                    </DropdownMenuItem>
                                ))}
                            </DropdownMenuContent>
                        </DropdownMenu>

                        <Button variant="ghost" size="sm" className="text-xs h-8 text-muted-foreground">
                            <Paperclip className="h-3.5 w-3.5 mr-1" />
                            Attach
                        </Button>
                    </div>

                    <p className="text-xs text-muted-foreground">
                        Press Enter to send, Shift+Enter for new line
                    </p>
                </div>
            </form>
        </div>
    )
}
