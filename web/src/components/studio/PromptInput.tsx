"use client"

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, ChevronDown, Paperclip, Loader2, Sparkles, Zap } from "lucide-react"
import { AVAILABLE_MODELS, DEFAULT_MODEL, ModelConfig } from "@/lib/models"
import { cn } from "@/lib/utils"
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
    const [isFocused, setIsFocused] = useState(false)

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
        <div className={cn(
            "border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 p-3 transition-all",
            isFocused && "shadow-lg"
        )}>
            <form onSubmit={handleSubmit} className="space-y-2.5">
                <div className={cn(
                    "relative flex items-end gap-2 rounded-xl border bg-background p-1 transition-all",
                    isFocused && "ring-2 ring-primary/20 border-primary/50"
                )}>
                    <Textarea
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        placeholder="Describe what you want to build..."
                        className="min-h-[20px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 py-2.5 px-3 text-sm"
                        disabled={isLoading}
                        rows={1}
                        style={{ height: 'auto', minHeight: '40px' }}
                        onInput={(e) => {
                            const target = e.target as HTMLTextAreaElement;
                            target.style.height = 'auto';
                            target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
                        }}
                    />
                    <Button
                        type="submit"
                        size="icon"
                        disabled={!input.trim() || isLoading}
                        className={cn(
                            "h-9 w-9 shrink-0 rounded-lg transition-all",
                            input.trim() && !isLoading && "bg-primary hover:bg-primary/90"
                        )}
                    >
                        {isLoading ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                            <Send className="h-4 w-4" />
                        )}
                    </Button>
                </div>

                <div className="flex items-center justify-between px-1">
                    <div className="flex items-center gap-1.5">
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    className="h-7 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
                                >
                                    <Zap className="h-3 w-3" />
                                    {selectedModel.name}
                                    <ChevronDown className="h-3 w-3" />
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="start" className="w-72">
                                {AVAILABLE_MODELS.map((model) => (
                                    <DropdownMenuItem
                                        key={model.id}
                                        onClick={() => setSelectedModel(model)}
                                        className={cn(
                                            "flex flex-col items-start gap-0.5 py-2",
                                            selectedModel.id === model.id && "bg-muted"
                                        )}
                                    >
                                        <div className="flex items-center gap-2">
                                            <Sparkles className="h-3.5 w-3.5 text-primary" />
                                            <span className="font-medium">{model.name}</span>
                                        </div>
                                        {model.description && (
                                            <span className="text-xs text-muted-foreground pl-5">{model.description}</span>
                                        )}
                                    </DropdownMenuItem>
                                ))}
                            </DropdownMenuContent>
                        </DropdownMenu>

                        <div className="w-px h-4 bg-border" />

                        <Button
                            variant="ghost"
                            size="sm"
                            type="button"
                            className="h-7 text-xs text-muted-foreground hover:text-foreground"
                        >
                            <Paperclip className="h-3 w-3 mr-1.5" />
                            Attach
                        </Button>
                    </div>

                    <p className="text-[10px] text-muted-foreground hidden sm:block">
                        <kbd className="px-1 py-0.5 rounded bg-muted text-[9px] font-mono">Enter</kbd> to send
                        <span className="mx-1">·</span>
                        <kbd className="px-1 py-0.5 rounded bg-muted text-[9px] font-mono">Shift+Enter</kbd> new line
                    </p>
                </div>
            </form>
        </div>
    )
}
