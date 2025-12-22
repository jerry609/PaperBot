"use client"

import { useState } from 'react'
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport } from 'ai'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, User, Code2, Sparkles, ChevronDown } from "lucide-react"
import { AVAILABLE_MODELS, DEFAULT_MODEL, ModelConfig } from "@/lib/models"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface DeepCodeChatProps {
    onApplyCode: (code: string) => void
}

export function DeepCodeChat({ onApplyCode }: DeepCodeChatProps) {
    const [selectedModel, setSelectedModel] = useState<ModelConfig>(
        AVAILABLE_MODELS.find(m => m.id === DEFAULT_MODEL) || AVAILABLE_MODELS[0]
    )
    const [input, setInput] = useState('')

    const { messages, sendMessage, status } = useChat({
        transport: new DefaultChatTransport({
            api: '/api/chat',
            body: { model: selectedModel.id }
        }),
    })

    const extractCodeBlock = (content: string) => {
        const match = content.match(/```[\s\S]*?\n([\s\S]*?)```/)
        return match ? match[1] : null
    }

    const getTextContent = (message: typeof messages[0]) => {
        return message.parts
            .filter(part => part.type === 'text')
            .map(part => (part as { type: 'text'; text: string }).text)
            .join('')
    }

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault()
        if (input.trim() && status === 'ready') {
            sendMessage({ text: input })
            setInput('')
        }
    }

    return (
        <div className="flex flex-col h-full bg-muted/20 border-l">
            {/* Header with Model Selector */}
            <div className="p-3 border-b bg-background flex items-center justify-between">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                    <Sparkles className="h-4 w-4 text-purple-500" /> AI Assistant
                </h3>

                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm" className="text-xs h-7 gap-1">
                            {selectedModel.name}
                            <ChevronDown className="h-3 w-3" />
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-56">
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
            </div>

            {/* Messages Area */}
            <ScrollArea className="flex-1 p-4">
                <div className="space-y-4">
                    {messages.length === 0 && (
                        <div className="text-center text-muted-foreground text-sm py-8 space-y-2">
                            <Bot className="h-8 w-8 mx-auto opacity-50" />
                            <p>Ask me to generate, explain, or debug code.</p>
                            <p className="text-xs">Using: <strong>{selectedModel.name}</strong></p>
                        </div>
                    )}

                    {messages.map(message => {
                        const textContent = getTextContent(message)
                        return (
                            <div key={message.id} className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-purple-100 text-purple-600 dark:bg-purple-900/30'}`}>
                                    {message.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                                </div>

                                <div className={`flex-1 max-w-[85%] rounded-lg p-3 text-sm ${message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-background border shadow-sm'}`}>
                                    <p className="whitespace-pre-wrap">{textContent}</p>

                                    {message.role === 'assistant' && extractCodeBlock(textContent) && (
                                        <div className="mt-3 pt-3 border-t">
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="w-full text-xs h-7"
                                                onClick={() => onApplyCode(extractCodeBlock(textContent)!)}
                                            >
                                                <Code2 className="mr-2 h-3.5 w-3.5" /> Apply to Editor
                                            </Button>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )
                    })}

                    {status === 'streaming' && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center">
                                <Bot className="h-4 w-4" />
                            </div>
                            <div className="bg-background border rounded-lg p-3">
                                <div className="flex gap-1 h-full items-center">
                                    <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                    <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                    <span className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce"></span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </ScrollArea>

            {/* Input Area */}
            <div className="p-3 bg-background border-t">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <Input
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder="Type a message..."
                        className="flex-1"
                        disabled={status !== 'ready'}
                    />
                    <Button type="submit" size="icon" disabled={status !== 'ready'}>
                        <Send className="h-4 w-4" />
                    </Button>
                </form>
            </div>
        </div>
    )
}
