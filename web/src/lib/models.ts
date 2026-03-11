
export interface ModelConfig {
    id: string
    name: string
    provider: 'google' | 'anthropic' | 'ollama'
    description?: string
    icon?: string
}

export const AVAILABLE_MODELS: ModelConfig[] = [
    {
        id: 'gemini-2.0-flash-exp',
        name: 'Gemini 2.0 Flash',
        provider: 'google',
        description: 'Fast & efficient, great for code generation',
    },
    {
        id: 'claude-sonnet-4-5',
        name: 'Claude Sonnet 4.5',
        provider: 'anthropic',
        description: 'Best balance of speed and reasoning',
    },
    {
        id: 'claude-opus-4-5',
        name: 'Claude Opus 4.5',
        provider: 'anthropic',
        description: 'Best for complex reasoning & analysis',
    },
]

export const DEFAULT_MODEL = 'gemini-2.0-flash-exp'
