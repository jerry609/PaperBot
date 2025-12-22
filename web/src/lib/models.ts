
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
        id: 'claude-3-5-sonnet-20241022',
        name: 'Claude 3.5 Sonnet',
        provider: 'anthropic',
        description: 'Best for complex reasoning & analysis',
    },
    {
        id: 'llama3.3:70b',
        name: 'Llama 3.3 70B',
        provider: 'ollama',
        description: 'Local model, privacy-first',
    },
]

export const DEFAULT_MODEL = 'gemini-2.0-flash-exp'
