import { create } from 'zustand'

// Agent Action Types
export type ActionType = 'thinking' | 'file_change' | 'function_call' | 'mcp_call' | 'error' | 'complete' | 'text'

export interface AgentAction {
    id: string
    type: ActionType
    timestamp: Date
    content: string
    metadata?: {
        // For file_change
        filename?: string
        linesAdded?: number
        linesDeleted?: number
        diff?: string
        oldContent?: string
        newContent?: string
        // For function_call
        functionName?: string
        params?: Record<string, unknown>
        result?: unknown
        // For mcp_call
        mcpServer?: string
        mcpTool?: string
        mcpResult?: unknown
    }
}

export interface Task {
    id: string
    name: string
    status: 'running' | 'completed' | 'pending' | 'error'
    actions: AgentAction[]
    createdAt: Date
}

interface StudioState {
    tasks: Task[]
    activeTaskId: string | null
    selectedFileForDiff: string | null

    // Actions
    addTask: (name: string) => string
    updateTaskStatus: (taskId: string, status: Task['status']) => void
    addAction: (taskId: string, action: Omit<AgentAction, 'id' | 'timestamp'>) => void
    setActiveTask: (taskId: string | null) => void
    setSelectedFileForDiff: (filename: string | null) => void
}

export const useStudioStore = create<StudioState>((set, get) => ({
    tasks: [],
    activeTaskId: null,
    selectedFileForDiff: null,

    addTask: (name) => {
        const id = `task-${Date.now()}`
        set(state => ({
            tasks: [...state.tasks, {
                id,
                name,
                status: 'running',
                actions: [],
                createdAt: new Date()
            }],
            activeTaskId: id
        }))
        return id
    },

    updateTaskStatus: (taskId, status) => {
        set(state => ({
            tasks: state.tasks.map(t =>
                t.id === taskId ? { ...t, status } : t
            )
        }))
    },

    addAction: (taskId, action) => {
        const newAction: AgentAction = {
            ...action,
            id: `action-${Date.now()}-${Math.random().toString(36).slice(2)}`,
            timestamp: new Date()
        }
        set(state => ({
            tasks: state.tasks.map(t =>
                t.id === taskId
                    ? { ...t, actions: [...t.actions, newAction] }
                    : t
            )
        }))
    },

    setActiveTask: (taskId) => set({ activeTaskId: taskId }),
    setSelectedFileForDiff: (filename) => set({ selectedFileForDiff: filename })
}))
