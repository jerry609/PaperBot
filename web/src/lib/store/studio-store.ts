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

export type GenCodeResult = {
    success?: boolean
    outputDir?: string
    files?: Array<{ name: string; lines: number; purpose: string }>
    blueprint?: { architectureType?: string; domain?: string }
    verificationPassed?: boolean
}

export type PaperDraft = {
    title: string
    abstract: string
    methodSection: string
}

interface StudioState {
    tasks: Task[]
    activeTaskId: string | null
    selectedFileForDiff: string | null
    paperDraft: PaperDraft
    lastGenCodeResult: GenCodeResult | null
    workspaceSnapshotId: number | null

    // Actions
    addTask: (name: string) => string
    updateTaskStatus: (taskId: string, status: Task['status']) => void
    addAction: (taskId: string, action: Omit<AgentAction, 'id' | 'timestamp'>) => void
    setActiveTask: (taskId: string | null) => void
    setSelectedFileForDiff: (filename: string | null) => void
    setPaperDraft: (partial: Partial<PaperDraft>) => void
    setLastGenCodeResult: (result: GenCodeResult | null) => void
    setWorkspaceSnapshotId: (snapshotId: number | null) => void
}

export const useStudioStore = create<StudioState>((set, _get) => ({
    tasks: [],
    activeTaskId: null,
    selectedFileForDiff: null,
    paperDraft: { title: "", abstract: "", methodSection: "" },
    lastGenCodeResult: null,
    workspaceSnapshotId: null,

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
    setSelectedFileForDiff: (filename) => set({ selectedFileForDiff: filename }),

    setPaperDraft: (partial) => set((state) => ({
        paperDraft: { ...state.paperDraft, ...partial }
    })),

    setLastGenCodeResult: (result) => set({ lastGenCodeResult: result }),
    setWorkspaceSnapshotId: (snapshotId) => set({ workspaceSnapshotId: snapshotId }),
}))
