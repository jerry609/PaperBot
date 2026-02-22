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
        // For chat messages
        role?: 'user' | 'assistant'
        mode?: 'Code' | 'Plan' | 'Ask'
        model?: string
    }
}

export interface Task {
    id: string
    name: string
    status: 'running' | 'completed' | 'pending' | 'error'
    actions: AgentAction[]
    createdAt: Date
    paperId?: string  // Link task to a paper
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

// Studio Paper - represents a paper being reproduced
export type StudioPaperStatus = 'draft' | 'generating' | 'ready' | 'running' | 'completed' | 'error'

export interface StudioPaper {
    id: string
    title: string
    abstract: string
    methodSection?: string

    // Reproduction state
    status: StudioPaperStatus
    outputDir?: string
    lastGenCodeResult?: GenCodeResult

    // Timestamps
    createdAt: string
    updatedAt: string

    // Linked runs
    taskIds: string[]
}

const STORAGE_KEY = 'paperbot-studio-papers'

function generateId(): string {
    return `paper-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function loadPapersFromStorage(): StudioPaper[] {
    if (typeof window === 'undefined') return []
    try {
        const stored = localStorage.getItem(STORAGE_KEY)
        if (stored) {
            return JSON.parse(stored) as StudioPaper[]
        }
    } catch (e) {
        console.error('Failed to load papers from localStorage:', e)
    }
    return []
}

function savePapersToStorage(papers: StudioPaper[]): void {
    if (typeof window === 'undefined') return
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(papers))
    } catch (e) {
        console.error('Failed to save papers to localStorage:', e)
    }
}

interface StudioState {
    // Paper management
    papers: StudioPaper[]
    selectedPaperId: string | null

    // Task management (scoped to selected paper)
    tasks: Task[]
    activeTaskId: string | null
    selectedFileForDiff: string | null
    paperDraft: PaperDraft
    lastGenCodeResult: GenCodeResult | null
    workspaceSnapshotId: number | null

    // Paper actions
    addPaper: (paper: Omit<StudioPaper, 'id' | 'createdAt' | 'updatedAt' | 'taskIds' | 'status'>) => string
    updatePaper: (paperId: string, updates: Partial<Omit<StudioPaper, 'id' | 'createdAt'>>) => void
    deletePaper: (paperId: string) => void
    selectPaper: (paperId: string | null) => void
    loadPapers: () => void
    getSelectedPaper: () => StudioPaper | null

    // Task actions
    addTask: (name: string) => string
    updateTaskStatus: (taskId: string, status: Task['status']) => void
    addAction: (taskId: string, action: Omit<AgentAction, 'id' | 'timestamp'>) => void
    setActiveTask: (taskId: string | null) => void
    setSelectedFileForDiff: (filename: string | null) => void
    setPaperDraft: (partial: Partial<PaperDraft>) => void
    setLastGenCodeResult: (result: GenCodeResult | null) => void
    setWorkspaceSnapshotId: (snapshotId: number | null) => void
}

export const useStudioStore = create<StudioState>((set, get) => ({
    // Paper state
    papers: [],
    selectedPaperId: null,

    // Task state
    tasks: [],
    activeTaskId: null,
    selectedFileForDiff: null,
    paperDraft: { title: "", abstract: "", methodSection: "" },
    lastGenCodeResult: null,
    workspaceSnapshotId: null,

    // Paper actions
    addPaper: (paper) => {
        const id = generateId()
        const now = new Date().toISOString()
        const newPaper: StudioPaper = {
            ...paper,
            id,
            status: 'draft',
            createdAt: now,
            updatedAt: now,
            taskIds: [],
        }
        set(state => {
            const newPapers = [...state.papers, newPaper]
            savePapersToStorage(newPapers)
            return {
                papers: newPapers,
                selectedPaperId: id,
                // Sync paperDraft with new paper
                paperDraft: {
                    title: newPaper.title,
                    abstract: newPaper.abstract,
                    methodSection: newPaper.methodSection || '',
                },
                lastGenCodeResult: null,
                workspaceSnapshotId: null,
            }
        })
        return id
    },

    updatePaper: (paperId, updates) => {
        set(state => {
            const newPapers = state.papers.map(p =>
                p.id === paperId
                    ? { ...p, ...updates, updatedAt: new Date().toISOString() }
                    : p
            )
            savePapersToStorage(newPapers)
            return { papers: newPapers }
        })
    },

    deletePaper: (paperId) => {
        set(state => {
            const newPapers = state.papers.filter(p => p.id !== paperId)
            savePapersToStorage(newPapers)
            // Clear selection if deleted paper was selected
            const newSelectedPaperId = state.selectedPaperId === paperId ? null : state.selectedPaperId
            return {
                papers: newPapers,
                selectedPaperId: newSelectedPaperId,
                // Clear draft if deleted paper was selected
                ...(state.selectedPaperId === paperId ? {
                    paperDraft: { title: '', abstract: '', methodSection: '' },
                    lastGenCodeResult: null,
                } : {}),
            }
        })
    },

    selectPaper: (paperId) => {
        const state = get()
        const paper = paperId ? state.papers.find(p => p.id === paperId) : null
        set({
            selectedPaperId: paperId,
            // Sync paperDraft with selected paper
            paperDraft: paper
                ? { title: paper.title, abstract: paper.abstract, methodSection: paper.methodSection || '' }
                : { title: '', abstract: '', methodSection: '' },
            // Load paper's lastGenCodeResult if available
            lastGenCodeResult: paper?.lastGenCodeResult || null,
            workspaceSnapshotId: null,
            // Clear active task when switching papers
            activeTaskId: null,
        })
    },

    loadPapers: () => {
        const papers = loadPapersFromStorage()
        set({ papers })
    },

    getSelectedPaper: () => {
        const state = get()
        return state.selectedPaperId
            ? state.papers.find(p => p.id === state.selectedPaperId) || null
            : null
    },

    // Task actions
    addTask: (name) => {
        const state = get()
        const id = `task-${Date.now()}`
        const paperId = state.selectedPaperId
        set(currentState => ({
            tasks: [...currentState.tasks, {
                id,
                name,
                status: 'running',
                actions: [],
                createdAt: new Date(),
                paperId: paperId || undefined,
            }],
            activeTaskId: id
        }))

        // Link task to paper if one is selected
        if (paperId) {
            const paper = state.papers.find(p => p.id === paperId)
            if (paper) {
                set(currentState => ({
                    papers: currentState.papers.map(p =>
                        p.id === paperId
                            ? { ...p, taskIds: [...p.taskIds, id], updatedAt: new Date().toISOString() }
                            : p
                    )
                }))
                // Persist updated papers
                savePapersToStorage(get().papers)
            }
        }
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

    setLastGenCodeResult: (result) => {
        const state = get()
        set({ lastGenCodeResult: result })
        // Also update the selected paper's lastGenCodeResult and outputDir
        if (state.selectedPaperId && result) {
            const updates: Partial<StudioPaper> = {
                lastGenCodeResult: result,
                status: result.success ? 'ready' : 'error',
            }
            if (result.outputDir) {
                updates.outputDir = result.outputDir
            }
            set(currentState => {
                const newPapers = currentState.papers.map(p =>
                    p.id === state.selectedPaperId
                        ? { ...p, ...updates, updatedAt: new Date().toISOString() }
                        : p
                )
                savePapersToStorage(newPapers)
                return { papers: newPapers }
            })
        }
    },

    setWorkspaceSnapshotId: (snapshotId) => set({ workspaceSnapshotId: snapshotId }),
}))
