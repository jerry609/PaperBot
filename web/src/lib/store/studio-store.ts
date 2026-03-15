import { create } from 'zustand'
import type { ReproContextPack, StageObservationsEvent, StageProgressEvent } from '@/lib/types/p2c'

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
    paperId?: string  // Link task to a paper
}

export type AgentTaskStatus = 'planning' | 'in_progress' | 'repairing' | 'human_review' | 'done' | 'paused' | 'cancelled'

export type BlockType = "think" | "tool" | "diff" | "info" | "result"

export interface AgentTaskLog {
    id: string
    timestamp: string
    event: string
    phase: string
    level: "info" | "warning" | "error" | "success"
    message: string
    blockType?: BlockType
    details?: Record<string, unknown>
}

export type PipelinePhase =
    | 'idle'
    | 'planning'
    | 'executing'
    | 'paused'
    | 'cancelled'
    | 'e2e_running'
    | 'e2e_repairing'
    | 'downloading'
    | 'completed'
    | 'failed'

export interface E2EState {
    status: 'waiting' | 'running' | 'passed' | 'failed' | 'repairing' | 'skipped'
    attempt: number
    maxAttempts: number
    entryPoint: string | null
    command: string | null
    lastExitCode: number | null
    lastStdout: string
    lastStderr: string
    history: Array<{
        attempt: number
        success: boolean
        exitCode: number
        duration: number
        stdoutPreview: string
    }>
}

export interface SandboxFileEntry {
    name: string
    type: 'file' | 'directory'
    children?: SandboxFileEntry[]
}

export interface TimeEstimate {
    elapsedMs: number
    remainingMs: number | null
    avgTaskMs: number | null
    completedTasks: number
    totalTasks: number
}

export interface AgentTask {
    id: string
    title: string
    description: string
    status: AgentTaskStatus
    assignee: string
    progress: number
    tags: string[]
    createdAt: string
    updatedAt: string
    subtasks: { id: string; title: string; done: boolean }[]
    codexOutput?: string
    generatedFiles?: string[]
    reviewFeedback?: string
    lastError?: string
    executionLog?: AgentTaskLog[]
    humanReviews?: Array<{ id: string; decision: string; notes: string; timestamp: string }>
    paperId?: string
    depends_on?: string[]
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
    authors?: string[]
    researchAreas?: string[]

    // Reproduction state
    status: StudioPaperStatus
    outputDir?: string
    lastGenCodeResult?: GenCodeResult
    contextPackId?: string
    boardSessionId?: string

    // Timestamps
    createdAt: string
    updatedAt: string

    // Linked runs
    taskIds: string[]
}

const STORAGE_KEY = 'paperbot-studio-papers'
const RUNTIME_STORAGE_KEY = 'paperbot-studio-runtime'
const RUNTIME_STORAGE_VERSION = 1

function generateId(): string {
    return `paper-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`
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

// Per-paper cached state (preserved across paper switches)
interface PerPaperCache {
    contextPack: ReproContextPack | null
    contextPackLoading: boolean
    contextPackError: string | null
    generationProgress: StageProgressEvent[]
    liveObservations: StageObservationsEvent[]
    activeTaskId: string | null
}

interface PersistedRuntimeState {
    version: number
    selectedPaperId: string | null
    paperCache: Record<string, PerPaperCache>
    boardSessionByPaper: Record<string, string>
    agentTasks: AgentTask[]
    pipelinePhase: PipelinePhase
    e2eState: E2EState | null
    sandboxFiles: SandboxFileEntry[]
    timeEstimate: TimeEstimate | null
}

function _defaultRuntimeState(): PersistedRuntimeState {
    return {
        version: RUNTIME_STORAGE_VERSION,
        selectedPaperId: null,
        paperCache: {},
        boardSessionByPaper: {},
        agentTasks: [],
        pipelinePhase: 'idle',
        e2eState: null,
        sandboxFiles: [],
        timeEstimate: null,
    }
}

function _normalizePaperCache(cache: unknown): Record<string, PerPaperCache> {
    if (!cache || typeof cache !== 'object') return {}
    const normalized: Record<string, PerPaperCache> = {}
    for (const [paperId, value] of Object.entries(cache as Record<string, unknown>)) {
        if (!value || typeof value !== 'object') continue
        const entry = value as Partial<PerPaperCache>
        normalized[paperId] = {
            contextPack: (entry.contextPack ?? null) as ReproContextPack | null,
            contextPackLoading: false,
            contextPackError: (entry.contextPackError ?? null) as string | null,
            generationProgress: Array.isArray(entry.generationProgress) ? entry.generationProgress : [],
            liveObservations: Array.isArray(entry.liveObservations) ? entry.liveObservations : [],
            activeTaskId: typeof entry.activeTaskId === 'string' ? entry.activeTaskId : null,
        }
    }
    return normalized
}

function loadRuntimeStateFromStorage(): PersistedRuntimeState {
    if (typeof window === 'undefined') return _defaultRuntimeState()
    try {
        const stored = localStorage.getItem(RUNTIME_STORAGE_KEY)
        if (!stored) return _defaultRuntimeState()

        const parsed = JSON.parse(stored) as Record<string, unknown>
        const legacyBoardSessionId =
            typeof parsed.boardSessionId === 'string' && parsed.boardSessionId.trim()
                ? parsed.boardSessionId.trim()
                : null

        const base = _defaultRuntimeState()
        const boardSessionByPaper: Record<string, string> = {}
        const rawBoardSessionByPaper = parsed.boardSessionByPaper
        if (rawBoardSessionByPaper && typeof rawBoardSessionByPaper === 'object') {
            for (const [paperId, sessionId] of Object.entries(rawBoardSessionByPaper as Record<string, unknown>)) {
                if (typeof sessionId === 'string' && sessionId.trim()) {
                    boardSessionByPaper[paperId] = sessionId.trim()
                }
            }
        }
        const selectedPaperId =
            typeof parsed.selectedPaperId === 'string' && parsed.selectedPaperId.trim()
                ? parsed.selectedPaperId.trim()
                : null
        // LEGACY MIGRATION: Handle old `boardSessionId` stored at the root of the runtime state.
        // This can be removed in a future version after users have migrated.
        if (legacyBoardSessionId && selectedPaperId && !boardSessionByPaper[selectedPaperId]) {
            boardSessionByPaper[selectedPaperId] = legacyBoardSessionId
        }

        return {
            version: typeof parsed.version === 'number' ? parsed.version : base.version,
            selectedPaperId,
            paperCache: _normalizePaperCache(parsed.paperCache),
            boardSessionByPaper,
            agentTasks: Array.isArray(parsed.agentTasks) ? (parsed.agentTasks as AgentTask[]) : [],
            pipelinePhase:
                typeof parsed.pipelinePhase === 'string' ? (parsed.pipelinePhase as PipelinePhase) : base.pipelinePhase,
            e2eState: (parsed.e2eState ?? null) as E2EState | null,
            sandboxFiles: Array.isArray(parsed.sandboxFiles) ? (parsed.sandboxFiles as SandboxFileEntry[]) : [],
            timeEstimate: (parsed.timeEstimate ?? null) as TimeEstimate | null,
        }
    } catch (e) {
        console.error('Failed to load studio runtime from localStorage:', e)
        return _defaultRuntimeState()
    }
}

function saveRuntimeStateToStorage(runtime: PersistedRuntimeState): void {
    if (typeof window === 'undefined') return
    try {
        localStorage.setItem(RUNTIME_STORAGE_KEY, JSON.stringify(runtime))
    } catch (e) {
        console.error('Failed to save studio runtime to localStorage:', e)
    }
}

interface StudioState {
    // Paper management
    papers: StudioPaper[]
    selectedPaperId: string | null

    // Per-paper state cache (not cleared on switch)
    _paperCache: Record<string, PerPaperCache>

    // Task management (scoped to selected paper)
    tasks: Task[]
    activeTaskId: string | null
    selectedFileForDiff: string | null
    paperDraft: PaperDraft
    lastGenCodeResult: GenCodeResult | null
    workspaceSnapshotId: number | null

    // P2C state (scoped to selected paper)
    contextPack: ReproContextPack | null
    contextPackLoading: boolean
    contextPackError: string | null
    generationProgress: StageProgressEvent[]
    liveObservations: StageObservationsEvent[]

    // Agent Board state
    agentTasks: AgentTask[]
    boardSessionId: string | null
    boardSessionByPaper: Record<string, string>
    pipelinePhase: PipelinePhase
    e2eState: E2EState | null
    sandboxFiles: SandboxFileEntry[]
    timeEstimate: TimeEstimate | null
    setBoardSessionId: (id: string | null) => void
    replaceAgentTasksForPaper: (paperId: string, tasks: AgentTask[]) => void
    addAgentTask: (task: Omit<AgentTask, 'createdAt' | 'updatedAt'> & { id?: string }) => string
    updateAgentTask: (taskId: string, updates: Partial<AgentTask>) => void
    moveAgentTask: (taskId: string, status: AgentTask['status']) => void
    clearAgentTasks: () => void
    setPipelinePhase: (phase: PipelinePhase) => void
    setE2EState: (state: Partial<E2EState>) => void
    setSandboxFiles: (files: SandboxFileEntry[]) => void
    setTimeEstimate: (estimate: TimeEstimate) => void

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
    appendToLastAction: (taskId: string, text: string) => void
    setActiveTask: (taskId: string | null) => void
    setSelectedFileForDiff: (filename: string | null) => void
    setPaperDraft: (partial: Partial<PaperDraft>) => void
    setLastGenCodeResult: (result: GenCodeResult | null) => void
    setWorkspaceSnapshotId: (snapshotId: number | null) => void

    // P2C actions
    setContextPack: (pack: ReproContextPack | null) => void
    setContextPackLoading: (loading: boolean) => void
    setContextPackError: (error: string | null) => void
    appendGenerationProgress: (event: StageProgressEvent) => void
    clearGenerationProgress: () => void
    appendLiveObservations: (event: StageObservationsEvent) => void
    clearLiveObservations: () => void
}

export const useStudioStore = create<StudioState>((set, get) => ({
    // Paper state
    papers: [],
    selectedPaperId: null,
    _paperCache: {},

    // Task state
    tasks: [],
    activeTaskId: null,
    selectedFileForDiff: null,
    paperDraft: { title: "", abstract: "", methodSection: "" },
    lastGenCodeResult: null,
    workspaceSnapshotId: null,

    // P2C state
    contextPack: null,
    contextPackLoading: false,
    contextPackError: null,
    generationProgress: [],
    liveObservations: [],

    // Agent Board state
    agentTasks: [],
    boardSessionId: null,
    boardSessionByPaper: {},
    pipelinePhase: 'idle' as PipelinePhase,
    e2eState: null,
    sandboxFiles: [],
    timeEstimate: null,

    setBoardSessionId: (id) => {
        set((state) => {
            const selectedPaperId = state.selectedPaperId
            if (!selectedPaperId) {
                return { boardSessionId: id }
            }
            const nextMap = { ...state.boardSessionByPaper }
            if (id && id.trim()) {
                nextMap[selectedPaperId] = id.trim()
            } else {
                delete nextMap[selectedPaperId]
            }
            const newPapers = state.papers.map((paper) =>
                paper.id === selectedPaperId
                    ? {
                        ...paper,
                        boardSessionId: id || undefined,
                        updatedAt: new Date().toISOString(),
                    }
                    : paper,
            )
            savePapersToStorage(newPapers)
            return { boardSessionId: id, boardSessionByPaper: nextMap, papers: newPapers }
        })
    },

    clearAgentTasks: () => {
        set((state) => {
            const selectedPaperId = state.selectedPaperId
            if (!selectedPaperId) {
                return {
                    agentTasks: [],
                    boardSessionId: null,
                    boardSessionByPaper: {},
                    pipelinePhase: 'idle',
                    e2eState: null,
                    sandboxFiles: [],
                    timeEstimate: null,
                }
            }
            const nextMap = { ...state.boardSessionByPaper }
            delete nextMap[selectedPaperId]
            const newPapers = state.papers.map((paper) =>
                paper.id === selectedPaperId
                    ? {
                        ...paper,
                        boardSessionId: undefined,
                        updatedAt: new Date().toISOString(),
                    }
                    : paper,
            )
            savePapersToStorage(newPapers)
            return {
                agentTasks: state.agentTasks.filter(task => task.paperId !== selectedPaperId),
                boardSessionId: null,
                boardSessionByPaper: nextMap,
                papers: newPapers,
                pipelinePhase: 'idle',
                e2eState: null,
                sandboxFiles: [],
                timeEstimate: null,
            }
        })
    },

    replaceAgentTasksForPaper: (paperId, tasks) => {
        set((state) => ({
            agentTasks: [
                ...state.agentTasks.filter(task => task.paperId !== paperId),
                ...tasks.map(task => ({
                    ...task,
                    paperId: task.paperId || paperId,
                    executionLog: task.executionLog || [],
                })),
            ],
        }))
    },

    setPipelinePhase: (phase) => set({ pipelinePhase: phase }),

    setE2EState: (partial) => set((state) => ({
        e2eState: state.e2eState
            ? { ...state.e2eState, ...partial }
            : { status: 'waiting', attempt: 0, maxAttempts: 3, entryPoint: null, command: null, lastExitCode: null, lastStdout: '', lastStderr: '', history: [], ...partial } as E2EState,
    })),

    setSandboxFiles: (files) => set({ sandboxFiles: files }),

    setTimeEstimate: (estimate) => set({ timeEstimate: estimate }),

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
                contextPack: null,
                contextPackLoading: false,
                contextPackError: null,
                generationProgress: [],
                liveObservations: [],
                boardSessionId: null,
                pipelinePhase: 'idle',
                e2eState: null,
                sandboxFiles: [],
                timeEstimate: null,
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
            // Remove from cache
            const newCache = { ...state._paperCache }
            delete newCache[paperId]
            const nextBoardSessionByPaper = { ...state.boardSessionByPaper }
            delete nextBoardSessionByPaper[paperId]
            return {
                papers: newPapers,
                selectedPaperId: newSelectedPaperId,
                _paperCache: newCache,
                boardSessionByPaper: nextBoardSessionByPaper,
                agentTasks: state.agentTasks.filter(task => task.paperId !== paperId),
                boardSessionId:
                    newSelectedPaperId && nextBoardSessionByPaper[newSelectedPaperId]
                        ? nextBoardSessionByPaper[newSelectedPaperId]
                        : null,
                // Clear draft if deleted paper was selected
                ...(state.selectedPaperId === paperId ? {
                    paperDraft: { title: '', abstract: '', methodSection: '' },
                    lastGenCodeResult: null,
                    contextPack: null,
                    contextPackLoading: false,
                    contextPackError: null,
                    generationProgress: [],
                    liveObservations: [],
                } : {}),
            }
        })
    },

    selectPaper: (paperId) => {
        const state = get()

        // Save current paper's state to cache before switching
        const prevCache = { ...state._paperCache }
        if (state.selectedPaperId) {
            prevCache[state.selectedPaperId] = {
                contextPack: state.contextPack,
                contextPackLoading: state.contextPackLoading,
                contextPackError: state.contextPackError,
                generationProgress: state.generationProgress,
                liveObservations: state.liveObservations,
                activeTaskId: state.activeTaskId,
            }
        }

        const paper = paperId ? state.papers.find(p => p.id === paperId) : null
        const cached = paperId ? prevCache[paperId] : undefined

        set({
            selectedPaperId: paperId,
            _paperCache: prevCache,
            // Sync paperDraft with selected paper
            paperDraft: paper
                ? { title: paper.title, abstract: paper.abstract, methodSection: paper.methodSection || '' }
                : { title: '', abstract: '', methodSection: '' },
            // Load paper's lastGenCodeResult if available
            lastGenCodeResult: paper?.lastGenCodeResult || null,
            workspaceSnapshotId: null,
            // Restore cached per-paper state, or defaults
            activeTaskId: cached?.activeTaskId ?? null,
            contextPack: cached?.contextPack ?? null,
            contextPackLoading: cached?.contextPackLoading ?? false,
            contextPackError: cached?.contextPackError ?? null,
            generationProgress: cached?.generationProgress ?? [],
            liveObservations: cached?.liveObservations ?? [],
            boardSessionId: paperId
                ? (state.boardSessionByPaper[paperId] ?? paper?.boardSessionId ?? null)
                : null,
        })
    },

    loadPapers: () => {
        const papers = loadPapersFromStorage()
        const runtime = loadRuntimeStateFromStorage()
        const mergedBoardSessionByPaper = { ...runtime.boardSessionByPaper }
        for (const paper of papers) {
            if (paper.boardSessionId && !mergedBoardSessionByPaper[paper.id]) {
                mergedBoardSessionByPaper[paper.id] = paper.boardSessionId
            }
        }
        const selectedPaperId =
            runtime.selectedPaperId && papers.some(p => p.id === runtime.selectedPaperId)
                ? runtime.selectedPaperId
                : null
        const selectedPaper = selectedPaperId ? papers.find(p => p.id === selectedPaperId) || null : null
        const cached = selectedPaperId ? runtime.paperCache[selectedPaperId] : undefined
        const boardSessionId =
            selectedPaperId && mergedBoardSessionByPaper[selectedPaperId]
                ? mergedBoardSessionByPaper[selectedPaperId]
                : null

        set({
            papers,
            selectedPaperId,
            _paperCache: runtime.paperCache,
            agentTasks: runtime.agentTasks,
            boardSessionByPaper: mergedBoardSessionByPaper,
            boardSessionId,
            pipelinePhase: runtime.pipelinePhase,
            e2eState: runtime.e2eState,
            sandboxFiles: runtime.sandboxFiles,
            timeEstimate: runtime.timeEstimate,
            paperDraft: selectedPaper
                ? {
                    title: selectedPaper.title,
                    abstract: selectedPaper.abstract,
                    methodSection: selectedPaper.methodSection || '',
                }
                : { title: '', abstract: '', methodSection: '' },
            lastGenCodeResult: selectedPaper?.lastGenCodeResult || null,
            workspaceSnapshotId: null,
            activeTaskId: cached?.activeTaskId ?? null,
            contextPack: cached?.contextPack ?? null,
            contextPackLoading: false,
            contextPackError: cached?.contextPackError ?? null,
            generationProgress: cached?.generationProgress ?? [],
            liveObservations: cached?.liveObservations ?? [],
        })
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

    addAgentTask: (task) => {
        const id = task.id?.trim() || `agent-task-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`
        const now = new Date().toISOString()
        set(state => ({
            agentTasks: [
                ...state.agentTasks,
                {
                    ...task,
                    id,
                    createdAt: now,
                    updatedAt: now,
                    executionLog: task.executionLog || [],
                    paperId: task.paperId || state.selectedPaperId || undefined,
                },
            ],
        }))
        return id
    },

    updateAgentTask: (taskId, updates) => {
        set(state => ({
            agentTasks: state.agentTasks.map(task =>
                task.id === taskId
                    ? { ...task, ...updates, updatedAt: new Date().toISOString() }
                    : task
            ),
        }))
    },

    moveAgentTask: (taskId, status) => {
        set(state => ({
            agentTasks: state.agentTasks.map(task =>
                task.id === taskId
                    ? { ...task, status, updatedAt: new Date().toISOString() }
                    : task
            ),
        }))
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
            id: `action-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`,
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

    appendToLastAction: (taskId, text) => {
        set(state => ({
            tasks: state.tasks.map(t => {
                if (t.id !== taskId || t.actions.length === 0) return t
                const last = t.actions[t.actions.length - 1]
                if (last.type !== 'text') return t
                const updated = { ...last, content: last.content + text }
                return { ...t, actions: [...t.actions.slice(0, -1), updated] }
            })
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

    setContextPack: (pack) => {
        set((state) => {
            if (!state.selectedPaperId || !pack?.context_pack_id) {
                return { contextPack: pack }
            }
            const newPapers = state.papers.map((paper) =>
                paper.id === state.selectedPaperId
                    ? {
                        ...paper,
                        contextPackId: pack.context_pack_id,
                        updatedAt: new Date().toISOString(),
                    }
                    : paper,
            )
            savePapersToStorage(newPapers)
            return { contextPack: pack, papers: newPapers }
        })
    },
    setContextPackLoading: (loading) => set({ contextPackLoading: loading }),
    setContextPackError: (error) => set({ contextPackError: error }),
    appendGenerationProgress: (event) => set((state) => ({
        generationProgress: [...state.generationProgress, event],
    })),
    clearGenerationProgress: () => set({ generationProgress: [] }),
    appendLiveObservations: (event) => set((state) => {
        const existingIndex = state.liveObservations.findIndex(item => item.stage === event.stage)
        if (existingIndex === -1) {
            return { liveObservations: [...state.liveObservations, event] }
        }
        const updated = [...state.liveObservations]
        updated[existingIndex] = event
        return { liveObservations: updated }
    }),
    clearLiveObservations: () => set({ liveObservations: [] }),
}))

function _snapshotRuntimeState(state: StudioState): PersistedRuntimeState {
    const paperCache = _normalizePaperCache(state._paperCache)
    if (state.selectedPaperId) {
        paperCache[state.selectedPaperId] = {
            contextPack: state.contextPack,
            contextPackLoading: false,
            contextPackError: state.contextPackError,
            generationProgress: state.generationProgress,
            liveObservations: state.liveObservations,
            activeTaskId: state.activeTaskId,
        }
    }

    const boardSessionByPaper = { ...state.boardSessionByPaper }
    if (state.selectedPaperId && state.boardSessionId) {
        boardSessionByPaper[state.selectedPaperId] = state.boardSessionId
    }

    return {
        version: RUNTIME_STORAGE_VERSION,
        selectedPaperId: state.selectedPaperId,
        paperCache,
        boardSessionByPaper,
        agentTasks: state.agentTasks,
        pipelinePhase: state.pipelinePhase,
        e2eState: state.e2eState,
        sandboxFiles: state.sandboxFiles,
        timeEstimate: state.timeEstimate,
    }
}

let _runtimeSubscriptionAttached = false
if (typeof window !== 'undefined' && !_runtimeSubscriptionAttached) {
    _runtimeSubscriptionAttached = true
    useStudioStore.subscribe((state) => {
        saveRuntimeStateToStorage(_snapshotRuntimeState(state))
    })
}
