export type PaperType = "experimental" | "theoretical" | "survey" | "benchmark" | "system"

export type ObservationType =
  | "method"
  | "architecture"
  | "hyperparameter"
  | "metric"
  | "environment"
  | "limitation"

export type ObservationConcept =
  | "core_method"
  | "gotcha"
  | "trade_off"
  | "limitation"
  | "baseline"
  | "reproduction_hint"

export interface EvidenceLink {
  type: "paper_span" | "table" | "figure" | "code_snippet" | "metadata"
  ref: string
  supports: string[]
  confidence: number
}

export interface ExtractionObservation {
  id: string
  stage: string
  type: ObservationType
  title: string
  narrative: string
  confidence: number
  concepts: ObservationConcept[]
  structured_data?: Record<string, unknown>
  evidence?: EvidenceLink[]
}

export interface PaperIdentity {
  paper_id: string
  title: string
  year: number
  authors: string[]
  identifiers: Record<string, string>
}

export interface TaskCheckpoint {
  id: string
  title: string
  description: string
  acceptance_criteria: string[]
  depends_on: string[]
  estimated_difficulty: "low" | "medium" | "high"
}

export interface ConfidenceScores {
  overall: number
  literature: number
  blueprint: number
  environment: number
  spec: number
  roadmap: number
  metrics: number
}

export interface ReproContextPack {
  context_pack_id: string
  version: string
  created_at: string
  paper: PaperIdentity
  paper_type: PaperType
  objective: string
  observations: ExtractionObservation[]
  task_roadmap: TaskCheckpoint[]
  confidence: ConfidenceScores
  warnings: string[]
}

export interface StageProgressEvent {
  stage: string
  progress: number
  message?: string
}

export interface StageObservationsEvent {
  stage: string
  observations: Array<{
    id: string
    type: ObservationType
    title: string
    confidence: number
  }>
}

export interface GenerateCompletedEvent {
  context_pack_id: string
  status: "completed"
  summary: string
  confidence: ConfidenceScores
  warnings: string[]
  next_action: "create_repro_session"
}

export interface GenerateErrorEvent {
  error?: string
  message?: string
  partial_pack_id?: string
}

export interface ContextPackSummary {
  context_pack_id: string
  paper_id?: string
  paper_title?: string
  depth?: string
  status: string
  confidence_overall: number
  warning_count: number
  created_at?: string | null
}

export interface SessionStep {
  step_id: string
  title: string
  command?: string
  status?: string
}

export interface ContextPackSession {
  session_id: string
  runbook_id?: string
  initial_steps: SessionStep[]
  initial_prompt: string
}

export type GenerationStatus = "idle" | "generating" | "completed" | "error"
