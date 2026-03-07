export type SkillTone = "teal" | "cyan" | "amber" | "blue" | "rose" | "indigo"
export type SkillStatusTone = "success" | "warning" | "neutral" | "accent"

export type SkillStatus = {
  label: string
  tone: SkillStatusTone
}

export type SkillSummaryStat = {
  label: string
  value: string
  caption: string
}

export type SkillRuntimeSignal = {
  label: string
  value: string
  tone: SkillStatusTone
}

export type SkillOutputItem = {
  name: string
  path: string
  updatedAt: string
}

export type SkillWorkspaceFact = {
  label: string
  value: string
}

export type SkillPipelineStep = {
  step: string
  title: string
  owner: string
  skillId: string
  body: string
  status: string
}

export type SkillCardData = {
  id: string
  title: string
  headline: string
  description: string
  category: string
  tone: SkillTone
  status: SkillStatus
  readiness: number
  footprint: number
  signalLabel: string
  signalValue: string
  signalCaption: string
  prerequisites: string[]
  outputs: string[]
  sourcePaths: string[]
  updatedAt: string
  requiresAi: boolean
  tags: string[]
}

export type SkillsStudioData = {
  title: string
  subtitle: string
  summary: SkillSummaryStat[]
  signals: SkillRuntimeSignal[]
  categories: string[]
  statusOptions: string[]
  skills: SkillCardData[]
  pipeline: SkillPipelineStep[]
  recentOutputs: SkillOutputItem[]
  workspaceFacts: SkillWorkspaceFact[]
}
