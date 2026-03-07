import "server-only"

import { promises as fs } from "node:fs"
import path from "node:path"

import type {
  SkillCardData,
  SkillsStudioData,
  SkillStatusTone,
  SkillTone,
} from "@/lib/skills-studio-types"

type SkillBlueprint = {
  id: string
  title: string
  headline: string
  description: string
  category: string
  tone: SkillTone
  paths: string[]
  requiresAi: boolean
  signalLabel: string
  signalKey:
    | "conferenceAgentCount"
    | "scholarCount"
    | "researchPanelCount"
    | "studioPanelCount"
    | "reportTemplateCount"
    | "workflowNodeCount"
  prerequisites: string[]
  outputs: string[]
}

type WorkspaceSnapshot = {
  anthropic: boolean
  github: boolean
  scholarCount: number
  checkInterval: string
  outputArtifactCount: number
  reportCount: number
  reportTemplateCount: number
  pageRouteCount: number
  apiRouteCount: number
  dashboardPanelCount: number
  researchPanelCount: number
  studioPanelCount: number
  reproAgentCount: number
  workflowNodeCount: number
  conferenceAgentCount: number
  recentOutputs: SkillsStudioData["recentOutputs"]
}

const TEXT_SUFFIXES = new Set([".py", ".ts", ".tsx", ".md", ".j2", ".yaml", ".yml", ".json", ".txt"])
const OUTPUT_SUFFIXES = new Set([".md", ".json", ".csv", ".txt"])
const IGNORED_NAMES = new Set([".git", "node_modules", "__pycache__", ".next", ".venv", "venv"])

const SKILL_BLUEPRINTS: SkillBlueprint[] = [
  {
    id: "feed-harvest",
    title: "Feed Harvest",
    headline: "Pull conference papers and feed signals into a unified intake layer.",
    description:
      "Connects conference agents, harvest routes, and feed workflows so the system can ingest fresh papers before any downstream review starts.",
    category: "Discovery",
    tone: "teal",
    paths: [
      "src/paperbot/agents/conference",
      "src/paperbot/api/routes/harvest.py",
      "src/paperbot/workflows/feed.py",
      "web/src/app/papers/page.tsx",
    ],
    requiresAi: false,
    signalLabel: "Conference connectors",
    signalKey: "conferenceAgentCount",
    prerequisites: [
      "Conference and feed sources configured",
      "Harvest endpoints reachable from the app runtime",
    ],
    outputs: [
      "Feed-ready paper metadata",
      "Harvested paper batches",
      "Seed material for research and review flows",
    ],
  },
  {
    id: "scholar-tracking",
    title: "Scholar Tracking",
    headline: "Monitor tracked researchers and surface new papers without manual polling.",
    description:
      "Pairs Semantic Scholar-aware agents with the scholars UI so subscriptions, updates, and alerts stay visible inside the main product.",
    category: "Tracking",
    tone: "cyan",
    paths: [
      "src/paperbot/agents/scholar_tracking",
      "src/paperbot/workflows/scholar_tracking.py",
      "config/scholar_subscriptions.yaml",
      "web/src/app/scholars/page.tsx",
      "web/src/components/scholars",
    ],
    requiresAi: false,
    signalLabel: "Watched scholars",
    signalKey: "scholarCount",
    prerequisites: [
      "Subscribed scholars saved in configuration",
      "Semantic Scholar access available to the backend",
    ],
    outputs: [
      "New paper detections",
      "Scholar-centric watchlists",
      "Schedule-aware tracking runs",
    ],
  },
  {
    id: "research-workspace",
    title: "Research Workspace",
    headline: "Turn raw papers into navigable discovery boards and topic workflows.",
    description:
      "Combines the research route, research panels, and backend research APIs into one operator-facing workspace for triage and curation.",
    category: "Exploration",
    tone: "amber",
    paths: [
      "web/src/app/research/page.tsx",
      "web/src/app/research/discovery/page.tsx",
      "web/src/components/research",
      "src/paperbot/api/routes/research.py",
      "src/paperbot/api/routes/feed.py",
    ],
    requiresAi: false,
    signalLabel: "Research panels",
    signalKey: "researchPanelCount",
    prerequisites: [
      "Research routes mounted in the web app",
      "Collections and feed APIs available on the backend",
    ],
    outputs: [
      "Discovery views and topic boards",
      "Saved paper collections",
      "Research pipeline entry points",
    ],
  },
  {
    id: "deepcode-studio",
    title: "DeepCode Studio",
    headline: "Move from papers to runnable reproduction workspaces.",
    description:
      "Brings together the studio route, reproduction agents, runbook endpoints, and execution panels into the coding surface for implementation work.",
    category: "Execution",
    tone: "blue",
    paths: [
      "web/src/app/studio/page.tsx",
      "web/src/components/studio",
      "src/paperbot/repro",
      "src/paperbot/api/routes/studio_chat.py",
      "src/paperbot/api/routes/runbook.py",
      "src/paperbot/api/routes/sandbox.py",
    ],
    requiresAi: true,
    signalLabel: "Studio panels",
    signalKey: "studioPanelCount",
    prerequisites: [
      "Studio frontend route enabled",
      "LLM credentials and reproduction services configured",
    ],
    outputs: [
      "Code generation sessions",
      "Runbook and sandbox actions",
      "Reproduction logs and workspace state",
    ],
  },
  {
    id: "review-reporting",
    title: "Review and Reporting",
    headline: "Score paper quality and package findings into reviewer-facing artifacts.",
    description:
      "Links review agents, documentation paths, report templates, and paper detail surfaces into the final presentation layer for decisions.",
    category: "Review",
    tone: "rose",
    paths: [
      "src/paperbot/agents/review",
      "src/paperbot/agents/documentation",
      "src/paperbot/api/routes/review.py",
      "reports/templates",
      "web/src/app/papers/[id]/page.tsx",
    ],
    requiresAi: true,
    signalLabel: "Report templates",
    signalKey: "reportTemplateCount",
    prerequisites: [
      "Review and documentation agents available",
      "Templates present in reports/templates",
    ],
    outputs: [
      "Review summaries",
      "Markdown report payloads",
      "Decision-ready paper detail context",
    ],
  },
  {
    id: "workflow-orchestration",
    title: "Workflow Orchestration",
    headline: "Coordinate jobs, nodes, and long-running automation across the platform.",
    description:
      "Maps backend workflow nodes and workflow pages into one control surface so operators can see how discovery, tracking, and execution connect.",
    category: "Automation",
    tone: "indigo",
    paths: [
      "src/paperbot/workflows",
      "src/paperbot/api/routes/jobs.py",
      "src/paperbot/api/routes/track.py",
      "web/src/app/workflows/page.tsx",
      "web/src/components/dashboard/PipelineStatus.tsx",
    ],
    requiresAi: false,
    signalLabel: "Workflow nodes",
    signalKey: "workflowNodeCount",
    prerequisites: [
      "Workflow nodes defined in the backend",
      "Jobs and tracking routes available for orchestration",
    ],
    outputs: [
      "Scheduled and on-demand workflow runs",
      "Pipeline state overviews",
      "Cross-surface automation context",
    ],
  },
]

function getWorkspaceRoot() {
  const cwd = process.cwd()
  return path.basename(cwd).toLowerCase() === "web" ? path.resolve(cwd, "..") : cwd
}

async function pathExists(targetPath: string) {
  try {
    await fs.access(targetPath)
    return true
  } catch {
    return false
  }
}

async function collectFiles(entryPath: string): Promise<string[]> {
  if (!(await pathExists(entryPath))) {
    return []
  }

  const stat = await fs.stat(entryPath)
  if (stat.isFile()) {
    return [entryPath]
  }

  const stack = [entryPath]
  const files: string[] = []

  while (stack.length > 0) {
    const current = stack.pop()
    if (!current) continue

    const entries = await fs.readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      if (IGNORED_NAMES.has(entry.name)) continue
      const nextPath = path.join(current, entry.name)
      if (entry.isDirectory()) {
        stack.push(nextPath)
      } else if (entry.isFile()) {
        files.push(nextPath)
      }
    }
  }

  return files
}

async function countFiles(entryPath: string, matcher?: (filePath: string) => boolean) {
  const files = await collectFiles(entryPath)
  return matcher ? files.filter(matcher).length : files.length
}

async function countLines(targetPaths: string[]) {
  let total = 0

  for (const targetPath of targetPaths) {
    const files = await collectFiles(targetPath)
    for (const filePath of files) {
      if (!TEXT_SUFFIXES.has(path.extname(filePath).toLowerCase())) continue
      const content = await fs.readFile(filePath, "utf8").catch(async () => {
        const buffer = await fs.readFile(filePath)
        return buffer.toString("utf8")
      })
      total += content.split(/\r?\n/).length
    }
  }

  return total
}

async function getLatestTimestamp(targetPaths: string[]) {
  let latest = 0

  for (const targetPath of targetPaths) {
    const files = await collectFiles(targetPath)
    for (const filePath of files) {
      const stat = await fs.stat(filePath)
      latest = Math.max(latest, stat.mtimeMs)
    }

    if (await pathExists(targetPath)) {
      const stat = await fs.stat(targetPath)
      latest = Math.max(latest, stat.mtimeMs)
    }
  }

  return latest
}

function toRepoRelative(workspaceRoot: string, targetPath: string) {
  return path.relative(workspaceRoot, targetPath).split(path.sep).join("/")
}

function formatTimestamp(timestampMs: number) {
  if (!timestampMs) return "-"
  const date = new Date(timestampMs)
  const year = date.getFullYear()
  const month = `${date.getMonth() + 1}`.padStart(2, "0")
  const day = `${date.getDate()}`.padStart(2, "0")
  const hours = `${date.getHours()}`.padStart(2, "0")
  const minutes = `${date.getMinutes()}`.padStart(2, "0")
  return `${year}-${month}-${day} ${hours}:${minutes}`
}

async function countScholars(configPath: string) {
  if (!(await pathExists(configPath))) return 0
  const content = await fs.readFile(configPath, "utf8")
  const matches = content.match(/semantic_scholar_id\s*:/g)
  return matches?.length ?? 0
}

async function readCheckInterval(configPath: string) {
  if (!(await pathExists(configPath))) return "Not configured"
  const content = await fs.readFile(configPath, "utf8")
  const match = content.match(/check_interval\s*:\s*['"]?([A-Za-z_-]+)['"]?/)?.[1]
  if (!match) return "Not configured"
  return match.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase())
}

async function collectRecentOutputs(workspaceRoot: string, limit = 6): Promise<SkillsStudioData["recentOutputs"]> {
  const outputDir = path.join(workspaceRoot, "output")
  const reportsDir = path.join(workspaceRoot, "reports")
  const candidates: Array<{ path: string; mtimeMs: number }> = []

  if (await pathExists(outputDir)) {
    const outputFiles = await collectFiles(outputDir)
    for (const filePath of outputFiles) {
      if (path.basename(filePath).startsWith(".")) continue
      if (!OUTPUT_SUFFIXES.has(path.extname(filePath).toLowerCase())) continue
      const stat = await fs.stat(filePath)
      candidates.push({ path: filePath, mtimeMs: stat.mtimeMs })
    }
  }

  if (await pathExists(reportsDir)) {
    const reportFiles = await collectFiles(reportsDir)
    for (const filePath of reportFiles) {
      if (!filePath.endsWith(".md")) continue
      if (filePath.includes(`${path.sep}templates${path.sep}`)) continue
      const stat = await fs.stat(filePath)
      candidates.push({ path: filePath, mtimeMs: stat.mtimeMs })
    }
  }

  return candidates
    .sort((left, right) => right.mtimeMs - left.mtimeMs)
    .slice(0, limit)
    .map((item) => ({
      name: path.basename(item.path),
      path: toRepoRelative(workspaceRoot, item.path),
      updatedAt: formatTimestamp(item.mtimeMs),
    }))
}

async function buildWorkspaceSnapshot(workspaceRoot: string): Promise<WorkspaceSnapshot> {
  const configPath = path.join(workspaceRoot, "config", "scholar_subscriptions.yaml")
  const reportsTemplatesPath = path.join(workspaceRoot, "reports", "templates")
  const reportsRootPath = path.join(workspaceRoot, "reports")
  const webAppPath = path.join(workspaceRoot, "web", "src", "app")
  const webDashboardPath = path.join(workspaceRoot, "web", "src", "components", "dashboard")
  const webResearchPath = path.join(workspaceRoot, "web", "src", "components", "research")
  const webStudioPath = path.join(workspaceRoot, "web", "src", "components", "studio")
  const workflowNodesPath = path.join(workspaceRoot, "src", "paperbot", "workflows", "nodes")
  const conferenceAgentsPath = path.join(workspaceRoot, "src", "paperbot", "agents", "conference")
  const reproAgentsPath = path.join(workspaceRoot, "src", "paperbot", "repro", "agents")
  const outputPath = path.join(workspaceRoot, "output")

  const outputFiles = (await collectFiles(outputPath)).filter((filePath) => !path.basename(filePath).startsWith("."))
  const reportFiles = (await collectFiles(reportsRootPath)).filter(
    (filePath) => filePath.endsWith(".md") && !filePath.includes(`${path.sep}templates${path.sep}`),
  )
  const reportCount = outputFiles.filter((filePath) => filePath.endsWith(".md")).length + reportFiles.length

  return {
    anthropic: Boolean(process.env.ANTHROPIC_API_KEY),
    github: Boolean(process.env.GITHUB_TOKEN || process.env.GH_TOKEN || process.env.GITHUB_API_TOKEN),
    scholarCount: await countScholars(configPath),
    checkInterval: await readCheckInterval(configPath),
    outputArtifactCount: outputFiles.length,
    reportCount,
    reportTemplateCount: await countFiles(reportsTemplatesPath, (filePath) => [".j2", ".md", ".txt"].includes(path.extname(filePath).toLowerCase())),
    pageRouteCount: await countFiles(webAppPath, (filePath) => path.basename(filePath) === "page.tsx" && !filePath.includes(`${path.sep}api${path.sep}`)),
    apiRouteCount: await countFiles(path.join(webAppPath, "api"), (filePath) => path.basename(filePath) === "route.ts"),
    dashboardPanelCount: await countFiles(webDashboardPath, (filePath) => filePath.endsWith(".tsx")),
    researchPanelCount: await countFiles(webResearchPath, (filePath) => filePath.endsWith(".tsx")),
    studioPanelCount: await countFiles(webStudioPath, (filePath) => filePath.endsWith(".tsx")),
    reproAgentCount: await countFiles(reproAgentsPath, (filePath) => filePath.endsWith(".py") && !filePath.endsWith("__init__.py")),
    workflowNodeCount: await countFiles(workflowNodesPath, (filePath) => filePath.endsWith(".py") && !filePath.endsWith("__init__.py")),
    conferenceAgentCount: await countFiles(conferenceAgentsPath, (filePath) => filePath.endsWith(".py") && !filePath.endsWith("__init__.py")),
    recentOutputs: await collectRecentOutputs(workspaceRoot),
  }
}

function buildStatus(blueprint: SkillBlueprint, snapshot: WorkspaceSnapshot): SkillStatus & { readiness: number } {
  const aiPreview = blueprint.requiresAi && !snapshot.anthropic

  switch (blueprint.id) {
    case "scholar-tracking":
      return snapshot.scholarCount > 0
        ? { label: "Configured", tone: "accent", readiness: 94 }
        : { label: "Needs setup", tone: "warning", readiness: 48 }
    case "deepcode-studio":
      if (aiPreview) return { label: "Preview", tone: "warning", readiness: 72 }
      return { label: "Live", tone: "success", readiness: 96 }
    case "review-reporting":
      if (aiPreview) return { label: "Preview", tone: "warning", readiness: 68 }
      return snapshot.reportCount > 0
        ? { label: "Live", tone: "success", readiness: 95 }
        : { label: "Ready", tone: "neutral", readiness: 86 }
    case "workflow-orchestration":
      return snapshot.workflowNodeCount > 0
        ? { label: "Live", tone: "success", readiness: 100 }
        : { label: "Ready", tone: "neutral", readiness: 82 }
    case "feed-harvest":
      return snapshot.outputArtifactCount > 0
        ? { label: "Live", tone: "success", readiness: 92 }
        : { label: "Ready", tone: "neutral", readiness: 84 }
    case "research-workspace":
      return snapshot.researchPanelCount > 0
        ? { label: "Live", tone: "success", readiness: 97 }
        : { label: "Ready", tone: "neutral", readiness: 88 }
    default:
      return { label: "Ready", tone: "neutral", readiness: 82 }
  }
}

function buildSignalValue(snapshot: WorkspaceSnapshot, signalKey: SkillBlueprint["signalKey"]) {
  return String(snapshot[signalKey])
}

function buildSignalCaption(skillId: SkillBlueprint["id"], snapshot: WorkspaceSnapshot) {
  switch (skillId) {
    case "feed-harvest":
      return `${snapshot.apiRouteCount} API routes online`
    case "scholar-tracking":
      return `${snapshot.checkInterval} polling cadence`
    case "research-workspace":
      return `${snapshot.pageRouteCount} page routes shipped`
    case "deepcode-studio":
      return `${snapshot.reproAgentCount} repro agents available`
    case "review-reporting":
      return `${snapshot.reportCount} markdown outputs detected`
    case "workflow-orchestration":
      return `${snapshot.apiRouteCount} backend routes connected`
    default:
      return "Runtime signal"
  }
}

function buildSignalTone(label: string, snapshot: WorkspaceSnapshot): SkillStatusTone {
  if (label === "Anthropic") return snapshot.anthropic ? "success" : "warning"
  if (label === "GitHub token") return snapshot.github ? "accent" : "neutral"
  return "neutral"
}

export async function getSkillsStudioData(): Promise<SkillsStudioData> {
  const workspaceRoot = getWorkspaceRoot()
  const snapshot = await buildWorkspaceSnapshot(workspaceRoot)

  const skills: SkillCardData[] = []

  for (const blueprint of SKILL_BLUEPRINTS) {
    const existingPaths = [] as string[]
    for (const relativePath of blueprint.paths) {
      const absolutePath = path.join(workspaceRoot, relativePath)
      if (await pathExists(absolutePath)) {
        existingPaths.push(absolutePath)
      }
    }

    const status = buildStatus(blueprint, snapshot)
    const latestTimestamp = await getLatestTimestamp(existingPaths)
    const footprint = await countLines(existingPaths)

    skills.push({
      id: blueprint.id,
      title: blueprint.title,
      headline: blueprint.headline,
      description: blueprint.description,
      category: blueprint.category,
      tone: blueprint.tone,
      status: { label: status.label, tone: status.tone },
      readiness: status.readiness,
      footprint,
      signalLabel: blueprint.signalLabel,
      signalValue: buildSignalValue(snapshot, blueprint.signalKey),
      signalCaption: buildSignalCaption(blueprint.id, snapshot),
      prerequisites: blueprint.prerequisites,
      outputs: blueprint.outputs,
      sourcePaths: existingPaths.map((targetPath) => toRepoRelative(workspaceRoot, targetPath)),
      updatedAt: formatTimestamp(latestTimestamp),
      requiresAi: blueprint.requiresAi,
      tags: [blueprint.category, status.label, blueprint.signalLabel],
    })
  }

  const categories = ["All", ...Array.from(new Set(skills.map((skill) => skill.category))).sort()]
  const statusOptions = ["All", ...Array.from(new Set(skills.map((skill) => skill.status.label))).sort()]
  const frontendSurfaceCount = snapshot.dashboardPanelCount + snapshot.researchPanelCount + snapshot.studioPanelCount

  return {
    title: "PaperBot Skills Studio",
    subtitle:
      "Auto-discovered capability surfaces mapped from the latest dev architecture across web routes, workflow nodes, review agents, and reproduction tools.",
    summary: [
      {
        label: "Skills mapped",
        value: String(skills.length),
        caption: `${snapshot.workflowNodeCount} workflow nodes currently wired`,
      },
      {
        label: "Frontend surfaces",
        value: String(frontendSurfaceCount),
        caption: `${snapshot.pageRouteCount} routed pages across the web app`,
      },
      {
        label: "API routes",
        value: String(snapshot.apiRouteCount),
        caption: "Backend endpoints available to power the UI",
      },
      {
        label: "Output artifacts",
        value: String(snapshot.outputArtifactCount),
        caption: `${snapshot.reportCount} markdown outputs already generated`,
      },
    ],
    signals: [
      { label: "Anthropic", value: snapshot.anthropic ? "Connected" : "Missing", tone: buildSignalTone("Anthropic", snapshot) },
      { label: "GitHub token", value: snapshot.github ? "Connected" : "Optional", tone: buildSignalTone("GitHub token", snapshot) },
      { label: "Tracked scholars", value: String(snapshot.scholarCount), tone: snapshot.scholarCount > 0 ? "accent" : "warning" },
      { label: "Report templates", value: String(snapshot.reportTemplateCount), tone: snapshot.reportTemplateCount > 0 ? "success" : "warning" },
    ],
    categories,
    statusOptions,
    skills,
    pipeline: [
      {
        step: "01",
        title: "Ingest",
        owner: "Feed Harvest",
        skillId: "feed-harvest",
        body: "Capture fresh papers from conference and feed sources before routing them into the research layer.",
        status: snapshot.outputArtifactCount > 0 ? "Live" : "Ready",
      },
      {
        step: "02",
        title: "Track",
        owner: "Scholar Tracking",
        skillId: "scholar-tracking",
        body: "Monitor configured scholars and surface publication changes without manual polling.",
        status: snapshot.scholarCount > 0 ? "Configured" : "Needs setup",
      },
      {
        step: "03",
        title: "Explore",
        owner: "Research Workspace",
        skillId: "research-workspace",
        body: "Triage harvested papers inside discovery boards, collections, and research dashboards.",
        status: snapshot.researchPanelCount > 0 ? "Live" : "Ready",
      },
      {
        step: "04",
        title: "Build",
        owner: "DeepCode Studio",
        skillId: "deepcode-studio",
        body: "Transform papers into reproduction plans, generated code, and runbook-backed execution flows.",
        status: snapshot.anthropic ? "Live" : "Preview",
      },
      {
        step: "05",
        title: "Review",
        owner: "Review and Reporting",
        skillId: "review-reporting",
        body: "Score quality, synthesize findings, and prepare reviewer-facing report artifacts.",
        status: snapshot.reportTemplateCount > 0 ? snapshot.anthropic ? "Ready" : "Preview" : "Needs setup",
      },
      {
        step: "06",
        title: "Operate",
        owner: "Workflow Orchestration",
        skillId: "workflow-orchestration",
        body: "Coordinate long-running jobs and expose the health of the whole system as one operator surface.",
        status: snapshot.workflowNodeCount > 0 ? "Live" : "Ready",
      },
    ],
    recentOutputs: snapshot.recentOutputs,
    workspaceFacts: [
      { label: "Workspace", value: workspaceRoot },
      { label: "Generated at", value: formatTimestamp(Date.now()) },
      { label: "Page routes", value: String(snapshot.pageRouteCount) },
      { label: "API routes", value: String(snapshot.apiRouteCount) },
    ],
  }
}
