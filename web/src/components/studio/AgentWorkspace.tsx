"use client"

import { type MouseEvent as ReactMouseEvent, useEffect, useMemo, useRef, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import {
  Activity,
  ArrowLeft,
  Bot,
  ExternalLink,
  FileCode2,
  GitBranch,
  MessageSquare,
  Sparkles,
  Wrench,
} from "lucide-react"

import { ActivityFeed } from "@/components/agent-events/ActivityFeed"
import { AgentStatusPanel } from "@/components/agent-events/AgentStatusPanel"
import { SubagentActivityPanel } from "@/components/agent-events/SubagentActivityPanel"
import { ToolCallTimeline } from "@/components/agent-events/ToolCallTimeline"
import { AgentDagPanel } from "@/components/agent-dashboard/AgentDagPanel"
import { FileListPanel } from "@/components/agent-dashboard/FileListPanel"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useAgentEventStore, type AgentInspectorView } from "@/lib/agent-events/store"
import { buildSubagentActivityGroups } from "@/lib/agent-events/subagent-groups"
import { useAgentEvents } from "@/lib/agent-events/useAgentEvents"
import { useStudioRuntime } from "@/hooks/useStudioRuntime"
import type { StudioRuntimeInfo } from "@/lib/studio-runtime"
import { getAgentPresentation } from "@/lib/agent-runtime"
import { useStudioStore, type AgentTask, type StudioPaperStatus } from "@/lib/store/studio-store"

import { AgentBoardSidebar } from "./AgentBoardSidebar"
import { ChatHistoryPanel } from "./ChatHistoryPanel"
import { ReproductionLog } from "./ReproductionLog"

type LeftRailView = "threads" | "tasks" | "workspace"
type CenterView = "log" | "context" | "board" | "commands"
type MobileView = "threads" | "console" | "monitor"

interface AgentWorkspaceProps {
  defaultCenterView?: CenterView
  onBackToStudio?: () => void
}

const LEFT_RAIL_STORAGE_KEY = "paperbot.studio.agent-workspace.left-width"
const RIGHT_INSPECTOR_STORAGE_KEY = "paperbot.studio.agent-workspace.right-width"
const LEFT_RAIL_MIN_WIDTH = 248
const LEFT_RAIL_MAX_WIDTH = 380
const RIGHT_INSPECTOR_MIN_WIDTH = 300
const RIGHT_INSPECTOR_MAX_WIDTH = 420
const DEFAULT_LEFT_RAIL_WIDTH = 280
const DEFAULT_RIGHT_INSPECTOR_WIDTH = 320

type PanelWidths = {
  leftWidth: number
  rightWidth: number
}

type DragState = {
  side: "left" | "right"
  startX: number
  startLeftWidth: number
  startRightWidth: number
}

function readStoredWidth(key: string, fallback: number): number {
  if (typeof window === "undefined") return fallback

  const raw = window.localStorage.getItem(key)
  if (!raw) return fallback

  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), Math.max(min, max))
}

function getCenterMinWidth(containerWidth: number): number {
  if (containerWidth >= 1600) return 720
  if (containerWidth >= 1360) return 560
  return 380
}

function clampPanelWidths(
  containerWidth: number,
  leftWidth: number,
  rightWidth: number,
): PanelWidths {
  if (containerWidth <= 0) {
    return {
      leftWidth: clamp(leftWidth, LEFT_RAIL_MIN_WIDTH, LEFT_RAIL_MAX_WIDTH),
      rightWidth: clamp(rightWidth, RIGHT_INSPECTOR_MIN_WIDTH, RIGHT_INSPECTOR_MAX_WIDTH),
    }
  }

  const centerMinWidth = getCenterMinWidth(containerWidth)

  let nextLeftWidth = clamp(
    leftWidth,
    LEFT_RAIL_MIN_WIDTH,
    Math.min(LEFT_RAIL_MAX_WIDTH, containerWidth - RIGHT_INSPECTOR_MIN_WIDTH - centerMinWidth),
  )

  let nextRightWidth = clamp(
    rightWidth,
    RIGHT_INSPECTOR_MIN_WIDTH,
    Math.min(RIGHT_INSPECTOR_MAX_WIDTH, containerWidth - nextLeftWidth - centerMinWidth),
  )

  nextLeftWidth = clamp(
    nextLeftWidth,
    LEFT_RAIL_MIN_WIDTH,
    Math.min(LEFT_RAIL_MAX_WIDTH, containerWidth - nextRightWidth - centerMinWidth),
  )

  nextRightWidth = clamp(
    nextRightWidth,
    RIGHT_INSPECTOR_MIN_WIDTH,
    Math.min(RIGHT_INSPECTOR_MAX_WIDTH, containerWidth - nextLeftWidth - centerMinWidth),
  )

  return { leftWidth: nextLeftWidth, rightWidth: nextRightWidth }
}

function normalizeCenterView(value: string | null | undefined, fallback: CenterView = "log"): CenterView {
  if (value === "context" || value === "board" || value === "log" || value === "commands") return value
  return fallback
}

function centerViewToMobileView(view: CenterView): MobileView {
  return view === "board" ? "monitor" : "console"
}

function paperStatusLabel(status: StudioPaperStatus): string {
  if (status === "generating") return "Generating"
  if (status === "ready") return "Ready"
  if (status === "running") return "Running"
  if (status === "completed") return "Completed"
  if (status === "error") return "Error"
  return "Draft"
}

function paperStatusClassName(status: StudioPaperStatus): string {
  if (status === "generating") return "border-sky-200 bg-sky-50 text-sky-700"
  if (status === "ready") return "border-emerald-200 bg-emerald-50 text-emerald-700"
  if (status === "running") return "border-violet-200 bg-violet-50 text-violet-700"
  if (status === "completed") return "border-emerald-200 bg-emerald-50 text-emerald-700"
  if (status === "error") return "border-rose-200 bg-rose-50 text-rose-700"
  return "border-zinc-200 bg-zinc-100 text-zinc-600"
}

function inspectorViewLabel(view: AgentInspectorView): string {
  if (view === "tools") return "Tools"
  if (view === "files") return "Files"
  if (view === "agents") return "Workers"
  if (view === "graph") return "Graph"
  return "Live"
}

function compactMonitorMetric(value: string): string {
  return value.trim()
}

function formatWorkspaceLabel(outputDir: string | null | undefined): string {
  if (!outputDir) return "Workspace not configured"
  const segments = outputDir.split("/").filter(Boolean)
  if (segments.length === 0) return outputDir
  return segments[segments.length - 1]
}

function taskStatusLabel(status: AgentTask["status"]): string {
  if (status === "in_progress") return "Running"
  if (status === "repairing") return "Repairing"
  if (status === "human_review") return "Review"
  if (status === "done") return "Done"
  if (status === "paused") return "Paused"
  if (status === "cancelled") return "Cancelled"
  return "Planned"
}

function taskStatusClassName(status: AgentTask["status"]): string {
  if (status === "in_progress") return "border-blue-200 bg-blue-50 text-blue-700"
  if (status === "repairing") return "border-amber-200 bg-amber-50 text-amber-700"
  if (status === "human_review") return "border-violet-200 bg-violet-50 text-violet-700"
  if (status === "done") return "border-emerald-200 bg-emerald-50 text-emerald-700"
  if (status === "paused") return "border-zinc-200 bg-zinc-100 text-zinc-600"
  if (status === "cancelled") return "border-zinc-200 bg-zinc-100 text-zinc-500"
  return "border-zinc-200 bg-white text-zinc-600"
}

function EmptyWorkspace({ onBack }: { onBack: () => void }) {
  return (
    <div className="flex flex-1 min-h-0 items-center justify-center bg-slate-100">
      <div className="max-w-md px-6 text-center">
        <h2 className="text-lg font-semibold text-slate-900">DeepCode Studio</h2>
        <p className="mt-2 text-sm text-slate-500">
          Monitor now lives inside Studio. Open a paper first, then switch to the Monitor view.
        </p>
        <Button className="mt-5" onClick={onBack}>
          Back to Studio
        </Button>
      </div>
    </div>
  )
}

function TaskRail({
  tasks,
}: {
  tasks: AgentTask[]
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f8f9f6]">
      <div className="border-b border-slate-200 px-3 py-2.5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Planned Tasks
            </p>
            <p className="mt-1 text-[11px] text-slate-500">Claude Code handoffs and review steps</p>
          </div>
          <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            {tasks.length}
          </span>
        </div>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        {tasks.length === 0 ? (
          <div className="px-4 py-6 text-sm text-slate-500">No planned tasks yet.</div>
        ) : (
          <ul className="space-y-2 px-3 py-3">
            {tasks.map((task) => {
              const assignee = getAgentPresentation(task.assignee)
              return (
                <li key={task.id} className="rounded-2xl border border-slate-200 bg-white px-3 py-3 shadow-sm">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-slate-900">{task.title}</p>
                      {task.description ? (
                        <p className="mt-1 line-clamp-2 text-[12px] text-slate-500">{task.description}</p>
                      ) : null}
                    </div>
                    <span
                      className={`shrink-0 border px-2 py-0.5 text-[10px] font-medium ${taskStatusClassName(task.status)}`}
                    >
                      {taskStatusLabel(task.status)}
                    </span>
                  </div>

                  <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500">
                    <span>{assignee.label}</span>
                    <span>{task.progress}%</span>
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </ScrollArea>
    </div>
  )
}

function LeftRail({
  selectedPaperTitle,
  selectedSessionId,
  selectedPaperStatus,
  projectDir,
  contextLabel,
  onOpenInVSCode,
  tasks,
  activeView,
  onViewChange,
}: {
  selectedPaperTitle: string
  selectedSessionId: string | null
  selectedPaperStatus: StudioPaperStatus
  projectDir: string | null
  contextLabel: string
  onOpenInVSCode: () => void
  tasks: AgentTask[]
  activeView: LeftRailView
  onViewChange: (value: LeftRailView) => void
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f8f9f6]">
      <div className="border-b border-slate-200 bg-[#f4f5f1] px-3 py-2.5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-1.5">
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                Claude Code
              </p>
              <span
                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium ${paperStatusClassName(selectedPaperStatus)}`}
              >
                {paperStatusLabel(selectedPaperStatus)}
              </span>
            </div>
            <h2 className="mt-1 truncate text-sm font-semibold text-slate-900">{selectedPaperTitle}</h2>
            <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] text-slate-500">
              {selectedSessionId ? (
                <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 font-mono text-[9px] text-slate-500">
                  {selectedSessionId.slice(0, 12)}
                </span>
              ) : (
                <span>No active session</span>
              )}
              <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5">
                {formatWorkspaceLabel(projectDir)}
              </span>
              <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5">
                Context {contextLabel}
              </span>
            </div>
          </div>

          <Button
            variant="ghost"
            size="icon"
            className="mt-0.5 h-7 w-7 shrink-0 rounded-full border border-slate-200 bg-white text-slate-600 hover:text-slate-900"
            onClick={onOpenInVSCode}
            disabled={!projectDir}
            title={projectDir ? `Open ${projectDir} in VS Code` : "Set up a workspace first"}
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <Tabs
        value={activeView}
        onValueChange={(value) => onViewChange(value as LeftRailView)}
        className="flex h-full min-h-0 flex-col"
      >
        <div className="border-b border-slate-200 px-3 py-3">
          <TabsList className="grid w-full grid-cols-3 rounded-2xl border border-slate-200 bg-white p-1">
            <TabsTrigger value="threads" className="rounded-xl border border-transparent px-1 text-[11px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              Threads
            </TabsTrigger>
            <TabsTrigger value="tasks" className="rounded-xl border border-transparent px-1 text-[11px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              Tasks
            </TabsTrigger>
            <TabsTrigger value="workspace" className="rounded-xl border border-transparent px-1 text-[11px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              Files
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="threads" className="m-0 flex-1 min-h-0">
          <ChatHistoryPanel />
        </TabsContent>

        <TabsContent value="tasks" className="m-0 flex-1 min-h-0">
          <TaskRail tasks={tasks} />
        </TabsContent>

        <TabsContent value="workspace" className="m-0 flex-1 min-h-0">
          <AgentBoardSidebar
            backgroundColor="#f8fafc"
            className="h-full w-full border-r-0"
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function RightInspector({
  activeAgents,
  subagentEvents,
  runtimeInfo,
  runtimeLoading,
  activeView,
  onViewChange,
  selectedWorkerTitle,
  selectedWorkerLabel,
  selectedWorkerControlMode,
  onClearWorkerSelection,
}: {
  activeAgents: number
  subagentEvents: number
  runtimeInfo: StudioRuntimeInfo
  runtimeLoading: boolean
  activeView: AgentInspectorView
  onViewChange: (value: AgentInspectorView) => void
  selectedWorkerTitle: string | null
  selectedWorkerLabel: string | null
  selectedWorkerControlMode: "managed" | "mirrored" | null
  onClearWorkerSelection: () => void
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f8f9f6] text-slate-900">
      <div className="border-b border-slate-200 bg-[#f4f5f1] px-3 py-2">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Monitor
            </p>
            <h2 className="mt-0.5 text-sm font-semibold text-slate-900">Runtime Activity</h2>
            <p className="mt-0.5 text-[11px] text-slate-500">
              {runtimeLoading ? "Checking Claude Code..." : runtimeInfo.statusLabel}
            </p>
          </div>
          <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            {inspectorViewLabel(activeView)}
          </span>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            {runtimeLoading ? "Checking runtime" : compactMonitorMetric(runtimeInfo.label)}
          </span>
          <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            {activeAgents} agents
          </span>
          <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
            {subagentEvents} workers
          </span>
        </div>

        {selectedWorkerTitle ? (
          <div
            className={`mt-2 rounded-[18px] border px-2.5 py-1.5 ${
              selectedWorkerControlMode === "managed"
                ? "border-emerald-200 bg-[linear-gradient(180deg,#fcfdf9_0%,#f3f7ef_100%)]"
                : "border-slate-200 bg-[linear-gradient(180deg,#ffffff_0%,#f2f5f8_100%)]"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-400">
                  Focused Worker
                </div>
                <div className="mt-0.5 truncate text-[12px] font-medium text-slate-800">
                  {selectedWorkerLabel ? `${selectedWorkerLabel} · ` : ""}
                  {selectedWorkerTitle}
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-1.5">
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] ${
                      selectedWorkerControlMode === "managed"
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-slate-200 bg-white text-slate-700"
                    }`}
                  >
                    {selectedWorkerControlMode === "managed" ? "Studio-controlled" : "Claude-controlled"}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-slate-500">
                  {selectedWorkerControlMode === "managed"
                    ? "Control stays in Studio."
                    : "Control stays in the parent Claude session."}
                </div>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-7 rounded-full px-2 text-[11px] text-slate-600"
                onClick={onClearWorkerSelection}
              >
                Clear
              </Button>
            </div>
          </div>
        ) : null}

        <div className="mt-2 rounded-[16px] border border-slate-200 bg-white px-2 py-1">
          <AgentStatusPanel compact />
        </div>
      </div>

      <Tabs
        value={activeView}
        onValueChange={(value) => onViewChange(value as AgentInspectorView)}
        className="flex flex-1 min-h-0 flex-col"
      >
        <div className="border-b border-slate-200 bg-slate-50 px-2 py-1">
          <TabsList className="grid w-full grid-cols-5 rounded-[16px] border border-slate-200 bg-white p-1">
            <TabsTrigger value="live" className="rounded-[12px] border border-transparent px-1 text-[10px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              <Activity className="mr-1 h-3.5 w-3.5" />
              Live
            </TabsTrigger>
            <TabsTrigger value="tools" className="rounded-[12px] border border-transparent px-1 text-[10px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              <Wrench className="mr-1 h-3.5 w-3.5" />
              Tools
            </TabsTrigger>
            <TabsTrigger value="files" className="rounded-[12px] border border-transparent px-1 text-[10px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              <FileCode2 className="mr-1 h-3.5 w-3.5" />
              Files
            </TabsTrigger>
            <TabsTrigger value="agents" className="rounded-[12px] border border-transparent px-1 text-[10px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              <Bot className="mr-1 h-3.5 w-3.5" />
              Workers
            </TabsTrigger>
            <TabsTrigger value="graph" className="rounded-[12px] border border-transparent px-1 text-[10px] text-slate-500 shadow-none data-[state=active]:border-slate-200 data-[state=active]:bg-[#eef1ea] data-[state=active]:text-slate-900">
              <GitBranch className="mr-1 h-3.5 w-3.5" />
              Graph
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="live" className="m-0 flex-1 min-h-0 bg-slate-50">
          <ActivityFeed />
        </TabsContent>
        <TabsContent value="tools" className="m-0 flex-1 min-h-0 bg-slate-50">
          <ToolCallTimeline />
        </TabsContent>
        <TabsContent value="files" className="m-0 flex-1 min-h-0 bg-slate-50">
          <FileListPanel />
        </TabsContent>
        <TabsContent value="agents" className="m-0 flex-1 min-h-0 bg-slate-50">
          <SubagentActivityPanel />
        </TabsContent>
        <TabsContent value="graph" className="m-0 flex-1 min-h-0 bg-slate-50 text-slate-900">
          <AgentDagPanel />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function CenterSurface({
  activeView,
  onViewChange,
  runtimeInfo,
  runtimeLoading,
}: {
  activeView: CenterView
  onViewChange: (value: CenterView) => void
  runtimeInfo: StudioRuntimeInfo
  runtimeLoading: boolean
}) {
  const visibleView = activeView === "commands" ? "log" : activeView
  const contentView: "log" | "context" | "commands" = activeView === "board" ? "log" : activeView

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#f1f2ed]">
      <div className="border-b border-slate-200 bg-[#f4f5f1] px-4 py-2.5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex items-center gap-1 rounded-2xl border border-slate-200 bg-white p-1 shadow-sm">
            <button
              type="button"
              onClick={() => onViewChange("log")}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors ${
                visibleView === "log"
                  ? "border-slate-200 bg-[#eef1ea] text-slate-900 shadow-sm"
                  : "border-transparent bg-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              <MessageSquare className="h-3.5 w-3.5" />
              Chat
            </button>
            <button
              type="button"
              onClick={() => onViewChange("board")}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors ${
                visibleView === "board"
                  ? "border-slate-200 bg-[#eef1ea] text-slate-900 shadow-sm"
                  : "border-transparent bg-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              <GitBranch className="h-3.5 w-3.5" />
              Monitor
            </button>
            <button
              type="button"
              onClick={() => onViewChange("context")}
              className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors ${
                visibleView === "context"
                  ? "border-slate-200 bg-[#eef1ea] text-slate-900 shadow-sm"
                  : "border-transparent bg-transparent text-slate-500 hover:text-slate-700"
              }`}
            >
              <Sparkles className="h-3.5 w-3.5" />
              Context
            </button>
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 bg-[#eceee8] p-3">
        <div className="h-full min-h-0 overflow-hidden rounded-[28px] border border-slate-200 bg-[#f7f7f4] shadow-[0_24px_80px_rgba(15,23,42,0.05)]">
          <ReproductionLog
            viewMode={contentView}
            onViewModeChange={onViewChange}
            hideNavigation
            onOpenBoardWorkspace={() => onViewChange("board")}
            runtimeInfo={runtimeInfo}
            runtimeLoading={runtimeLoading}
          />
        </div>
      </div>
    </div>
  )
}

function ResizeDivider({
  isActive,
  onMouseDown,
}: {
  isActive: boolean
  onMouseDown: (event: ReactMouseEvent<HTMLDivElement>) => void
}) {
  return (
    <div
      aria-hidden="true"
      className={`relative flex w-3 shrink-0 cursor-col-resize items-stretch justify-center bg-slate-100 ${
        isActive ? "bg-slate-200" : "hover:bg-slate-200"
      }`}
      onMouseDown={onMouseDown}
    >
      <span className={`my-2 w-px rounded-full ${isActive ? "bg-slate-400" : "bg-slate-300"}`} />
    </div>
  )
}

export function AgentWorkspace({
  defaultCenterView = "log",
  onBackToStudio,
}: AgentWorkspaceProps) {
  useAgentEvents()
  const { info: runtimeInfo, loading: runtimeLoading } = useStudioRuntime()

  const router = useRouter()
  const searchParams = useSearchParams()
  const [leftRailView, setLeftRailView] = useState<LeftRailView>("threads")
  const desktopLayoutRef = useRef<HTMLDivElement | null>(null)
  const [desktopWidth, setDesktopWidth] = useState(0)
  const [dragState, setDragState] = useState<DragState | null>(null)
  const [panelWidths, setPanelWidths] = useState<PanelWidths>(() => ({
    leftWidth: clamp(
      readStoredWidth(LEFT_RAIL_STORAGE_KEY, DEFAULT_LEFT_RAIL_WIDTH),
      LEFT_RAIL_MIN_WIDTH,
      LEFT_RAIL_MAX_WIDTH,
    ),
    rightWidth: clamp(
      readStoredWidth(RIGHT_INSPECTOR_STORAGE_KEY, DEFAULT_RIGHT_INSPECTOR_WIDTH),
      RIGHT_INSPECTOR_MIN_WIDTH,
      RIGHT_INSPECTOR_MAX_WIDTH,
    ),
  }))

  const loadPapers = useStudioStore((state) => state.loadPapers)
  const selectPaper = useStudioStore((state) => state.selectPaper)
  const papers = useStudioStore((state) => state.papers)
  const selectedPaperId = useStudioStore((state) => state.selectedPaperId)
  const boardSessionId = useStudioStore((state) => state.boardSessionId)
  const studioTasks = useStudioStore((state) => state.agentTasks)
  const contextPack = useStudioStore((state) => state.contextPack)
  const contextPackLoading = useStudioStore((state) => state.contextPackLoading)

  const agentStatuses = useAgentEventStore((state) => state.agentStatuses)
  const codexDelegations = useAgentEventStore((state) => state.codexDelegations)
  const toolCalls = useAgentEventStore((state) => state.toolCalls)
  const inspectorView = useAgentEventStore((state) => state.inspectorView)
  const setInspectorView = useAgentEventStore((state) => state.setInspectorView)
  const selectedWorkerRunId = useAgentEventStore((state) => state.selectedWorkerRunId)
  const setSelectedWorkerRunId = useAgentEventStore((state) => state.setSelectedWorkerRunId)
  const requestedWorkspaceSurface = useAgentEventStore((state) => state.requestedWorkspaceSurface)
  const clearWorkspaceSurfaceRequest = useAgentEventStore((state) => state.clearWorkspaceSurfaceRequest)

  const requestedPaperId = searchParams.get("paperId") || searchParams.get("paper_id")
  const requestedSurface = normalizeCenterView(searchParams.get("surface"), defaultCenterView)
  const [localCenterView, setLocalCenterView] = useState<CenterView>(requestedSurface)
  const [mobilePinnedView, setMobilePinnedView] = useState<"threads" | null>(null)
  const centerView: CenterView = requestedWorkspaceSurface ?? localCenterView

  useEffect(() => {
    loadPapers()
  }, [loadPapers])

  useEffect(() => {
    if (!requestedPaperId) return
    if (selectedPaperId === requestedPaperId) return
    if (!papers.some((paper) => paper.id === requestedPaperId)) return
    selectPaper(requestedPaperId)
  }, [papers, requestedPaperId, selectPaper, selectedPaperId])

  function updateCenterView(value: CenterView) {
    clearWorkspaceSurfaceRequest()
    setLocalCenterView(value)
  }

  const selectedPaper = useMemo(
    () => (selectedPaperId ? papers.find((paper) => paper.id === selectedPaperId) || null : null),
    [papers, selectedPaperId],
  )

  const boardTasks = useMemo(() => {
    if (!selectedPaperId) {
      return studioTasks
    }
    return studioTasks.filter((task) => task.paperId === selectedPaperId)
  }, [selectedPaperId, studioTasks])

  const activeAgents = Object.values(agentStatuses).filter((entry) => entry.status === "working").length
  const selectedSessionId = selectedPaper?.boardSessionId || boardSessionId || null
  const selectedPaperTitle = selectedPaper?.title || "Untitled paper"
  const selectedPaperStatus = selectedPaper?.status || "draft"
  const projectDir = selectedPaper?.outputDir || selectedPaper?.lastGenCodeResult?.outputDir || null
  const showMonitorInspector = centerView === "board"
  const mobileView: MobileView =
    mobilePinnedView === "threads" ? "threads" : centerViewToMobileView(centerView)
  const workerGroups = useMemo(
    () => buildSubagentActivityGroups(codexDelegations, toolCalls),
    [codexDelegations, toolCalls],
  )
  const selectedWorkerGroup = useMemo(
    () =>
      selectedWorkerRunId
        ? workerGroups.find((group) => group.workerRunId === selectedWorkerRunId) ?? null
        : null,
    [selectedWorkerRunId, workerGroups],
  )
  const selectedWorkerPresentation = selectedWorkerGroup
    ? getAgentPresentation(selectedWorkerGroup.assignee)
    : null
  const contextLabel = contextPackLoading
    ? "Generating"
    : contextPack?.context_pack_id || selectedPaper?.contextPackId
      ? "Ready"
      : "Missing"

  useEffect(() => {
    const layoutElement = desktopLayoutRef.current
    if (!layoutElement) return

    const updateWidth = (width: number) => {
      const nextWidth = Math.round(width)
      setDesktopWidth(nextWidth)
      setPanelWidths((current) => {
        const next = clampPanelWidths(nextWidth, current.leftWidth, current.rightWidth)
        if (
          next.leftWidth === current.leftWidth &&
          next.rightWidth === current.rightWidth
        ) {
          return current
        }
        return next
      })
    }

    updateWidth(layoutElement.getBoundingClientRect().width)

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      updateWidth(entry.contentRect.width)
    })

    resizeObserver.observe(layoutElement)
    return () => resizeObserver.disconnect()
  }, [])

  useEffect(() => {
    if (typeof window === "undefined") return
    window.localStorage.setItem(LEFT_RAIL_STORAGE_KEY, String(panelWidths.leftWidth))
    window.localStorage.setItem(RIGHT_INSPECTOR_STORAGE_KEY, String(panelWidths.rightWidth))
  }, [panelWidths.leftWidth, panelWidths.rightWidth])

  useEffect(() => {
    if (!dragState) return

    const handleMouseMove = (event: MouseEvent) => {
      setPanelWidths(() => {
        const delta = event.clientX - dragState.startX
        return dragState.side === "left"
          ? clampPanelWidths(
              desktopWidth,
              dragState.startLeftWidth + delta,
              dragState.startRightWidth,
            )
          : clampPanelWidths(
              desktopWidth,
              dragState.startLeftWidth,
              dragState.startRightWidth - delta,
            )
      })
    }

    const handleMouseUp = () => {
      setDragState(null)
    }

    const { style } = document.body
    const previousCursor = style.cursor
    const previousUserSelect = style.userSelect
    style.cursor = "col-resize"
    style.userSelect = "none"

    window.addEventListener("mousemove", handleMouseMove)
    window.addEventListener("mouseup", handleMouseUp)

    return () => {
      style.cursor = previousCursor
      style.userSelect = previousUserSelect
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("mouseup", handleMouseUp)
    }
  }, [desktopWidth, dragState])

  function beginResize(side: DragState["side"]) {
    return (event: ReactMouseEvent<HTMLDivElement>) => {
      event.preventDefault()
      setDragState({
        side,
        startX: event.clientX,
        startLeftWidth: panelWidths.leftWidth,
        startRightWidth: panelWidths.rightWidth,
      })
    }
  }

  if (!selectedPaperId) {
    return <EmptyWorkspace onBack={() => router.push("/studio")} />
  }

  return (
    <div className="flex h-screen min-h-0 flex-col bg-[#f1f2ed]">
      <div className="border-b border-slate-200 bg-[#f6f7f3]">
        <div className="flex min-h-12 items-center gap-3 px-4 py-2">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 gap-1.5 rounded-full border border-transparent px-3 text-slate-600 hover:bg-white hover:text-slate-900"
            onClick={() => {
              if (onBackToStudio) {
                onBackToStudio()
                return
              }
              router.push("/studio")
            }}
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Papers
          </Button>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <div className="truncate text-sm font-semibold text-slate-900">{selectedPaperTitle}</div>
              <span
                className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-medium ${paperStatusClassName(selectedPaperStatus)}`}
              >
                {paperStatusLabel(selectedPaperStatus)}
              </span>
            </div>
            <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[10px] text-slate-500">
              {selectedSessionId ? (
                <Badge variant="outline" className="h-5 border-slate-200 font-mono text-[10px] text-slate-600">
                  {selectedSessionId.slice(0, 12)}
                </Badge>
              ) : (
                <span>No session</span>
              )}
              {projectDir ? (
                <Badge variant="outline" className="h-5 border-slate-200 text-[10px] text-slate-600">
                  {formatWorkspaceLabel(projectDir)}
                </Badge>
              ) : null}
              {showMonitorInspector ? (
                <span>{runtimeLoading ? "Checking runtime" : runtimeInfo.label}</span>
              ) : (
                <span>Chat view</span>
              )}
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2 overflow-hidden">
            {showMonitorInspector ? (
              <Badge variant="outline" className="border-slate-200 text-[10px] text-slate-600">
                {activeAgents} active
              </Badge>
            ) : null}
          </div>
        </div>

      </div>

      <div ref={desktopLayoutRef} className="hidden flex-1 min-h-0 overflow-hidden bg-slate-100 lg:flex">
        <div className="flex min-w-0 flex-1">
          <div
            className="shrink-0 border-r border-slate-200 bg-slate-50"
            style={{ width: panelWidths.leftWidth }}
          >
            <LeftRail
              selectedPaperTitle={selectedPaperTitle}
              selectedSessionId={selectedSessionId}
              selectedPaperStatus={selectedPaperStatus}
              projectDir={projectDir}
              contextLabel={contextLabel}
              onOpenInVSCode={() => {
                if (projectDir) {
                  window.open(`vscode://file${projectDir}`, "_blank")
                }
              }}
              tasks={boardTasks}
              activeView={leftRailView}
              onViewChange={setLeftRailView}
            />
          </div>

          <ResizeDivider isActive={dragState?.side === "left"} onMouseDown={beginResize("left")} />

          <div className={`min-w-[380px] flex-1 bg-slate-100 ${showMonitorInspector ? "border-r border-slate-200" : ""}`}>
            <CenterSurface
              activeView={centerView}
              onViewChange={updateCenterView}
              runtimeInfo={runtimeInfo}
              runtimeLoading={runtimeLoading}
            />
          </div>

          {showMonitorInspector ? (
            <>
              <ResizeDivider isActive={dragState?.side === "right"} onMouseDown={beginResize("right")} />

              <div
                className="shrink-0 bg-slate-50"
                style={{ width: panelWidths.rightWidth }}
              >
                <RightInspector
                  activeAgents={activeAgents}
                  subagentEvents={codexDelegations.length}
                  runtimeInfo={runtimeInfo}
                  runtimeLoading={runtimeLoading}
                  activeView={inspectorView}
                  onViewChange={setInspectorView}
                  selectedWorkerTitle={selectedWorkerGroup?.taskTitle ?? null}
                  selectedWorkerLabel={selectedWorkerPresentation?.label ?? null}
                  selectedWorkerControlMode={selectedWorkerGroup?.controlMode ?? null}
                  onClearWorkerSelection={() => setSelectedWorkerRunId(null)}
                />
              </div>
            </>
          ) : null}
        </div>
      </div>

      <div className="flex flex-1 min-h-0 lg:hidden">
        <Tabs
          value={mobileView}
          onValueChange={(value) => {
            const nextView = value as MobileView
            if (nextView === "threads") {
              setMobilePinnedView("threads")
              return
            }

            setMobilePinnedView(null)
            if (nextView === "monitor") {
              updateCenterView("board")
              return
            }
            if (nextView === "console" && centerView === "board") {
              updateCenterView("log")
            }
          }}
          className="flex h-full w-full flex-col"
        >
          <TabsList className="grid w-full grid-cols-3 rounded-none border-b bg-white p-0">
            <TabsTrigger value="threads" className="rounded-none">Threads</TabsTrigger>
            <TabsTrigger value="console" className="rounded-none">Chat</TabsTrigger>
            <TabsTrigger value="monitor" className="rounded-none">Monitor</TabsTrigger>
          </TabsList>

          <TabsContent value="threads" className="m-0 flex-1 min-h-0">
            <LeftRail
              selectedPaperTitle={selectedPaperTitle}
              selectedSessionId={selectedSessionId}
              selectedPaperStatus={selectedPaperStatus}
              projectDir={projectDir}
              contextLabel={contextLabel}
              onOpenInVSCode={() => {
                if (projectDir) {
                  window.open(`vscode://file${projectDir}`, "_blank")
                }
              }}
              tasks={boardTasks}
              activeView="threads"
              onViewChange={() => {}}
            />
          </TabsContent>

          <TabsContent value="console" className="m-0 flex-1 min-h-0">
            <ReproductionLog
              viewMode={centerView === "board" ? "log" : centerView}
              onViewModeChange={(value) => {
                updateCenterView(value)
                setMobilePinnedView(null)
              }}
              hideNavigation={false}
              onOpenBoardWorkspace={() => {
                updateCenterView("board")
                setMobilePinnedView(null)
              }}
              runtimeInfo={runtimeInfo}
              runtimeLoading={runtimeLoading}
            />
          </TabsContent>

          <TabsContent value="monitor" className="m-0 flex-1 min-h-0">
            <RightInspector
              activeAgents={activeAgents}
              subagentEvents={codexDelegations.length}
              runtimeInfo={runtimeInfo}
              runtimeLoading={runtimeLoading}
              activeView={inspectorView}
              onViewChange={setInspectorView}
              selectedWorkerTitle={selectedWorkerGroup?.taskTitle ?? null}
              selectedWorkerLabel={selectedWorkerPresentation?.label ?? null}
              selectedWorkerControlMode={selectedWorkerGroup?.controlMode ?? null}
              onClearWorkerSelection={() => setSelectedWorkerRunId(null)}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
