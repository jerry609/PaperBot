"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Folder, FolderOpen, Loader2, Circle, CheckCircle2, ShieldAlert, ChevronRight, ChevronLeft, MessageSquare, LayoutDashboard } from "lucide-react"
import { StudioPaper } from "@/lib/store/studio-store"
import { cn } from "@/lib/utils"

interface WorkspaceSetupDialogProps {
    paper: StudioPaper
    open: boolean
    onConfirm: (directory: string) => void
    onCancel: () => void
}

interface StudioCwdPayload {
    cwd?: string | null
    home?: string | null
    allowed_prefixes?: string[] | null
    allowlist_mutation_enabled?: boolean | null
}

async function readErrorDetail(res: Response, fallback: string): Promise<string> {
    const text = await res.text()
    try {
        const payload = JSON.parse(text) as { detail?: string }
        return payload.detail || fallback
    } catch {
        return text || fallback
    }
}

function trimTrailingSlashes(directory: string): string {
    const trimmed = directory.trim()
    if (!trimmed) return ""
    let end = trimmed.length
    while (end > 1 && trimmed[end - 1] === "/") {
        end -= 1
    }
    const normalized = trimmed.slice(0, end)
    return normalized || "/"
}

function deriveAllowDirCandidate(directory: string, choice: "current" | "custom"): string {
    const normalized = trimTrailingSlashes(directory)
    if (!normalized || normalized === "/") return normalized
    if (choice === "current") return normalized

    const lastSlash = normalized.lastIndexOf("/")
    if (lastSlash <= 0) return normalized
    return normalized.slice(0, lastSlash)
}

function isUnderAllowedStudioRoot(directory: string, allowedPrefixes: string[]): boolean {
    const normalized = trimTrailingSlashes(directory)
    if (!normalized) return false

    return allowedPrefixes.some((prefix) => {
        const normalizedPrefix = trimTrailingSlashes(prefix)
        return normalized === normalizedPrefix || normalized.startsWith(`${normalizedPrefix}/`)
    })
}

function buildDisallowedDirectoryMessage(directory: string, allowedPrefixes: string[]): string {
    const normalized = trimTrailingSlashes(directory)
    const allowedSummary =
        allowedPrefixes.length > 0
            ? `Choose a directory under one of the allowed Studio roots: ${allowedPrefixes.join(", ")}.`
            : "Choose a directory under an allowed Studio workspace root or /tmp."
    return `${normalized} is outside the allowed Studio workspace roots. ${allowedSummary}`
}

export function WorkspaceSetupDialog({
    paper,
    open,
    onConfirm,
    onCancel,
}: WorkspaceSetupDialogProps) {
    const [step, setStep] = useState<"workspace" | "review">("workspace")
    const [loading, setLoading] = useState(true)
    const [currentDir, setCurrentDir] = useState("")
    const [choice, setChoice] = useState<"current" | "custom">("current")
    const [customDir, setCustomDir] = useState("")
    const [error, setError] = useState<string | null>(null)
    const [validating, setValidating] = useState(false)
    const [pendingAllowDir, setPendingAllowDir] = useState<string | null>(null)
    const [allowedPrefixes, setAllowedPrefixes] = useState<string[]>([])
    const [allowlistMutationEnabled, setAllowlistMutationEnabled] = useState(false)

    // Fetch current working directory on mount
    useEffect(() => {
        if (open) {
            setStep("workspace")
            setLoading(true)
            setError(null)
            setPendingAllowDir(null)
            setAllowedPrefixes([])
            setAllowlistMutationEnabled(false)
            fetch("/api/studio/cwd")
                .then((res) => res.json())
                .then((data: StudioCwdPayload) => {
                    const dir = data.cwd || data.home || "/tmp"
                    setCurrentDir(dir)
                    setAllowedPrefixes(
                        Array.isArray(data.allowed_prefixes)
                            ? data.allowed_prefixes.filter((item): item is string => typeof item === "string" && item.trim().length > 0)
                            : []
                    )
                    setAllowlistMutationEnabled(data.allowlist_mutation_enabled === true)
                    // Suggest a paper-specific subdirectory
                    const safeName = paper.title
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, "-")
                        .slice(0, 40)
                    setCustomDir(`${dir}/${safeName}`)
                })
                .catch(() => {
                    setCurrentDir("/tmp")
                    setError("Could not detect current directory")
                })
                .finally(() => {
                    setLoading(false)
                })
        }
    }, [open, paper.title])

    const prepareDirectory = async (directory: string): Promise<boolean> => {
        const res = await fetch("/api/runbook/project-dir/prepare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                project_dir: directory,
                create_if_missing: true,
            }),
        })
        if (!res.ok) {
            if (res.status === 403) {
                const detail = await readErrorDetail(res, "Directory is not available")
                if (isUnderAllowedStudioRoot(directory, allowedPrefixes)) {
                    setPendingAllowDir(null)
                    setError(detail)
                    return false
                }
                if (!allowlistMutationEnabled) {
                    setPendingAllowDir(null)
                    setError(buildDisallowedDirectoryMessage(directory, allowedPrefixes))
                    return false
                }
                const allowDir = deriveAllowDirCandidate(directory, choice)
                setPendingAllowDir(allowDir || null)
                return false
            }
            setError(await readErrorDetail(res, `Directory is not available (${res.status})`))
            return false
        }
        const data = await res.json() as { project_dir?: string }
        onConfirm(data.project_dir || directory)
        return true
    }

    const handleConfirm = async () => {
        const directory = choice === "current" ? currentDir : customDir.trim()
        if (!directory) {
            setError("Please enter a valid directory path")
            return
        }
        setValidating(true)
        setError(null)
        setPendingAllowDir(null)
        try {
            await prepareDirectory(directory)
        } catch {
            setError("Failed to validate directory path")
        } finally {
            setValidating(false)
        }
    }

    const handleAllowDir = async () => {
        if (!pendingAllowDir) return
        setValidating(true)
        setError(null)
        try {
            const res = await fetch("/api/runbook/allowed-dirs", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ directory: pendingAllowDir }),
            })
            if (!res.ok) {
                const detail = await readErrorDetail(res, "Failed to add directory to allowed list")
                if (res.status === 403 && detail === "runtime allowlist mutation is disabled") {
                    setPendingAllowDir(null)
                    const directory = choice === "current" ? currentDir : customDir.trim()
                    setError(buildDisallowedDirectoryMessage(directory, allowedPrefixes))
                } else {
                    setError(detail)
                }
                return
            }
            setPendingAllowDir(null)
            // Retry the original prepare call
            const directory = choice === "current" ? currentDir : customDir.trim()
            await prepareDirectory(directory)
        } catch {
            setError("Failed to add directory to allowed list")
        } finally {
            setValidating(false)
        }
    }

    const selectedDir = choice === "current" ? currentDir : customDir
    const normalizedSelectedDir = trimTrailingSlashes(selectedDir)
    const directoryAllowed =
        !normalizedSelectedDir || allowedPrefixes.length === 0
            ? false
            : isUnderAllowedStudioRoot(normalizedSelectedDir, allowedPrefixes)
    const directoryNeedsAccess =
        Boolean(normalizedSelectedDir) &&
        !directoryAllowed &&
        choice === "custom" &&
        allowedPrefixes.length > 0
    const canContinueToReview = Boolean(selectedDir && selectedDir.trim())

    return (
        <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
            <DialogContent className="sm:max-w-[640px]">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <FolderOpen className="h-5 w-5 text-blue-500" />
                        Set Up Workspace
                    </DialogTitle>
                    <DialogDescription>
                        Choose where to save generated code for &quot;{paper.title.slice(0, 50)}
                        {paper.title.length > 50 ? "..." : ""}&quot;
                    </DialogDescription>
                </DialogHeader>

                <div className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-[#f7f8f4] p-1">
                    {[
                        { key: "workspace" as const, label: "Workspace" },
                        { key: "review" as const, label: "Review" },
                    ].map((item, index) => {
                        const active = step === item.key
                        const complete = step === "review" && item.key === "workspace"
                        return (
                            <div
                                key={item.key}
                                className={cn(
                                    "flex min-w-0 flex-1 items-center justify-center gap-2 rounded-xl px-3 py-2 text-[11px] font-medium",
                                    active
                                        ? "bg-white text-slate-900 shadow-[0_1px_0_rgba(255,255,255,0.9)_inset]"
                                        : "text-slate-500",
                                )}
                            >
                                <span
                                    className={cn(
                                        "flex h-5 w-5 items-center justify-center rounded-full border text-[10px]",
                                        active || complete
                                            ? "border-slate-300 bg-[#eef1ea] text-slate-800"
                                            : "border-slate-200 bg-white text-slate-400",
                                    )}
                                >
                                    {index + 1}
                                </span>
                                <span className="truncate">{item.label}</span>
                            </div>
                        )
                    })}
                </div>

                {loading ? (
                    <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                    </div>
                ) : (
                    <div className="space-y-4 py-4">
                        {pendingAllowDir && (
                            <div className="text-sm bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 px-3 py-3 rounded-md space-y-2">
                                <div className="flex items-start gap-2">
                                    <ShieldAlert className="h-4 w-4 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
                                    <div className="text-blue-700 dark:text-blue-300">
                                        This directory is outside the currently allowed Studio workspace roots.
                                        Allow access to <code className="bg-blue-100 dark:bg-blue-900 px-1 rounded text-xs">{pendingAllowDir}</code>?
                                    </div>
                                </div>
                                <div className="flex gap-2 ml-6">
                                    <Button size="sm" onClick={handleAllowDir} disabled={validating}>
                                        {validating ? "Allowing..." : "Allow Access"}
                                    </Button>
                                    <Button size="sm" variant="ghost" onClick={() => setPendingAllowDir(null)}>
                                        Cancel
                                    </Button>
                                </div>
                            </div>
                        )}

                        {error && (
                            <div className="text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 px-3 py-2 rounded-md">
                                {error}
                            </div>
                        )}

                        {step === "workspace" ? (
                            <div className="space-y-4">
                                <div className="rounded-[22px] border border-slate-200 bg-[#f8faf5] px-4 py-3">
                                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                                        Workspace Choice
                                    </p>
                                    <p className="mt-1 text-[12px] text-slate-600">
                                        Pick the directory that Claude Code should use when this paper thread starts.
                                    </p>
                                </div>

                                <div className="space-y-3">
                                    <button
                                        type="button"
                                        onClick={() => setChoice("current")}
                                        className={cn(
                                            "w-full flex items-start gap-3 p-3 rounded-xl border text-left transition-colors",
                                            choice === "current"
                                                ? "border-primary bg-primary/5"
                                                : "border-border hover:bg-muted/50"
                                        )}
                                    >
                                        {choice === "current" ? (
                                            <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                                        ) : (
                                            <Circle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                                        )}
                                        <div className="flex-1 min-w-0">
                                            <Label className="font-medium cursor-pointer">
                                                Use current directory
                                            </Label>
                                            <div className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
                                                <Folder className="h-4 w-4 shrink-0" />
                                                <code className="bg-muted px-2 py-0.5 rounded text-xs break-all">
                                                    {currentDir}
                                                </code>
                                            </div>
                                        </div>
                                    </button>

                                    <button
                                        type="button"
                                        onClick={() => setChoice("custom")}
                                        className={cn(
                                            "w-full flex items-start gap-3 p-3 rounded-xl border text-left transition-colors",
                                            choice === "custom"
                                                ? "border-primary bg-primary/5"
                                                : "border-border hover:bg-muted/50"
                                        )}
                                    >
                                        {choice === "custom" ? (
                                            <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                                        ) : (
                                            <Circle className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                                        )}
                                        <div className="flex-1 min-w-0 space-y-2">
                                            <Label className="font-medium cursor-pointer">
                                                Choose a different directory
                                            </Label>
                                            <Input
                                                value={customDir}
                                                onChange={(e) => {
                                                    setCustomDir(e.target.value)
                                                    setChoice("custom")
                                                }}
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    setChoice("custom")
                                                }}
                                                placeholder="/path/to/project"
                                                className="font-mono text-sm"
                                            />
                                            <p className="text-xs text-muted-foreground">
                                                Tip: Use terminal <code className="bg-muted px-1 rounded">pwd</code> to get current path
                                            </p>
                                        </div>
                                    </button>
                                </div>

                                <div className="pt-2 border-t">
                                    <p className="text-xs text-muted-foreground">
                                        Generated code will be saved to:{" "}
                                        <code className="bg-muted px-1 rounded">{selectedDir || "..."}</code>
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="rounded-[22px] border border-slate-200 bg-[#f8faf5] px-4 py-3">
                                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                                        Review Session Launch
                                    </p>
                                    <p className="mt-1 text-[12px] text-slate-600">
                                        Confirm the paper, workspace, and launch surface before starting the Claude Code session.
                                    </p>
                                </div>

                                <div className="grid gap-3 sm:grid-cols-2">
                                    <div className="rounded-[20px] border border-slate-200 bg-white px-4 py-3">
                                        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">Paper</p>
                                        <p className="mt-1 text-[13px] font-medium text-slate-900">
                                            {paper.title}
                                        </p>
                                    </div>
                                    <div className="rounded-[20px] border border-slate-200 bg-white px-4 py-3">
                                        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">Workspace</p>
                                        <p className="mt-1 break-all font-mono text-[11px] text-slate-700">
                                            {selectedDir || "..."}
                                        </p>
                                    </div>
                                    <div className="rounded-[20px] border border-slate-200 bg-white px-4 py-3">
                                        <div className="flex items-center gap-2">
                                            <MessageSquare className="h-4 w-4 text-slate-500" />
                                            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">Session</p>
                                        </div>
                                        <p className="mt-1 text-[12px] font-medium text-slate-900">
                                            Claude Code chat
                                        </p>
                                        <p className="mt-1 text-[11px] leading-5 text-slate-500">
                                            Main replies stay in chat and tool/thinking activity streams underneath.
                                        </p>
                                    </div>
                                    <div className="rounded-[20px] border border-slate-200 bg-white px-4 py-3">
                                        <div className="flex items-center gap-2">
                                            <LayoutDashboard className="h-4 w-4 text-slate-500" />
                                            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">Monitor</p>
                                        </div>
                                        <p className="mt-1 text-[12px] font-medium text-slate-900">
                                            Mirrored activity
                                        </p>
                                        <p className="mt-1 text-[11px] leading-5 text-slate-500">
                                            Full worker, tool, and delegation detail remains available in Monitor.
                                        </p>
                                    </div>
                                </div>

                                <div className="rounded-[20px] border border-slate-200 bg-white px-4 py-3">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className="rounded-full border border-slate-200 bg-[#f7f8f4] px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-500">
                                            {choice === "current" ? "current dir" : "custom dir"}
                                        </span>
                                        <span
                                            className={cn(
                                                "rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em]",
                                                directoryAllowed
                                                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                                    : directoryNeedsAccess
                                                        ? "border-amber-200 bg-amber-50 text-amber-700"
                                                        : "border-slate-200 bg-[#f7f8f4] text-slate-500",
                                            )}
                                        >
                                            {directoryAllowed ? "allowed root" : directoryNeedsAccess ? "needs access" : "validate on start"}
                                        </span>
                                    </div>
                                    <p className="mt-2 text-[11px] leading-5 text-slate-600">
                                        {directoryAllowed
                                            ? "This workspace is already inside an allowed Studio root."
                                            : directoryNeedsAccess
                                                ? "This workspace sits outside the current Studio roots. Start Session will request access before continuing."
                                                : "Studio will validate this workspace when the session starts."}
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
                )}

                <DialogFooter>
                    {step === "workspace" ? (
                        <>
                            <Button variant="outline" onClick={onCancel}>
                                Cancel
                            </Button>
                            <Button onClick={() => setStep("review")} disabled={loading || !canContinueToReview}>
                                Continue
                                <ChevronRight className="ml-1 h-4 w-4" />
                            </Button>
                        </>
                    ) : (
                        <>
                            <Button variant="outline" onClick={() => setStep("workspace")} disabled={validating}>
                                <ChevronLeft className="mr-1 h-4 w-4" />
                                Back
                            </Button>
                            <Button onClick={handleConfirm} disabled={loading || !selectedDir || validating}>
                                {validating ? "Preparing..." : "Start Session"}
                            </Button>
                        </>
                    )}
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}
