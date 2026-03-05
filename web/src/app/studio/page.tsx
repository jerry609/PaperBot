"use client"

import { useEffect, Suspense, useRef } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { ArrowLeft, PanelsTopLeft, Loader2 } from "lucide-react"
import { PaperGallery } from "@/components/studio/PaperGallery"
import { ReproductionLog } from "@/components/studio/ReproductionLog"
import { FilesPanel } from "@/components/studio/FilesPanel"
import { MCPProvider } from "@/lib/mcp"
import { useStudioStore, type StudioPaperStatus } from "@/lib/store/studio-store"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import type { ReproContextPack } from "@/lib/types/p2c"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable"
import { cn } from "@/lib/utils"

const statusConfig: Record<StudioPaperStatus, { label: string; className: string }> = {
    draft: { label: "Draft", className: "bg-zinc-500/90 text-white" },
    generating: { label: "Code", className: "bg-blue-500 text-white" },
    ready: { label: "Ready", className: "bg-emerald-500 text-white" },
    running: { label: "Run", className: "bg-violet-500 text-white" },
    completed: { label: "Done", className: "bg-emerald-500 text-white" },
    error: { label: "Error", className: "bg-red-500 text-white" },
}

function isReproContextPack(payload: unknown): payload is ReproContextPack {
    if (!payload || typeof payload !== "object") return false
    const value = payload as Partial<ReproContextPack> & {
        paper?: { paper_id?: unknown }
        confidence?: { overall?: unknown }
    }
    return (
        typeof value.context_pack_id === "string" &&
        typeof value.version === "string" &&
        typeof value.created_at === "string" &&
        typeof value.objective === "string" &&
        typeof value.paper_type === "string" &&
        Array.isArray(value.observations) &&
        Array.isArray(value.task_roadmap) &&
        Array.isArray(value.warnings) &&
        typeof value.paper?.paper_id === "string" &&
        typeof value.confidence?.overall === "number"
    )
}

function StudioContent() {
    const {
        addPaper,
        selectPaper,
        loadPapers,
        papers,
        selectedPaperId,
        setContextPack,
        setContextPackLoading,
        setContextPackError,
        clearGenerationProgress,
    } = useStudioStore()
    const { generate } = useContextPackGeneration()
    const searchParams = useSearchParams()
    const router = useRouter()
    const hasProcessedParams = useRef(false)

    // Load papers from localStorage on mount
    useEffect(() => {
        loadPapers()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const normalizePack = (payload: unknown): ReproContextPack | null => {
        if (!payload || typeof payload !== "object") return null
        const raw = payload as Record<string, unknown>
        if (raw.pack && typeof raw.pack === "object") {
            const pack = raw.pack as Record<string, unknown>
            const maybePack: Record<string, unknown> =
                !pack.context_pack_id && typeof raw.context_pack_id === "string"
                    ? { ...pack, context_pack_id: raw.context_pack_id }
                    : pack
            return isReproContextPack(maybePack) ? maybePack : null
        }
        return isReproContextPack(raw) ? raw : null
    }

    // Handle URL params from the Papers detail entry point and context pack deep links
    useEffect(() => {
        if (hasProcessedParams.current) return

        const paperId = searchParams.get("paper_id")
        const title = searchParams.get("title")
        const abstract = searchParams.get("abstract")
        const generateFlag = searchParams.get("generate") === "true"
        const contextPackId = searchParams.get("context_pack_id")
        console.info("[P2C:M3] studio:params", { paperId, generateFlag, contextPackId })

        let shouldCleanUrl = false

        if (paperId) {
            const existingPaper = papers.find(p => p.id === paperId)
            if (existingPaper) {
                selectPaper(paperId)
            } else if (title) {
                addPaper({
                    title: title || "Untitled Paper",
                    abstract: abstract || "",
                })
            }
            shouldCleanUrl = true
        } else if (title && abstract) {
            addPaper({ title, abstract })
            shouldCleanUrl = true
        }

        if (contextPackId) {
            shouldCleanUrl = true
            setContextPackLoading(true)
            setContextPackError(null)
            clearGenerationProgress()
            fetch(`/api/research/repro/context/${contextPackId}`)
                .then((res) => {
                    if (!res.ok) {
                        throw new Error(`Failed to load context pack (${res.status})`)
                    }
                    return res.json()
                })
                .then((payload) => {
                    const pack = normalizePack(payload)
                    if (pack) setContextPack(pack)
                })
                .catch((err) => setContextPackError(err instanceof Error ? err.message : String(err)))
                .finally(() => setContextPackLoading(false))
        }

        if (generateFlag) {
            shouldCleanUrl = true
            if (paperId) {
                console.info("[P2C:M3] studio:trigger_generate", { paperId })
                generate({ paperId })
            } else {
                console.warn("[P2C:M3] studio:missing_paper_id")
                setContextPackError("Missing paper_id for generation.")
            }
        }

        if (shouldCleanUrl) {
            hasProcessedParams.current = true
            router.replace("/studio", { scroll: false })
        }
    }, [
        addPaper,
        clearGenerationProgress,
        generate,
        papers,
        router,
        searchParams,
        selectPaper,
        setContextPack,
        setContextPackError,
        setContextPackLoading,
    ])

    // Gallery view — no paper selected
    if (!selectedPaperId) {
        return <PaperGallery />
    }

    // Workspace view — paper selected
    const selectedPaper = papers.find(p => p.id === selectedPaperId)
    const paperTitle = selectedPaper?.title || "Untitled"
    const paperStatus = selectedPaper?.status || "draft"
    const config = statusConfig[paperStatus]
    const isLoading = paperStatus === "generating" || paperStatus === "running"

    return (
        <div className="flex h-screen min-h-0 flex-col">
            {/* Top Bar */}
            <div className="border-b bg-background h-11 px-3 flex items-center gap-2 shrink-0">
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0"
                    onClick={() => selectPaper(null)}
                    title="Back to gallery"
                >
                    <ArrowLeft className="h-4 w-4" />
                </Button>
                <PanelsTopLeft className="h-4 w-4 text-primary shrink-0" />
                <span className="text-sm font-medium truncate">{paperTitle}</span>
                <span
                    className={cn(
                        "inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium shrink-0",
                        config.className,
                    )}
                >
                    {isLoading && <Loader2 className="mr-1 h-2.5 w-2.5 animate-spin" />}
                    {config.label}
                </span>
            </div>

            {/* Desktop: 2-panel workspace (FilesPanel left, ReproductionLog right) */}
            <div className="hidden md:flex flex-1 min-h-0">
                <ResizablePanelGroup orientation="horizontal" className="flex-1">
                    {/* Left: Files Panel */}
                    <ResizablePanel defaultSize={18} minSize={12}>
                        <FilesPanel />
                    </ResizablePanel>

                    <ResizableHandle withHandle />

                    {/* Right: Reproduction Log */}
                    <ResizablePanel defaultSize={82} minSize={40}>
                        <ReproductionLog />
                    </ResizablePanel>
                </ResizablePanelGroup>
            </div>

            {/* Mobile: 2-tab workspace */}
            <div className="flex md:hidden flex-1 min-h-0">
                <Tabs defaultValue="log" className="h-full w-full flex flex-col">
                    <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0 h-10 shrink-0">
                        <TabsTrigger value="log" className="rounded-none text-xs h-10 px-4">
                            Reproduction
                        </TabsTrigger>
                        <TabsTrigger value="files" className="rounded-none text-xs h-10 px-4">
                            Files
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="log" className="flex-1 min-h-0 m-0">
                        <ReproductionLog />
                    </TabsContent>
                    <TabsContent value="files" className="flex-1 min-h-0 m-0">
                        <FilesPanel />
                    </TabsContent>
                </Tabs>
            </div>
        </div>
    )
}

export default function DeepCodeStudioPage() {
    return (
        <MCPProvider>
            <Suspense fallback={<div className="flex items-center justify-center h-screen">Loading...</div>}>
                <StudioContent />
            </Suspense>
        </MCPProvider>
    )
}
