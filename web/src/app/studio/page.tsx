"use client"

import { useEffect, Suspense, useRef } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { PanelsTopLeft } from "lucide-react"
import { PapersPanel } from "@/components/studio/PapersPanel"
import { ReproductionLog } from "@/components/studio/ReproductionLog"
import { FilesPanel } from "@/components/studio/FilesPanel"
import { MCPProvider } from "@/lib/mcp"
import { useStudioStore } from "@/lib/store/studio-store"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import type { ReproContextPack } from "@/lib/types/p2c"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable"

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

    return (
        <div className="flex h-screen min-h-0 flex-col">
            {/* Top Bar - minimal */}
            <div className="border-b bg-background h-11 px-4 flex items-center gap-2 shrink-0">
                <PanelsTopLeft className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">DeepCode Studio</span>
            </div>

            {/* Desktop: 3-panel CodePilot-style layout */}
            <div className="hidden md:flex flex-1 min-h-0">
                {/* Left: Papers Panel - fixed width */}
                <div className="w-64 shrink-0 border-r">
                    <PapersPanel />
                </div>

                {/* Middle + Right: Resizable */}
                <ResizablePanelGroup orientation="horizontal" className="flex-1">
                    {/* Middle: Reproduction Log */}
                    <ResizablePanel defaultSize={75} minSize={40}>
                        <ReproductionLog />
                    </ResizablePanel>

                    <ResizableHandle withHandle />

                    {/* Right: Files Panel */}
                    <ResizablePanel defaultSize={25} minSize={15}>
                        <FilesPanel />
                    </ResizablePanel>
                </ResizablePanelGroup>
            </div>

            {/* Mobile: Tab navigation */}
            <div className="flex md:hidden flex-1 min-h-0">
                <Tabs defaultValue="papers" className="h-full w-full flex flex-col">
                    <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0 h-10 shrink-0">
                        <TabsTrigger value="papers" className="rounded-none text-xs h-10 px-4">
                            Papers
                        </TabsTrigger>
                        <TabsTrigger value="log" className="rounded-none text-xs h-10 px-4">
                            Reproduction
                        </TabsTrigger>
                        <TabsTrigger value="files" className="rounded-none text-xs h-10 px-4">
                            Files
                        </TabsTrigger>
                    </TabsList>
                    <TabsContent value="papers" className="flex-1 min-h-0 m-0">
                        <PapersPanel />
                    </TabsContent>
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
