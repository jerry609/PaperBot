"use client"

import { useEffect, Suspense, useRef } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { PaperGallery } from "@/components/studio/PaperGallery"
import { AgentWorkspace } from "@/components/studio/AgentWorkspace"
import { MCPProvider } from "@/lib/mcp"
import { useStudioStore } from "@/lib/store/studio-store"
import { useContextPackGeneration } from "@/hooks/useContextPackGeneration"
import { normalizePack } from "@/lib/context-pack-utils"
import { backendUrl } from "@/lib/backend-url"

function StudioContent() {
    const {
        addPaper,
        selectPaper,
        loadPapers,
        papers,
        selectedPaperId,
        contextPack,
        setContextPack,
        setContextPackLoading,
        setContextPackError,
        clearGenerationProgress,
        updatePaper,
    } = useStudioStore()
    const { generate } = useContextPackGeneration()
    const searchParams = useSearchParams()
    const router = useRouter()
    const requestedSurface = searchParams.get("surface")
    const requestedPaperId = searchParams.get("paperId") || searchParams.get("paper_id")
    const hasProcessedParams = useRef(false)
    const latestContextLookupRef = useRef<Set<string>>(new Set())
    const contextFetchInFlightRef = useRef(false)

    // Load papers from localStorage on mount
    useEffect(() => {
        loadPapers()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useEffect(() => {
        if (requestedSurface !== "board") return

        const params = new URLSearchParams()
        const targetPaperId = requestedPaperId || selectedPaperId
        if (targetPaperId) {
            params.set("paperId", targetPaperId)
        }
        const query = params.toString()
        router.replace(query ? `/studio/agent-board?${query}` : "/studio/agent-board", { scroll: false })
    }, [requestedPaperId, requestedSurface, router, selectedPaperId])

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

    useEffect(() => {
        if (!selectedPaperId || contextFetchInFlightRef.current) return
        const selected = papers.find((paper) => paper.id === selectedPaperId)
        if (!selected) return

        let cancelled = false

        const fetchContextPack = async (contextPackId: string) => {
            contextFetchInFlightRef.current = true
            setContextPackLoading(true)
            setContextPackError(null)
            try {
                const res = await fetch(`/api/research/repro/context/${contextPackId}`)
                if (!res.ok) {
                    throw new Error(`Failed to load context pack (${res.status})`)
                }
                const payload = await res.json()
                if (cancelled) return
                const pack = normalizePack(payload)
                if (pack) setContextPack(pack)
            } catch (err) {
                if (!cancelled) {
                    setContextPackError(err instanceof Error ? err.message : String(err))
                }
            } finally {
                contextFetchInFlightRef.current = false
                if (!cancelled) {
                    setContextPackLoading(false)
                }
            }
        }

        const lookupLatestSessionForContext = async () => {
            if (latestContextLookupRef.current.has(selectedPaperId)) return
            latestContextLookupRef.current.add(selectedPaperId)
            try {
                const latestResp = await fetch(
                    backendUrl(`/api/agent-board/sessions/latest/by-paper?paper_id=${encodeURIComponent(selectedPaperId)}`),
                )
                if (!latestResp.ok) return
                const latestPayload = (await latestResp.json()) as {
                    found?: unknown
                    session_id?: unknown
                    context_pack_id?: unknown
                }
                if (cancelled) return
                if (latestPayload.found === false) return

                const latestContextPackId =
                    typeof latestPayload.context_pack_id === "string"
                        ? latestPayload.context_pack_id.trim()
                        : ""
                const latestBoardSessionId =
                    typeof latestPayload.session_id === "string" ? latestPayload.session_id.trim() : ""

                if (latestContextPackId || latestBoardSessionId) {
                    updatePaper(selectedPaperId, {
                        ...(latestContextPackId ? { contextPackId: latestContextPackId } : {}),
                        ...(!selected.boardSessionId && latestBoardSessionId
                            ? { boardSessionId: latestBoardSessionId }
                            : {}),
                    })
                }

                if (latestContextPackId && contextPack?.context_pack_id !== latestContextPackId) {
                    await fetchContextPack(latestContextPackId)
                }
            } catch {
                // Keep current UI state; user can still regenerate manually.
            }
        }

        const currentContextPackId = selected.contextPackId?.trim() || ""
        if (currentContextPackId) {
            if (contextPack?.context_pack_id !== currentContextPackId) {
                void fetchContextPack(currentContextPackId)
            }
        } else {
            void lookupLatestSessionForContext()
        }

        return () => {
            cancelled = true
            contextFetchInFlightRef.current = false
        }
    }, [
        contextPack?.context_pack_id,
        papers,
        selectedPaperId,
        setContextPack,
        setContextPackError,
        setContextPackLoading,
        updatePaper,
    ])

    const defaultCenterView =
        requestedSurface === "context" ||
        requestedSurface === "skills" ||
        requestedSurface === "log" ||
        requestedSurface === "commands"
            ? requestedSurface === "skills"
                ? "context"
                : requestedSurface
            : "log"

    if (requestedSurface === "board") {
        return null
    }

    // Gallery view — no paper selected
    if (!selectedPaperId) {
        return <PaperGallery />
    }

    return (
        <AgentWorkspace
            defaultCenterView={defaultCenterView}
            onBackToStudio={() => selectPaper(null)}
        />
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
