"use client"

import { useEffect, Suspense, useRef } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { PanelsTopLeft } from "lucide-react"
import { PapersPanel } from "@/components/studio/PapersPanel"
import { ReproductionLog } from "@/components/studio/ReproductionLog"
import { FilesPanel } from "@/components/studio/FilesPanel"
import { MCPProvider } from "@/lib/mcp"
import { useStudioStore } from "@/lib/store/studio-store"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable"

function StudioContent() {
    const { addPaper, selectPaper, loadPapers, papers } = useStudioStore()
    const searchParams = useSearchParams()
    const router = useRouter()
    const hasProcessedParams = useRef(false)

    // Load papers from localStorage on mount
    useEffect(() => {
        loadPapers()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    // Handle URL params for paper_id (from /papers page "Run Reproduction" button)
    useEffect(() => {
        if (hasProcessedParams.current) return

        const paperId = searchParams.get("paper_id")
        const title = searchParams.get("title")
        const abstract = searchParams.get("abstract")

        if (paperId) {
            hasProcessedParams.current = true
            const existingPaper = papers.find(p => p.id === paperId)
            if (existingPaper) {
                selectPaper(paperId)
            } else if (title) {
                addPaper({
                    title: title || "Untitled Paper",
                    abstract: abstract || "",
                })
            }
            router.replace("/studio", { scroll: false })
        } else if (title && abstract) {
            hasProcessedParams.current = true
            addPaper({ title, abstract })
            router.replace("/studio", { scroll: false })
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchParams, papers])

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
