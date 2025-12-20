import { fetchPaperDetails } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Download, Play, MessageSquare, FileCode } from "lucide-react"

import { ImpactRadar } from "@/components/paper/ImpactRadar"
import { SentimentChart } from "@/components/paper/SentimentChart"
import { VelocityChart } from "@/components/paper/VelocityChart"

export default async function PaperPage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params
    const paper = await fetchPaperDetails(id)

    return (
        <div className="flex-1 space-y-4 p-8 pt-6 h-screen flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between space-y-2">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">{paper.title}</h2>
                    <p className="text-muted-foreground mt-1">
                        {paper.authors} • {paper.venue} • {paper.citations} citations
                    </p>
                </div>
                <div className="flex items-center space-x-2">
                    <Button variant="outline">
                        <Download className="mr-2 h-4 w-4" /> PDF
                    </Button>
                    <Button variant="outline">
                        <MessageSquare className="mr-2 h-4 w-4" /> Chat
                    </Button>
                    <Button>
                        <Play className="mr-2 h-4 w-4" /> Run Reproduction
                    </Button>
                </div>
            </div>

            <Separator />

            {/* Content */}
            <Tabs defaultValue="overview" className="flex-1 flex flex-col">
                <div className="flex items-center justify-between">
                    <TabsList>
                        <TabsTrigger value="overview">Overview</TabsTrigger>
                        <TabsTrigger value="intelligence">Deep Intelligence</TabsTrigger>
                        <TabsTrigger value="reproduction">Reproduction</TabsTrigger>
                    </TabsList>
                </div>

                <TabsContent value="overview" className="flex-1 space-y-4 mt-4">
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7 h-full">
                        {/* Left Column: Abstract & TLDR */}
                        <Card className="col-span-4 border-none shadow-none">
                            <CardHeader className="px-0 pt-0">
                                <CardTitle>Abstract</CardTitle>
                            </CardHeader>
                            <CardContent className="px-0">
                                <p className="leading-7 text-justify text-muted-foreground">
                                    {paper.abstract}
                                </p>
                                <div className="mt-6 p-4 bg-muted/50 rounded-lg border">
                                    <h4 className="font-semibold mb-2 flex items-center gap-2">
                                        <span className="text-xl">🤖</span> AI TL;DR
                                    </h4>
                                    <p className="text-sm italic">{paper.tldr}</p>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Right Column: Radar Chart */}
                        <Card className="col-span-3">
                            <CardHeader>
                                <CardTitle>Impact Radar</CardTitle>
                                <CardDescription>PIS Score Breakdown</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ImpactRadar data={paper.impact_radar} />
                                <div className="text-center mt-4">
                                    <span className="text-4xl font-bold text-primary">{paper.pis_score}</span>
                                    <span className="text-muted-foreground text-sm"> / 100 PIS</span>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="intelligence" className="space-y-4 mt-4">
                    <div className="grid gap-4 md:grid-cols-2">
                        <Card>
                            <CardHeader>
                                <CardTitle>Citation Sentiment</CardTitle>
                                <CardDescription>How the community discusses this paper</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <SentimentChart data={paper.sentiment_analysis} />
                            </CardContent>
                        </Card>
                        <Card>
                            <CardHeader>
                                <CardTitle>Growth Velocity</CardTitle>
                                <CardDescription>Citations gained over last 6 months</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <VelocityChart data={paper.citation_velocity} />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="reproduction" className="flex-1 mt-4 h-full">
                    <div className="grid gap-4 md:grid-cols-2 h-[500px]">
                        <Card className="flex flex-col h-full">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <FileCode className="h-4 w-4" /> Dockerfile (Generated)
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="flex-1 p-0">
                                <ScrollArea className="h-[400px] w-full rounded-md border p-4 font-mono text-sm bg-muted/50">
                                    <pre>{paper.reproduction.dockerfile}</pre>
                                </ScrollArea>
                            </CardContent>
                        </Card>
                        <Card className="flex flex-col h-full">
                            <CardHeader>
                                <CardTitle>Execution Logs</CardTitle>
                            </CardHeader>
                            <CardContent className="flex-1 p-0">
                                <ScrollArea className="h-[400px] w-full rounded-md border p-4 font-mono text-sm bg-black text-green-400">
                                    {paper.reproduction.logs.map((log, i) => (
                                        <div key={i} className="mb-1 border-b border-white/10 pb-1 last:border-0">{log}</div>
                                    ))}
                                </ScrollArea>
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    )
}
