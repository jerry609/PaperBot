import { fetchScholarDetails } from "@/lib/api"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { MapPin, Link as LinkIcon, Users, BookOpen, TrendingUp } from "lucide-react"
import Link from "next/link"
import { ImpactRadar } from "@/components/paper/ImpactRadar"

export default async function ScholarProfilePage({ params }: { params: Promise<{ id: string }> }) {
    const { id } = await params
    const scholar = await fetchScholarDetails(id)

    return (
        <div className="flex-1 space-y-4 p-8 pt-6 h-screen flex flex-col">
            {/* Header Profile */}
            <div className="flex items-start justify-between">
                <div className="flex items-center gap-6">
                    <Avatar className="h-24 w-24 border-4 border-background shadow-sm">
                        <AvatarImage src={`https://avatar.vercel.sh/${scholar.id}.png`} />
                        <AvatarFallback>{scholar.name[0]}</AvatarFallback>
                    </Avatar>
                    <div>
                        <h2 className="text-3xl font-bold tracking-tight">{scholar.name}</h2>
                        <p className="text-lg text-muted-foreground">{scholar.affiliation}</p>
                        <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                            <div className="flex items-center gap-1"><MapPin className="h-4 w-4" /> {scholar.location}</div>
                            <div className="flex items-center gap-1"><LinkIcon className="h-4 w-4" /> <a href={scholar.website} className="hover:underline" target="_blank">Website</a></div>
                        </div>
                    </div>
                </div>
                <div className="flex gap-2">
                    <Button>Follow</Button>
                    <Button variant="outline">Generate Report</Button>
                </div>
            </div>

            <div className="grid grid-cols-4 gap-4 mt-6">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">H-Index</CardTitle>
                        <TrendingUp className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{scholar.stats.h_index}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Citations</CardTitle>
                        <BookOpen className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{scholar.stats.total_citations.toLocaleString()}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Papers</CardTitle>
                        <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{scholar.stats.papers_count}</div>
                    </CardContent>
                </Card>
            </div>

            <Tabs defaultValue="overview" className="flex-1 mt-6">
                <TabsList>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="publications">Publications</TabsTrigger>
                    <TabsTrigger value="network">Network</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4 mt-4">
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                        <Card className="col-span-4">
                            <CardHeader>
                                <CardTitle>Biography</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <p className="leading-7 text-muted-foreground">{scholar.bio}</p>
                            </CardContent>
                        </Card>
                        <Card className="col-span-3">
                            <CardHeader>
                                <CardTitle>Expertise Radar</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <ImpactRadar data={scholar.expertise_radar} />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="publications" className="mt-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Recent Publications</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4">
                                {scholar.publications.map(paper => (
                                    <div key={paper.id} className="flex items-center justify-between border-b pb-4 last:border-0 last:pb-0">
                                        <div>
                                            <div className="font-semibold hover:underline cursor-pointer">
                                                <Link href={`/papers/${paper.id}`}>{paper.title}</Link>
                                            </div>
                                            <div className="text-sm text-muted-foreground">
                                                {paper.venue} • {paper.citations} citations • {paper.tags.join(", ")}
                                            </div>
                                        </div>
                                        <Badge variant={paper.status === "Reproduced" ? "default" : "secondary"}>{paper.status}</Badge>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="network" className="mt-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Co-Author Network</CardTitle>
                            <CardDescription>Filtering by frequent collaborators</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-6 flex-wrap">
                                {scholar.co_authors.map((author, i) => (
                                    <div key={i} className="flex flex-col items-center gap-2">
                                        <Avatar className="h-16 w-16">
                                            <AvatarImage src={author.avatar} />
                                            <AvatarFallback>{author.name[0]}</AvatarFallback>
                                        </Avatar>
                                        <span className="text-sm font-medium">{author.name}</span>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
