import { fetchWikiConcepts } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import {
    Search, Book, Lightbulb, ArrowRight, Network,
    Layers, Target, BarChart2, Waves, Image as ImageIcon,
    BookOpen, type LucideIcon
} from "lucide-react"

// Icon mapping: string identifier -> Lucide component
const iconMap: Record<string, LucideIcon> = {
    "layers": Layers,
    "target": Target,
    "bar-chart": BarChart2,
    "waves": Waves,
    "image": ImageIcon,
}

export default async function WikiPage() {
    const concepts = await fetchWikiConcepts()
    const categories = ["All", "Method", "Architecture", "Metric", "Dataset", "Task"]

    return (
        <div className="flex-1 p-8 pt-6 min-h-screen">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight flex items-center gap-3">
                        <BookOpen className="h-8 w-8 text-primary" />
                        Knowledge Base
                    </h2>
                    <p className="text-muted-foreground mt-1">
                        Explore {concepts.length} core concepts in AI/ML research.
                    </p>
                </div>
                <div className="flex w-full md:max-w-md items-center gap-2 bg-background p-1.5 rounded-lg border shadow-sm">
                    <Search className="ml-2 h-4 w-4 text-muted-foreground" />
                    <Input
                        type="text"
                        placeholder="Search concepts, methods, metrics..."
                        className="border-none shadow-none focus-visible:ring-0"
                    />
                    <Button size="sm">Search</Button>
                </div>
            </div>

            {/* Category Tabs */}
            <Tabs defaultValue="All" className="w-full">
                <TabsList className="mb-6 bg-muted/50">
                    {categories.map(cat => (
                        <TabsTrigger key={cat} value={cat} className="data-[state=active]:bg-background">
                            {cat}
                        </TabsTrigger>
                    ))}
                </TabsList>

                {categories.map(cat => (
                    <TabsContent key={cat} value={cat}>
                        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                            {concepts
                                .filter(c => cat === "All" || c.category === cat)
                                .map((concept) => {
                                    const IconComponent = iconMap[concept.icon] || Layers
                                    return (
                                        <Card key={concept.id} className="group hover:shadow-md transition-all duration-200 hover:border-primary/30">
                                            <CardHeader className="pb-3">
                                                <div className="flex items-start gap-3">
                                                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                                                        <IconComponent className="h-5 w-5" />
                                                    </div>
                                                    <div className="flex-1">
                                                        <CardTitle className="text-base">{concept.name}</CardTitle>
                                                        <Badge variant="secondary" className="mt-1 text-xs font-normal">{concept.category}</Badge>
                                                    </div>
                                                </div>
                                            </CardHeader>

                                            <CardContent className="space-y-3 pt-0">
                                                <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2">
                                                    {concept.description}
                                                </p>

                                                <Separator />

                                                {/* Definition */}
                                                <div>
                                                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1">
                                                        <Lightbulb className="h-3 w-3" /> Definition
                                                    </h4>
                                                    <p className="text-sm line-clamp-2">{concept.definition}</p>
                                                </div>

                                                {/* Related Concepts */}
                                                <div>
                                                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1">
                                                        <Network className="h-3 w-3" /> Related Concepts
                                                    </h4>
                                                    <div className="flex flex-wrap gap-1">
                                                        {concept.related_concepts.slice(0, 3).map((rc, i) => (
                                                            <Badge key={i} variant="outline" className="text-xs font-normal">
                                                                {rc}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>

                                                {/* Key Papers */}
                                                <div>
                                                    <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1 flex items-center gap-1">
                                                        <Book className="h-3 w-3" /> Key Papers
                                                    </h4>
                                                    <div className="flex flex-wrap gap-1">
                                                        {concept.related_papers.slice(0, 2).map((paper, i) => (
                                                            <Badge key={i} variant="outline" className="text-xs font-normal">
                                                                {paper}
                                                            </Badge>
                                                        ))}
                                                    </div>
                                                </div>

                                                {/* Examples */}
                                                {concept.examples.length > 0 && (
                                                    <p className="text-xs text-muted-foreground">
                                                        <span className="font-medium">Examples:</span> {concept.examples.slice(0, 2).join(", ")}
                                                    </p>
                                                )}

                                                {/* Explore Button */}
                                                <Button variant="ghost" size="sm" className="w-full mt-1 text-primary hover:bg-primary/5">
                                                    Explore Concept <ArrowRight className="ml-1 h-4 w-4" />
                                                </Button>
                                            </CardContent>
                                        </Card>
                                    )
                                })}
                        </div>
                    </TabsContent>
                ))}
            </Tabs>
        </div>
    )
}
