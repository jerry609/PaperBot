import { fetchWikiConcepts } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Search, Book } from "lucide-react"

export default async function WikiPage() {
    const concepts = await fetchWikiConcepts()

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Knowledge Base</h2>
                    <p className="text-muted-foreground"> Explore concepts, methods, and metrics in AI Science.</p>
                </div>
                <div className="flex w-full max-w-sm items-center space-x-2">
                    <Input type="email" placeholder="Search concepts..." />
                    <Button type="submit" size="icon"><Search className="h-4 w-4" /></Button>
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 mt-6">
                {concepts.map((concept) => (
                    <Card key={concept.id} className="hover:bg-muted/50 transition-colors cursor-pointer group">
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <CardTitle className="group-hover:text-primary transition-colors">{concept.name}</CardTitle>
                                <Badge variant="outline">{concept.category}</Badge>
                            </div>
                            <CardDescription className="line-clamp-2">
                                {concept.description}
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex flex-col gap-2">
                                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Related Papers</span>
                                <div className="flex flex-wrap gap-2">
                                    {concept.related_papers.length > 0 ? (
                                        concept.related_papers.map((paper, i) => (
                                            <Badge key={i} variant="secondary" className="text-[10px] flex items-center gap-1">
                                                <Book className="h-3 w-3" /> {paper}
                                            </Badge>
                                        ))
                                    ) : (
                                        <span className="text-xs text-muted-foreground italic">No papers linked yet.</span>
                                    )}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </div>
    )
}
