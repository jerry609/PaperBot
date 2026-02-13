import Link from "next/link"
import { ArrowLeft, ArrowUpRight, BookOpen, Compass, Network, Sparkles, Workflow } from "lucide-react"

import { ImpactRadar } from "@/components/paper/ImpactRadar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { fetchScholarDetails } from "@/lib/api"

function trendTone(value?: string): "default" | "secondary" | "destructive" {
  if (value === "up") return "default"
  if (value === "down") return "destructive"
  return "secondary"
}

export default async function ScholarProfilePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const scholar = await fetchScholarDetails(id)

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 pb-10 sm:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          <Button asChild variant="ghost" size="sm" className="px-0">
            <Link href="/scholars">
              <ArrowLeft className="mr-1 h-4 w-4" />
              Back to Scholars
            </Link>
          </Button>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{scholar.name}</h1>
          <p className="text-sm text-muted-foreground">{scholar.affiliation}</p>
          <div className="flex flex-wrap gap-1.5">
            <Badge variant={scholar.status === "active" ? "default" : "secondary"} className="capitalize">
              {scholar.status}
            </Badge>
            {scholar.trend_summary ? (
              <>
                <Badge variant={trendTone(scholar.trend_summary.publication_trend)}>
                  publication {scholar.trend_summary.publication_trend}
                </Badge>
                <Badge variant={trendTone(scholar.trend_summary.citation_trend)}>
                  citation {scholar.trend_summary.citation_trend}
                </Badge>
              </>
            ) : null}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline">
            <Link href={`/research?query=${encodeURIComponent(scholar.name)}`}>Search in Research</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/workflows">Run Tracking Workflow</Link>
          </Button>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">H-Index</p>
            <p className="mt-1 text-2xl font-semibold">{scholar.stats.h_index}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Total Citations</p>
            <p className="mt-1 text-2xl font-semibold">{scholar.stats.total_citations.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Tracked Papers</p>
            <p className="mt-1 text-2xl font-semibold">{scholar.stats.papers_count}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Cached Papers</p>
            <p className="mt-1 text-2xl font-semibold">{scholar.cached_papers || 0}</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="signals" className="space-y-4">
        <TabsList>
          <TabsTrigger value="signals">Signals</TabsTrigger>
          <TabsTrigger value="papers">Papers</TabsTrigger>
          <TabsTrigger value="network">Network</TabsTrigger>
          <TabsTrigger value="actions">Actions</TabsTrigger>
        </TabsList>

        <TabsContent value="signals" className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-7">
            <Card className="lg:col-span-4">
              <CardHeader>
                <CardTitle className="text-base">Signal Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">{scholar.bio}</p>
                <div className="grid gap-2 sm:grid-cols-2">
                  {(scholar.top_topics || []).slice(0, 6).map((topic) => (
                    <div key={topic.topic} className="rounded-lg border p-2.5">
                      <p className="text-sm font-medium">{topic.topic}</p>
                      <p className="text-xs text-muted-foreground">{topic.count} papers in window</p>
                    </div>
                  ))}
                </div>
                {(scholar.top_venues || []).length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {(scholar.top_venues || []).slice(0, 5).map((venue) => (
                      <Badge key={venue.venue} variant="outline">
                        {venue.venue} ({venue.count})
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card className="lg:col-span-3">
              <CardHeader>
                <CardTitle className="text-base">Topic Radar</CardTitle>
              </CardHeader>
              <CardContent>
                <ImpactRadar data={scholar.expertise_radar} />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Publication Velocity</CardTitle>
            </CardHeader>
            <CardContent>
              {(scholar.publication_velocity || []).length === 0 ? (
                <p className="text-sm text-muted-foreground">No publication velocity data yet.</p>
              ) : (
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
                  {(scholar.publication_velocity || []).map((row) => (
                    <div key={row.year} className="rounded-lg border p-3">
                      <p className="text-sm font-medium">{row.year}</p>
                      <p className="text-xs text-muted-foreground">papers {row.papers}</p>
                      <p className="text-xs text-muted-foreground">citations {row.citations}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="papers" className="space-y-3">
          {(scholar.publications || []).length === 0 ? (
            <Card>
              <CardContent className="p-6 text-sm text-muted-foreground">No recent papers found for this scholar.</CardContent>
            </Card>
          ) : (
            (scholar.publications || []).map((paper) => (
              <Card key={paper.id}>
                <CardContent className="flex flex-wrap items-start justify-between gap-3 p-4">
                  <div className="space-y-1">
                    <p className="text-sm font-semibold">{paper.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {paper.venue} · {paper.citations} citations
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {(paper.tags || []).slice(0, 3).map((tag) => (
                        <Badge key={`${paper.id}-${tag}`} variant="outline" className="text-[11px]">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {paper.url ? (
                      <Button asChild variant="outline" size="sm">
                        <a href={paper.url} target="_blank" rel="noreferrer">
                          Evidence
                          <ArrowUpRight className="ml-1 h-3.5 w-3.5" />
                        </a>
                      </Button>
                    ) : null}
                    <Button asChild size="sm" variant="ghost">
                      <Link href={`/research?query=${encodeURIComponent(paper.title)}`}>Open in Research</Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="network">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Co-author Network</CardTitle>
            </CardHeader>
            <CardContent>
              {(scholar.co_authors || []).length === 0 ? (
                <p className="text-sm text-muted-foreground">No collaborator signals available.</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {(scholar.co_authors || []).map((author) => (
                    <Badge key={author.name} variant="secondary">
                      <Network className="mr-1 h-3.5 w-3.5" />
                      {author.name}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="actions">
          <div className="grid gap-3 md:grid-cols-2">
            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Route to Research Feed</p>
                <p className="text-xs text-muted-foreground">
                  Build context pack with scholar keywords and evaluate candidate papers.
                </p>
                <Button asChild size="sm">
                  <Link href={`/research?query=${encodeURIComponent(scholar.keywords?.[0] || scholar.name)}`}>
                    <Compass className="mr-1 h-3.5 w-3.5" />
                    Open Research
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Track in Workflows</p>
                <p className="text-xs text-muted-foreground">
                  Trigger scholar pipeline and keep monitoring new papers.
                </p>
                <Button asChild size="sm" variant="outline">
                  <Link href="/workflows">
                    <Workflow className="mr-1 h-3.5 w-3.5" />
                    Open Workflows
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Review Saved Papers</p>
                <p className="text-xs text-muted-foreground">
                  Jump to library and compare saved papers with this scholar trend.
                </p>
                <Button asChild size="sm" variant="outline">
                  <Link href="/papers">
                    <BookOpen className="mr-1 h-3.5 w-3.5" />
                    Open Library
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Signal Snapshot</p>
                <p className="text-xs text-muted-foreground">
                  Publication trend: {scholar.trend_summary?.publication_trend || "flat"} · Citation trend: {scholar.trend_summary?.citation_trend || "flat"}
                </p>
                <Badge variant="secondary" className="w-fit">
                  <Sparkles className="mr-1 h-3.5 w-3.5" />
                  Evidence-first mode
                </Badge>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
