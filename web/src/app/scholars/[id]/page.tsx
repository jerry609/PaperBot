import Link from "next/link"
import {
  ArrowLeft,
  ArrowUpRight,
  Bell,
  BookOpen,
  Compass,
  Gauge,
  Network,
  Sparkles,
} from "lucide-react"

import { ImpactRadar } from "@/components/paper/ImpactRadar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { fetchScholarDetails } from "@/lib/api"
import { fetchDashboardTracks } from "@/lib/dashboard-api"

function trendTone(value?: string): "default" | "secondary" | "destructive" {
  if (value === "up") return "default"
  if (value === "down") return "destructive"
  return "secondary"
}

function buildResearchLink(query: string, trackId?: number): string {
  const qs = new URLSearchParams({ query })
  if (trackId) qs.set("track_id", String(trackId))
  return `/research?${qs.toString()}`
}

export default async function ScholarProfilePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const [scholar, tracks] = await Promise.all([
    fetchScholarDetails(id),
    fetchDashboardTracks("default"),
  ])

  const activeTrack = tracks.find((track) => track.is_active) || tracks[0]
  const primaryTopic = scholar.top_topics?.[0]?.topic || scholar.keywords?.[0] || scholar.name
  const recommendation =
    scholar.status === "active" && scholar.trend_summary?.publication_trend === "up"
      ? "High priority follow"
      : scholar.status === "idle"
        ? "Monitor monthly"
        : "Follow weekly"

  const directionSummary =
    (scholar.top_topics || []).slice(0, 2).map((row) => row.topic).join(" -> ") || "Insufficient trend data"

  const evidenceTopicLink = buildResearchLink(`${scholar.name} ${primaryTopic}`, activeTrack?.id)
  const evidenceTrendLink = buildResearchLink(`${scholar.name} recent papers`, activeTrack?.id)
  const actionLink = buildResearchLink(`${scholar.keywords?.[0] || scholar.name}`, activeTrack?.id)

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
          <p className="max-w-3xl text-sm text-muted-foreground">
            Researcher intelligence cockpit: decide if this scholar is worth tracking, detect direction changes,
            and trigger next actions with evidence links.
          </p>
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
            <Badge variant="outline">No evidence, no claim</Badge>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button asChild>
            <Link href={actionLink}>Open in Research</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/settings">Manage Alerts</Link>
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

      <div className="grid gap-3 md:grid-cols-3">
        <Card>
          <CardContent className="space-y-2 p-4">
            <p className="text-xs text-muted-foreground">1) Worth following?</p>
            <p className="text-sm font-semibold">{recommendation}</p>
            <Button asChild size="sm" variant="outline">
              <Link href={evidenceTrendLink}>
                Evidence
                <ArrowUpRight className="ml-1 h-3.5 w-3.5" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-2 p-4">
            <p className="text-xs text-muted-foreground">2) Direction shift</p>
            <p className="text-sm font-semibold">{directionSummary}</p>
            <Button asChild size="sm" variant="outline">
              <Link href={evidenceTopicLink}>
                Evidence
                <ArrowUpRight className="ml-1 h-3.5 w-3.5" />
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-2 p-4">
            <p className="text-xs text-muted-foreground">3) Next action</p>
            <p className="text-sm font-semibold">Build reading queue from latest matched papers</p>
            <Button asChild size="sm">
              <Link href={actionLink}>
                Start Action
                <Compass className="ml-1 h-3.5 w-3.5" />
              </Link>
            </Button>
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
                      <Button asChild size="sm" variant="ghost" className="mt-1 h-7 px-2 text-xs">
                        <Link href={buildResearchLink(`${scholar.name} ${topic.topic}`, activeTrack?.id)}>
                          Evidence
                          <ArrowUpRight className="ml-1 h-3 w-3" />
                        </Link>
                      </Button>
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
                      <Button asChild size="sm" variant="ghost" className="mt-1 h-7 px-2 text-xs">
                        <Link href={buildResearchLink(`${scholar.name} ${row.year}`, activeTrack?.id)}>
                          Evidence
                          <ArrowUpRight className="ml-1 h-3 w-3" />
                        </Link>
                      </Button>
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
                      <Link href={buildResearchLink(paper.title, activeTrack?.id)}>Open in Research</Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        <TabsContent value="network" className="space-y-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Co-author Network</CardTitle>
            </CardHeader>
            <CardContent>
              {(scholar.co_authors || []).length === 0 ? (
                <p className="text-sm text-muted-foreground">No collaborator signals available.</p>
              ) : (
                <div className="space-y-2">
                  {(scholar.co_authors || []).slice(0, 16).map((author) => (
                    <div key={author.name} className="flex items-center justify-between gap-3 rounded-lg border p-2.5">
                      <Badge variant="secondary">
                        <Network className="mr-1 h-3.5 w-3.5" />
                        {author.name}
                      </Badge>
                      <Button asChild size="sm" variant="ghost">
                        <Link href={buildResearchLink(`${scholar.name} ${author.name}`, activeTrack?.id)}>
                          Evidence
                          <ArrowUpRight className="ml-1 h-3.5 w-3.5" />
                        </Link>
                      </Button>
                    </div>
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
                  <Link href={actionLink}>
                    <Compass className="mr-1 h-3.5 w-3.5" />
                    Open Research
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Create Track from Scholar</p>
                <p className="text-xs text-muted-foreground">
                  Use Scholars watchlist action to create a dedicated track and route papers into feed ranking.
                </p>
                <Button asChild size="sm" variant="outline">
                  <Link href="/scholars">
                    <Gauge className="mr-1 h-3.5 w-3.5" />
                    Open Watchlist Console
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Weekly Digest / Alert</p>
                <p className="text-xs text-muted-foreground">
                  Manage recurring digest delivery and keyword alerts from settings.
                </p>
                <Button asChild size="sm" variant="outline">
                  <Link href="/settings">
                    <Bell className="mr-1 h-3.5 w-3.5" />
                    Open Delivery Settings
                  </Link>
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="space-y-3 p-4">
                <p className="text-sm font-semibold">Submission Window Radar</p>
                <p className="text-xs text-muted-foreground">
                  Link scholar trends with deadline radar and saved-paper decisions.
                </p>
                <div className="flex flex-wrap gap-2">
                  <Button asChild size="sm" variant="outline">
                    <Link href="/dashboard">
                      <Bell className="mr-1 h-3.5 w-3.5" />
                      Open Dashboard
                    </Link>
                  </Button>
                  <Button asChild size="sm" variant="outline">
                    <Link href="/papers">
                      <BookOpen className="mr-1 h-3.5 w-3.5" />
                      Open Papers
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="md:col-span-2">
              <CardContent className="flex flex-wrap items-center justify-between gap-3 p-4">
                <div>
                  <p className="text-sm font-semibold">Evidence-first policy</p>
                  <p className="text-xs text-muted-foreground">
                    Every AI-derived signal in this view must link to verifiable paper evidence.
                  </p>
                </div>
                <Badge variant="secondary" className="w-fit">
                  <Sparkles className="mr-1 h-3.5 w-3.5" />
                  No evidence, no claim
                </Badge>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
