import { ActivityFeed } from "@/components/dashboard/ActivityFeed"
import { StatsCard } from "@/components/dashboard/StatsCard"
import { PipelineStatus } from "@/components/dashboard/PipelineStatus"
import { ReadingQueue } from "@/components/dashboard/ReadingQueue"
import { LLMUsageChart } from "@/components/dashboard/LLMUsageChart"
import { QuickActions } from "@/components/dashboard/QuickActions"
import { Users, FileText, Zap, BookOpen, Download, Search } from "lucide-react"
import { fetchStats, fetchTrendingTopics, fetchPipelineTasks, fetchReadingQueue, fetchLLMUsage } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

export default async function DashboardPage() {
  const [stats, trends, tasks, readingQueue, llmUsage] = await Promise.all([
    fetchStats(),
    fetchTrendingTopics(),
    fetchPipelineTasks(),
    fetchReadingQueue(),
    fetchLLMUsage()
  ])

  return (
    <div className="flex-1 p-8 pt-6 space-y-6 bg-slate-50/50 dark:bg-slate-950/50 min-h-screen">
      {/* Premium Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">Hello, Researcher</h2>
          <p className="text-muted-foreground mt-1">Here is your daily intelligence briefing.</p>
        </div>
        <div className="flex w-full md:max-w-md items-center gap-2 bg-background p-1 rounded-lg border shadow-sm">
          <Search className="ml-2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Paste arXiv ID, URL or search papers..."
            className="border-none shadow-none focus-visible:ring-0"
          />
          <Button slot="append" size="sm">Import</Button>
        </div>
      </div>

      {/* Stats Overview - Premium Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Tracked Scholars"
          value={stats.tracked_scholars.toString()}
          description="+4 this week"
          icon={Users}
        />
        <StatsCard
          title="New Papers"
          value={stats.new_papers.toString()}
          description="Last 24 hours"
          icon={FileText}
        />
        <StatsCard
          title="LLM Usage"
          value={stats.llm_usage}
          description="Tokens used today"
          icon={Zap}
        />
        <StatsCard
          title="Read Later"
          value={stats.read_later.toString()}
          description="From 3 venues"
          icon={BookOpen}
        />
      </div>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-7 gap-4 auto-rows-[minmax(180px,auto)]">

        {/* Large Item: Activity Feed (2 Cols, 2 Rows) */}
        <div className="md:col-span-2 lg:col-span-4 row-span-2">
          <ActivityFeed />
        </div>

        {/* Tall Item: Pipeline Status (1 Col, 2 Rows) */}
        <div className="md:col-span-1 lg:col-span-3 row-span-2 h-full">
          <PipelineStatus tasks={tasks} />
        </div>

        {/* Medium: LLM Chart (2 Cols) */}
        <div className="md:col-span-2 lg:col-span-3">
          <LLMUsageChart data={llmUsage} />
        </div>

        {/* Medium: Reading Queue (2 Cols) */}
        <div className="md:col-span-1 lg:col-span-2">
          <ReadingQueue items={readingQueue} />
        </div>

        {/* Small: Quick Actions (1 Col) */}
        <div className="md:col-span-1 lg:col-span-2">
          <QuickActions />
        </div>

        {/* Trending Word Cloud (Wide or Flexible) */}
        <Card className="md:col-span-3 lg:col-span-7 bg-muted/20 border-dashed">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              🔥 Trending Concepts
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap justify-center gap-3">
              {trends.map((topic, i) => (
                <Badge
                  key={i}
                  variant="secondary"
                  className="py-1 px-3 cursor-pointer hover:scale-105 transition-transform hover:bg-primary hover:text-primary-foreground"
                  style={{ fontSize: `${Math.max(0.8, topic.value / 60)}rem`, opacity: Math.max(0.6, topic.value / 100) }}
                >
                  {topic.text}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
