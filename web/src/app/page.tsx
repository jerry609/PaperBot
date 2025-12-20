import { ActivityFeed } from "@/components/dashboard/ActivityFeed"
import { StatsCard } from "@/components/dashboard/StatsCard"
import { PipelineStatus } from "@/components/dashboard/PipelineStatus"
import { ReadingQueue } from "@/components/dashboard/ReadingQueue"
import { LLMUsageChart } from "@/components/dashboard/LLMUsageChart"
import { QuickActions } from "@/components/dashboard/QuickActions"
import { Users, FileText, Zap, BookOpen, Search, TrendingUp } from "lucide-react"
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
    <div className="flex-1 p-4 space-y-3 min-h-screen">
      {/* Compact Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Hello, Researcher</h2>
          <p className="text-sm text-muted-foreground">Daily intelligence briefing</p>
        </div>
        <div className="flex w-full sm:max-w-sm items-center gap-1.5 bg-muted/50 p-1 rounded-md border">
          <Search className="ml-2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="arXiv ID, URL or search..."
            className="h-8 border-none bg-transparent shadow-none focus-visible:ring-0 text-sm"
          />
          <Button size="sm" className="h-7 text-xs">Import</Button>
        </div>
      </div>

      {/* Stats Row - Compact */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <StatsCard title="Scholars" value={stats.tracked_scholars.toString()} description="+4" icon={Users} />
        <StatsCard title="Papers" value={stats.new_papers.toString()} description="24h" icon={FileText} />
        <StatsCard title="LLM" value={stats.llm_usage} description="tokens" icon={Zap} />
        <StatsCard title="Queue" value={stats.read_later.toString()} description="to read" icon={BookOpen} />
      </div>

      {/* Main Grid - 12 Column Dense Layout */}
      <div className="grid grid-cols-12 gap-2">
        {/* Activity Feed - Main Content */}
        <div className="col-span-12 lg:col-span-8">
          <ActivityFeed />
        </div>

        {/* Right Sidebar */}
        <div className="col-span-12 lg:col-span-4 space-y-2">
          <PipelineStatus tasks={tasks} />
          <ReadingQueue items={readingQueue} />
          <QuickActions />
        </div>

        {/* Bottom Row */}
        <div className="col-span-12 lg:col-span-6">
          <LLMUsageChart data={llmUsage} />
        </div>

        <Card className="col-span-12 lg:col-span-6">
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-orange-500" /> Trending
            </CardTitle>
          </CardHeader>
          <CardContent className="py-2 px-4">
            <div className="flex flex-wrap gap-1.5">
              {trends.map((topic, i) => (
                <Badge
                  key={i}
                  variant="secondary"
                  className="text-xs cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors"
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
