import { ActivityFeed } from "@/components/dashboard/ActivityFeed"
import { StatsCard } from "@/components/dashboard/StatsCard"
import { Users, FileText, Zap, BookOpen } from "lucide-react"

export default function DashboardPage() {
  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">The Morning Paper</h2>
        <div className="flex items-center space-x-2">
          {/* Add Date Picker or controls here */}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="Tracked Scholars"
          value="128"
          description="+4 this week"
          icon={Users}
        />
        <StatsCard
          title="New Papers"
          value="12"
          description="Last 24 hours"
          icon={FileText}
        />
        <StatsCard
          title="LLM Usage"
          value="45k"
          description="Tokens used today"
          icon={Zap}
        />
        <StatsCard
          title="Read Later"
          value="8"
          description="From 3 venues"
          icon={BookOpen}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <ActivityFeed />
        {/* Right Stats / Quick Actions can go here if needed, currently ActivityFeed takes 4 cols */}
      </div>
    </div>
  )
}
