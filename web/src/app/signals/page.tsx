import { redirect } from "next/navigation"

import { auth } from "@/auth"
import SignalsWorkspace from "@/components/signals/SignalsWorkspace"
import { fetchDashboardTracks, fetchIntelligenceFeed } from "@/lib/dashboard-api"

export default async function SignalsPage() {
  const session = await auth()
  if (!session) {
    redirect("/login?callbackUrl=/signals")
  }

  const accessToken = session.accessToken
  const [tracksResult, feedResult] = await Promise.allSettled([
    fetchDashboardTracks(accessToken),
    fetchIntelligenceFeed(accessToken, 20, { sortBy: "delta", sortOrder: "desc" }),
  ])

  const tracks = tracksResult.status === "fulfilled" ? tracksResult.value : []
  const initialFeed = feedResult.status === "fulfilled"
    ? feedResult.value
    : {
        items: [],
        refreshed_at: null,
        refresh_scheduled: false,
        keywords: [],
        watch_repos: [],
        subreddits: [],
      }
  const initialNowMs = Date.now()

  return (
    <div className="flex-1 bg-stone-50/60">
      <SignalsWorkspace
        initialFeed={initialFeed}
        initialTracks={tracks}
        initialNowMs={initialNowMs}
      />
    </div>
  )
}
