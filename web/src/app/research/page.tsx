import { Suspense } from "react"
import { redirect } from "next/navigation"

import { auth } from "@/auth"
import ResearchPageNew from "@/components/research/ResearchPageNew"

export default async function ResearchPage() {
  const session = await auth()
  if (!session) {
    redirect("/login?callbackUrl=/research")
  }

  return (
    <div className="flex-1 bg-stone-50/50 dark:bg-background">
      <Suspense fallback={<div className="p-4 text-sm text-muted-foreground">Loading research workspace...</div>}>
        <ResearchPageNew />
      </Suspense>
    </div>
  )
}
