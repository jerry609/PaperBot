import { redirect } from "next/navigation"

import { auth } from "@/auth"
import SavedPapersList from "@/components/research/SavedPapersList"

export default async function PapersPage() {
  const session = await auth()
  if (!session) {
    redirect("/login?callbackUrl=/papers")
  }

  return (
    <div className="flex-1 space-y-4 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Papers Library</h2>
          <p className="text-sm text-muted-foreground">Saved papers from registry feedback and reading states.</p>
        </div>
      </div>
      <SavedPapersList />
    </div>
  )
}
