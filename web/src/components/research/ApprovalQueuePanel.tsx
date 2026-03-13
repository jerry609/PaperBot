"use client"

import { useCallback, useEffect, useState } from "react"
import { Check, Loader2, RefreshCw, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { getErrorMessage } from "@/lib/fetch"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

type ApprovalItem = {
  session_id: string
  status: string
  checkpoint?: string
  updated_at?: string | null
  title?: string
  query_count?: number
  unique_items?: number
}

type ApprovalQueueResponse = {
  items: ApprovalItem[]
}

export function ApprovalQueuePanel() {
  const [items, setItems] = useState<ApprovalItem[]>([])
  const [loading, setLoading] = useState(false)
  const [actingId, setActingId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/research/paperscool/approvals?limit=20", { cache: "no-store" })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const payload = (await res.json()) as ApprovalQueueResponse
      setItems(payload.items || [])
    } catch (e) {
      setError(getErrorMessage(e))
      setItems([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load().catch(() => {})
  }, [load])

  async function decide(sessionId: string, action: "approve" | "reject") {
    setActingId(sessionId)
    setError(null)
    try {
      const body = action === "reject" ? { reason: "Rejected from UI queue" } : {}
      const res = await fetch(`/api/research/paperscool/sessions/${encodeURIComponent(sessionId)}/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `${res.status} ${res.statusText}`)
      }
      await load()
    } catch (e) {
      setError(getErrorMessage(e))
    } finally {
      setActingId(null)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Pending Approvals</CardTitle>
        <CardDescription>Manual approve/reject queue for gated enrichment sessions.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex justify-end">
          <Button size="sm" variant="outline" onClick={() => load()} disabled={loading}>
            {loading ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="mr-1 h-3.5 w-3.5" />}
            Refresh
          </Button>
        </div>

        {error && <p className="text-xs text-destructive">{error}</p>}

        {!items.length ? (
          <p className="text-xs text-muted-foreground">No pending approvals.</p>
        ) : (
          items.map((item) => (
            <div key={item.session_id} className="rounded border p-2 text-xs">
              <p className="font-medium truncate">{item.title || "DailyPaper Session"}</p>
              <p className="text-muted-foreground">session: {item.session_id}</p>
              <p className="text-muted-foreground">queries: {item.query_count || 0} · unique: {item.unique_items || 0}</p>
              <div className="mt-2 flex gap-1">
                <Button
                  size="sm"
                  className="h-7"
                  disabled={actingId === item.session_id}
                  onClick={() => decide(item.session_id, "approve")}
                >
                  {actingId === item.session_id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
                  Approve
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  className="h-7"
                  disabled={actingId === item.session_id}
                  onClick={() => decide(item.session_id, "reject")}
                >
                  <X className="h-3.5 w-3.5" />
                  Reject
                </Button>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}
