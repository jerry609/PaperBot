"use client"

import { useEffect, useState } from "react"
import { Loader2, RefreshCw } from "lucide-react"

import { Button } from "@/components/ui/button"
import { getErrorMessage } from "@/lib/fetch"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

type MemoryItem = {
  id: number
  kind?: string
  content?: string
  tags?: string[]
  created_at?: string | null
}

type MemoryResponse = {
  user_id: string
  items: MemoryItem[]
}

interface MemoryTabProps {
  userId: string
  trackId: number | null
}

export function MemoryTab({ userId, trackId }: MemoryTabProps) {
  const [items, setItems] = useState<MemoryItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    if (!trackId) {
      setItems([])
      return
    }
    setLoading(true)
    setError(null)
    try {
      const qs = new URLSearchParams({
        user_id: userId,
        track_id: String(trackId),
        limit: "100",
      })
      const res = await fetch(`/api/research/memory/inbox?${qs.toString()}`)
      if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}`)
      }
      const payload = (await res.json()) as MemoryResponse
      setItems(payload.items || [])
    } catch (e) {
      setError(getErrorMessage(e))
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load().catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, trackId])

  if (!trackId) {
    return <div className="py-8 text-sm text-muted-foreground">Select a track to view memory items.</div>
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Memory inbox for current track.</p>
        <Button variant="outline" size="sm" onClick={() => load().catch(() => {})} disabled={loading}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Refresh
        </Button>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {loading && !items.length ? (
        <div className="py-8 text-sm text-muted-foreground">Loading memory items...</div>
      ) : !items.length ? (
        <div className="py-8 text-sm text-muted-foreground">No memory items for this track yet.</div>
      ) : (
        <div className="space-y-2">
          {items.map((item) => (
            <Card key={item.id}>
              <CardContent className="py-3 space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    {item.kind || "note"}
                  </Badge>
                  {item.created_at && (
                    <span className="text-xs text-muted-foreground">{new Date(item.created_at).toLocaleString()}</span>
                  )}
                </div>
                <p className="text-sm leading-relaxed">{item.content || "(empty)"}</p>
                {!!item.tags?.length && (
                  <div className="flex flex-wrap gap-1.5">
                    {item.tags.map((tag) => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
