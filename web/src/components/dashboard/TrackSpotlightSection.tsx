"use client"

import { useCallback, useState } from "react"

import { TrackSpotlight } from "./TrackSpotlight"
import type { AnchorPreviewItem, ResearchTrackSummary, TrackFeedItem } from "@/lib/types"

interface TrackSpotlightSectionProps {
  initialTracks: ResearchTrackSummary[]
  initialActiveTrack: ResearchTrackSummary | null
  initialFeedItems: TrackFeedItem[]
  initialFeedTotal: number
  initialAnchors: AnchorPreviewItem[]
}

export function TrackSpotlightSection({
  initialTracks,
  initialActiveTrack,
  initialFeedItems,
  initialFeedTotal,
  initialAnchors = "default",
}: TrackSpotlightSectionProps) {
  const [tracks] = useState(initialTracks)
  const [activeTrack, setActiveTrack] = useState(initialActiveTrack)
  const [feedItems, setFeedItems] = useState(initialFeedItems)
  const [feedTotal, setFeedTotal] = useState(initialFeedTotal)
  const [anchors, setAnchors] = useState(initialAnchors)
  const [isLoading, setIsLoading] = useState(false)

  const handleSelectTrack = useCallback(
    async (trackId: number) => {
      const selectedTrack = tracks.find((t) => t.id === trackId)
      if (!selectedTrack || selectedTrack.id === activeTrack?.id) return

      setIsLoading(true)
      setActiveTrack(selectedTrack)

      try {
        // Fetch feed and anchors for the new track
        const [feedRes, anchorsRes] = await Promise.all([
          fetch(`/api/research/tracks/${trackId}/feed?limit=6`),
          fetch(`/api/research/tracks/${trackId}/anchors/discover?limit=4`),
        ])

        if (feedRes.ok) {
          const feedData = await feedRes.json()
          setFeedItems(feedData.items || [])
          setFeedTotal(feedData.total || 0)
        } else {
          setFeedItems([])
          setFeedTotal(0)
        }

        if (anchorsRes.ok) {
          const anchorsData = await anchorsRes.json()
          setAnchors(anchorsData.items || [])
        } else {
          setAnchors([])
        }

        // Optionally activate the track on the backend
        await fetch(`/api/research/tracks/${trackId}/activate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        })
      } catch (error) {
        console.error("Failed to fetch track data:", error)
        setFeedItems([])
        setFeedTotal(0)
        setAnchors([])
      } finally {
        setIsLoading(false)
      }
    },
    [tracks, activeTrack?.id]
  )

  return (
    <TrackSpotlight
      tracks={tracks}
      activeTrack={activeTrack}
      onSelectTrack={handleSelectTrack}
      feedItems={feedItems}
      feedTotal={feedTotal}
      anchors={anchors}
      isLoading={isLoading}
    />
  )
}
