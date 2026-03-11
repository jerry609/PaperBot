"use client"

import {
  BarChart3,
  BookOpen,
  Bot,
  FlaskConical,
  MessageSquare,
  Plus,
  ScanEye,
  ShieldCheck,
  type LucideIcon,
} from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

import type { Track } from "./TrackSelector"

interface TrackPillsProps {
  tracks: Track[]
  activeTrackId: number | null
  onSelectTrack: (trackId: number) => void
  onNewTrack: () => void
  disabled?: boolean
  maxVisible?: number
}

const trackIcons: Record<string, LucideIcon> = {
  RAG: BarChart3,
  LLM: Bot,
  "ML Security": FlaskConical,
  Security: ShieldCheck,
  CV: ScanEye,
  NLP: MessageSquare,
}

const defaultIcon = BookOpen

function getTrackIcon(name: string): LucideIcon {
  return trackIcons[name] || defaultIcon
}

export function TrackPills({
  tracks,
  activeTrackId,
  onSelectTrack,
  onNewTrack,
  disabled = false,
  maxVisible = 5,
}: TrackPillsProps) {
  const visibleTracks = tracks.slice(0, maxVisible)
  const hasMore = tracks.length > maxVisible

  return (
    <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3">
      {visibleTracks.map((track) => (
        <Button
          key={track.id}
          variant={track.id === activeTrackId ? "default" : "outline"}
          className={cn(
            "h-9 sm:h-10 gap-1.5 sm:gap-2 rounded-full px-3 sm:px-5 text-sm sm:text-base transition-all",
            track.id === activeTrackId
              ? "bg-primary text-primary-foreground shadow-sm"
              : "bg-background hover:bg-accent"
          )}
          onClick={() => {
            if (track.id !== activeTrackId) onSelectTrack(track.id)
          }}
          disabled={disabled}
        >
          {(() => {
            const Icon = getTrackIcon(track.name)
            return <Icon className="h-4 w-4" />
          })()}
          <span className="truncate max-w-[100px] sm:max-w-none">{track.name}</span>
        </Button>
      ))}

      {hasMore && (
        <span className="text-sm sm:text-base text-muted-foreground">
          +{tracks.length - maxVisible} more
        </span>
      )}

      <Button
        variant="outline"
        className="h-9 sm:h-10 gap-1.5 sm:gap-2 rounded-full px-3 sm:px-5 text-sm sm:text-base border-dashed"
        onClick={onNewTrack}
        disabled={disabled}
      >
        <Plus className="h-4 w-4" />
        <span>New Track</span>
      </Button>
    </div>
  )
}
