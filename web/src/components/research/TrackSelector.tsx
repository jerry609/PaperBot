"use client"

import { Check, ChevronDown, Plus, Settings } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export type Track = {
  id: number
  name: string
  description?: string
  keywords?: string[]
  venues?: string[]
  methods?: string[]
  is_active?: boolean
}

interface TrackSelectorProps {
  tracks: Track[]
  activeTrack: Track | null
  onSelectTrack: (trackId: number) => void
  onNewTrack: () => void
  onManageTracks: () => void
  disabled?: boolean
}

export function TrackSelector({
  tracks,
  activeTrack,
  onSelectTrack,
  onNewTrack,
  onManageTracks,
  disabled = false,
}: TrackSelectorProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 gap-1 px-2 text-sm font-medium text-muted-foreground hover:text-foreground"
          disabled={disabled}
        >
          {activeTrack?.name || "Select Track"}
          <ChevronDown className="h-3.5 w-3.5" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        {tracks.length === 0 ? (
          <div className="px-2 py-1.5 text-sm text-muted-foreground">
            No tracks yet
          </div>
        ) : (
          tracks.map((track) => (
            <DropdownMenuItem
              key={track.id}
              onClick={() => {
                if (!activeTrack || track.id !== activeTrack.id) {
                  onSelectTrack(track.id)
                }
              }}
              className="flex items-center justify-between"
            >
              <span className="truncate">{track.name}</span>
              {track.id === activeTrack?.id && (
                <Check className="h-4 w-4 text-primary" />
              )}
            </DropdownMenuItem>
          ))
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onNewTrack}>
          <Plus className="mr-2 h-4 w-4" />
          New Track
        </DropdownMenuItem>
        <DropdownMenuItem onClick={onManageTracks}>
          <Settings className="mr-2 h-4 w-4" />
          Manage Tracks
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
