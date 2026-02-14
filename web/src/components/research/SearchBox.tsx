"use client"

import { KeyboardEvent, useRef } from "react"
import { Brain, CalendarRange, Loader2, Search } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"

import { TrackSelector, type Track } from "./TrackSelector"

interface SearchBoxProps {
  query: string
  onQueryChange: (query: string) => void
  onSearch: () => void
  tracks: Track[]
  activeTrack: Track | null
  onSelectTrack: (trackId: number) => void
  onNewTrack: () => void
  onManageTracks: () => void
  isSearching?: boolean
  disabled?: boolean
  placeholder?: string
  className?: string
  anchorMode?: "personalized" | "global"
  onAnchorModeChange?: (mode: "personalized" | "global") => void
  onOpenMemory?: () => void
  yearFrom?: string
  yearTo?: string
  onYearFromChange?: (value: string) => void
  onYearToChange?: (value: string) => void
}

export function SearchBox({
  query,
  onQueryChange,
  onSearch,
  tracks,
  activeTrack,
  onSelectTrack,
  onNewTrack,
  onManageTracks,
  isSearching = false,
  disabled = false,
  placeholder = "Search for papers on RAG, transformers, security...",
  className,
  anchorMode = "personalized",
  onAnchorModeChange,
  onOpenMemory,
  yearFrom = "",
  yearTo = "",
  onYearFromChange,
  onYearToChange,
}: SearchBoxProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (query.trim() && !isSearching && !disabled) {
        onSearch()
      }
    }
  }

  const handleSubmit = () => {
    if (query.trim() && !isSearching && !disabled) {
      onSearch()
    }
  }

  const hasYearFilter = !!(yearFrom.trim() || yearTo.trim())

  return (
    <div
      className={cn(
        "w-full",
        className
      )}
    >
      <div
        className={cn(
          "relative rounded-2xl border bg-background shadow-md transition-all duration-200",
          "focus-within:shadow-lg focus-within:border-primary/50",
          "hover:shadow-lg",
          disabled && "opacity-60"
        )}
      >
        {/* Search Input */}
        <Textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isSearching}
          className={cn(
            "min-h-[56px] max-h-[200px] resize-none border-0 bg-transparent",
            "px-5 sm:px-6 pt-4 pb-[56px]",
            "text-base placeholder:text-muted-foreground/50",
            "focus-visible:ring-0 focus-visible:ring-offset-0"
          )}
          rows={1}
        />

        {/* Bottom Toolbar */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 sm:px-5 py-2.5 sm:py-3">
          {/* Left side - mode + year range */}
          <div className="flex items-center gap-1.5 sm:gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs"
              onClick={() =>
                onAnchorModeChange?.(anchorMode === "personalized" ? "global" : "personalized")
              }
              disabled={disabled || isSearching}
            >
              {anchorMode === "personalized" ? "Personalized" : "Global"}
            </Button>

            <div className="flex h-8 items-center gap-1 rounded-md border bg-muted/30 px-2">
              <CalendarRange className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="hidden text-[10px] font-medium uppercase tracking-wide text-muted-foreground md:inline">
                Year
              </span>
              <Input
                type="number"
                value={yearFrom}
                onChange={(e) => onYearFromChange?.(e.target.value)}
                placeholder="YYYY"
                aria-label="Year from"
                className="h-6 w-14 border-0 bg-transparent px-1 text-xs shadow-none [appearance:textfield] focus-visible:ring-0"
                min={1900}
                max={2100}
                disabled={disabled || isSearching}
              />
              <span className="text-muted-foreground">-</span>
              <Input
                type="number"
                value={yearTo}
                onChange={(e) => onYearToChange?.(e.target.value)}
                placeholder="YYYY"
                aria-label="Year to"
                className="h-6 w-14 border-0 bg-transparent px-1 text-xs shadow-none [appearance:textfield] focus-visible:ring-0"
                min={1900}
                max={2100}
                disabled={disabled || isSearching}
              />
            </div>

            {hasYearFilter && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 px-2 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => {
                  onYearFromChange?.("")
                  onYearToChange?.("")
                }}
                disabled={disabled || isSearching}
              >
                All years
              </Button>
            )}
          </div>

          {/* Right side - Memory, Track selector and Search button */}
          <div className="flex items-center gap-1.5 sm:gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onOpenMemory}
              disabled={disabled || isSearching}
              title="Track Memory"
            >
              <Brain className="h-4 w-4" />
            </Button>

            <TrackSelector
              tracks={tracks}
              activeTrack={activeTrack}
              onSelectTrack={onSelectTrack}
              onNewTrack={onNewTrack}
              onManageTracks={onManageTracks}
              disabled={disabled || isSearching}
            />

            <Button
              size="icon"
              className="h-9 w-9 sm:h-8 sm:w-8 rounded-lg"
              onClick={handleSubmit}
              aria-label={isSearching ? "Searching" : "Search"}
              disabled={disabled || isSearching || !query.trim()}
            >
              {isSearching ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
