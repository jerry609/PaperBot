"use client"

import { KeyboardEvent, useRef, useState } from "react"
import { Brain, CalendarRange, Check, Loader2, Search } from "lucide-react"

import { cn } from "@/lib/utils"
import { showTrackMemoryButton } from "@/config/features"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Popover, PopoverContent, PopoverAnchor } from "@/components/ui/popover"

const YEAR_MIN = 1990
const YEAR_MAX = new Date().getFullYear()
const YEAR_OPTIONS = Array.from({ length: YEAR_MAX - YEAR_MIN + 1 }, (_, i) =>
  String(YEAR_MAX - i)
)

interface YearComboboxProps {
  value: string
  onChange: (v: string) => void
  placeholder: string
  options: string[]
  disabled?: boolean
}

function YearCombobox({ value, onChange, placeholder, options, disabled }: YearComboboxProps) {
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState(value)

  const filtered = input ? options.filter((y) => y.startsWith(input)) : options

  const commit = (year: string) => {
    onChange(year)
    setInput(year)
    setOpen(false)
  }

  const handleBlur = () => {
    setTimeout(() => {
      const num = Number(input)
      if (input && num >= YEAR_MIN && num <= YEAR_MAX) onChange(input)
      else setInput(value)
      setOpen(false)
    }, 150)
  }

  const handleOpen = (next: boolean) => {
    if (next) setInput(value)
    setOpen(next)
  }

  return (
    <Popover open={open} onOpenChange={handleOpen}>
      <PopoverAnchor asChild>
        <input
          type="text"
          inputMode="numeric"
          maxLength={4}
          value={open ? input : (value || "")}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(e) => { setInput(e.target.value.replace(/\D/g, "")); setOpen(true) }}
          onFocus={() => { setInput(value); setOpen(true) }}
          onBlur={handleBlur}
          onKeyDown={(e) => {
            if (e.key === "Enter") { const n = Number(input); if (input && n >= YEAR_MIN && n <= YEAR_MAX) commit(input) }
            if (e.key === "Escape") { setInput(value); setOpen(false) }
          }}
          className={cn(
            "h-6 w-12 bg-transparent px-1 text-xs outline-none",
            "placeholder:text-muted-foreground/60",
            disabled && "cursor-not-allowed opacity-50"
          )}
        />
      </PopoverAnchor>
      <PopoverContent className="w-24 p-1" align="start" onOpenAutoFocus={(e) => e.preventDefault()}>
        {filtered.length === 0 ? (
          <p className="py-1 text-center text-xs text-muted-foreground">No results</p>
        ) : (
          <ul className="max-h-48 overflow-y-auto">
            {filtered.map((y) => (
              <li
                key={y}
                onMouseDown={() => commit(y)}
                className={cn(
                  "flex cursor-pointer items-center rounded px-2 py-1 text-xs hover:bg-accent hover:text-accent-foreground",
                  y === value && "font-medium"
                )}
              >
                <Check className={cn("mr-1 h-3 w-3 shrink-0", y === value ? "opacity-100" : "opacity-0")} />
                {y}
              </li>
            ))}
          </ul>
        )}
      </PopoverContent>
    </Popover>
  )
}

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

            <div className="flex h-8 items-center gap-0.5 rounded-md border bg-muted/30 px-1.5">
              <CalendarRange className="h-3.5 w-3.5 text-muted-foreground" />
              <YearCombobox
                value={yearFrom}
                onChange={(v) => onYearFromChange?.(v)}
                placeholder="From"
                options={YEAR_OPTIONS.filter((y) => !yearTo || Number(y) <= Number(yearTo))}
                disabled={disabled || isSearching}
              />
              <span className="text-muted-foreground">–</span>
              <YearCombobox
                value={yearTo}
                onChange={(v) => onYearToChange?.(v)}
                placeholder="To"
                options={YEAR_OPTIONS.filter((y) => !yearFrom || Number(y) >= Number(yearFrom))}
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
            {onOpenMemory && showTrackMemoryButton() && (
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
            )}

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
