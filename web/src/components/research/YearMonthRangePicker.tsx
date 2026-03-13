"use client"

import { useMemo } from "react"
import { CalendarRange } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
} from "@/components/ui/dropdown-menu"

export type DatePreset = "all" | "last12" | "last3" | "last5" | "custom"

interface YearRangePickerProps {
  preset: DatePreset
  onPresetChange: (next: DatePreset) => void

  yearFrom: string
  yearTo: string

  onYearFromChange: (value: string) => void
  onYearToChange: (value: string) => void

  disabled?: boolean
  className?: string
}

export function YearMonthRangePicker({
  preset,
  onPresetChange,
  yearFrom,
  yearTo,
  onYearFromChange,
  onYearToChange,
  disabled = false,
  className,
}: YearRangePickerProps) {
  const currentYear = new Date().getFullYear()

  const summaryLabel = useMemo(() => {
    if (!yearFrom.trim() && !yearTo.trim()) {
      if (preset === "last12") return "Last 12 months"
      if (preset === "last3") return "Last 3 years"
      if (preset === "last5") return "Last 5 years"
      return "All time"
    }
    const from = yearFrom || "…"
    const to = yearTo || "…"
    return `${from} → ${to}`
  }, [preset, yearFrom, yearTo])

  const handleYearInputChange = (raw: string, kind: "from" | "to") => {
    const trimmed = raw.trim()
    if (!trimmed) {
      if (kind === "from") onYearFromChange("")
      else onYearToChange("")
      onPresetChange("custom")
      return
    }
    const n = Number(trimmed)
    if (!Number.isInteger(n)) return
    if (n < 1900 || n > currentYear) return
    if (kind === "from") onYearFromChange(String(n))
    else onYearToChange(String(n))
    onPresetChange("custom")
  }

  const applyPreset = (next: DatePreset) => {
    const year = new Date().getFullYear()
    if (next === "all") {
      onYearFromChange("")
      onYearToChange("")
    } else if (next === "last12") {
      onYearFromChange(String(year - 1))
      onYearToChange(String(year))
    } else if (next === "last3") {
      onYearFromChange(String(year - 2))
      onYearToChange(String(year))
    } else if (next === "last5") {
      onYearFromChange(String(year - 4))
      onYearToChange(String(year))
    }
    onPresetChange(next)
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild disabled={disabled}>
        <Button
          variant="outline"
          size="sm"
          className={cn(
            "h-8 text-xs justify-start gap-2 w-[220px]",
            !yearFrom && !yearTo && preset === "all" && "text-muted-foreground",
            className,
          )}
        >
          <CalendarRange className="h-3.5 w-3.5" />
          <span className="truncate text-left">{summaryLabel}</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-[320px] p-3" align="start">
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Presets</span>
          <div className="flex flex-wrap gap-1">
            <PresetChip label="All time" active={preset === "all"} onClick={() => applyPreset("all")} />
            <PresetChip label="Last 12m" active={preset === "last12"} onClick={() => applyPreset("last12")} />
            <PresetChip label="Last 3y" active={preset === "last3"} onClick={() => applyPreset("last3")} />
            <PresetChip label="Last 5y" active={preset === "last5"} onClick={() => applyPreset("last5")} />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <YearColumn
            title="From"
            year={yearFrom}
            onYearChange={(v) => handleYearInputChange(v, "from")}
            currentYear={currentYear}
          />
          <YearColumn
            title="To"
            year={yearTo}
            onYearChange={(v) => handleYearInputChange(v, "to")}
            currentYear={currentYear}
          />
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function YearColumn({ title, year, onYearChange, currentYear }: { title: string; year: string; onYearChange: (v: string) => void; currentYear: number }) {
  return (
    <div className="space-y-1.5">
      <div className="text-xs font-medium text-muted-foreground">{title}</div>
      <div className="flex items-center gap-1.5">
        <Input
          type="number"
          value={year}
          onChange={(e) => onYearChange(e.target.value)}
          placeholder="YYYY"
          aria-label={`${title} year`}
          className="h-7 w-[90px] border bg-background px-1 text-xs shadow-none [appearance:textfield] focus-visible:ring-0"
          min={1900}
          max={currentYear}
        />
      </div>
    </div>
  )
}

function PresetChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "h-6 rounded-full border px-2 text-[10px]",
        active
          ? "border-primary bg-primary text-primary-foreground"
          : "border-muted bg-muted/60 text-muted-foreground hover:bg-muted",
      )}
    >
      {label}
    </button>
  )
}
