import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function safeHref(url?: string | null) {
  if (!url) return null
  const trimmed = url.trim()
  if (!trimmed) return null
  if (trimmed.startsWith("//")) return null
  if (/[\u0000-\u001F\u007F]/.test(trimmed)) return null
  const hasScheme = /^[a-zA-Z][a-zA-Z\d+.-]*:/.test(trimmed)
  if (!hasScheme) return trimmed
  try {
    const parsed = new URL(trimmed)
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? trimmed : null
  } catch {
    return null
  }
}

export function safeInternalHref(url?: string | null) {
  const safe = safeHref(url)
  if (!safe) return null
  if (safe.startsWith("/") || safe.startsWith("?") || safe.startsWith("#")) return safe
  return null
}


export function mergeTracksStable<T extends { id: number }>(prev: T[], next: T[]): T[] {
  if (!prev.length) return next.slice()
  const indexMap = new Map(prev.map((item, index) => [item.id, index]))
  return [...next].sort((a, b) => {
    const ia = indexMap.has(a.id) ? indexMap.get(a.id)! : Number.MAX_SAFE_INTEGER
    const ib = indexMap.has(b.id) ? indexMap.get(b.id)! : Number.MAX_SAFE_INTEGER
    if (ia !== ib) return ia - ib
    return a.id - b.id
  })
}
