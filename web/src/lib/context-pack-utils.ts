import type { ReproContextPack } from "@/lib/types/p2c"

export function isReproContextPack(payload: unknown): payload is ReproContextPack {
  if (!payload || typeof payload !== "object") return false
  const value = payload as Partial<ReproContextPack> & {
    paper?: { paper_id?: unknown }
    confidence?: { overall?: unknown }
  }
  return (
    typeof value.context_pack_id === "string" &&
    typeof value.version === "string" &&
    typeof value.created_at === "string" &&
    typeof value.objective === "string" &&
    typeof value.paper_type === "string" &&
    Array.isArray(value.observations) &&
    Array.isArray(value.task_roadmap) &&
    Array.isArray(value.warnings) &&
    typeof value.paper?.paper_id === "string" &&
    typeof value.confidence?.overall === "number"
  )
}

export function normalizePack(payload: unknown): ReproContextPack | null {
  if (!payload || typeof payload !== "object") return null
  const raw = payload as Record<string, unknown>
  if (raw.pack && typeof raw.pack === "object") {
    const pack = raw.pack as Record<string, unknown>
    const maybePack: Record<string, unknown> =
      !pack.context_pack_id && typeof raw.context_pack_id === "string"
        ? { ...pack, context_pack_id: raw.context_pack_id }
        : pack
    return isReproContextPack(maybePack) ? maybePack : null
  }
  return isReproContextPack(raw) ? raw : null
}
