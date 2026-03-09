export type UpstreamErrorBody = {
  detail?: string
  error?: string
}

export function toFriendlyErrorMessage(status: number, rawText: string): string | null {
  if (!rawText) return null

  let parsed: UpstreamErrorBody | null = null
  try {
    parsed = JSON.parse(rawText) as UpstreamErrorBody
  } catch {
    parsed = null
  }

  const detail = parsed && typeof parsed.detail === "string" ? parsed.detail : undefined

  if (detail && (detail.includes("Upstream API unreachable") || detail.includes("Upstream API timed out"))) {
    if (detail.includes("timed out")) {
      return "Unable to connect to service (request timed out). Please ensure the backend is running. Please try again."
    }
    return "Unable to connect to service. Please ensure the backend is running."
  }

  // Fallback: surface backend-provided detail when available
  if (detail) return detail

  return null
}

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    const friendly = toFriendlyErrorMessage(res.status, text)
    if (friendly) {
      throw new Error(friendly)
    }
    const statusLabel = `${res.status} ${res.statusText}`.trim()
    const base = statusLabel || "Request failed"
    const message = text ? `${base} ${text}`.trim() : base
    throw new Error(message)
  }
  return res.json() as Promise<T>
}

export function getErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}
