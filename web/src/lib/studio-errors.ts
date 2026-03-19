export async function readStudioErrorDetail(
  res: Response,
  fallback: string,
): Promise<string> {
  const text = (await res.text()).trim()
  if (!text) return fallback

  try {
    const payload = JSON.parse(text) as {
      detail?: unknown
      error?: unknown
      message?: unknown
    }

    for (const candidate of [payload.detail, payload.message, payload.error]) {
      if (typeof candidate === "string" && candidate.trim().length > 0) {
        return candidate.trim()
      }
    }
  } catch {
    return text || fallback
  }

  return fallback
}

export function normalizeStudioTransportError(
  message: string | null | undefined,
): string | null {
  const text = typeof message === "string" ? message.trim() : ""
  if (!text) return null

  const normalized = text.toLowerCase()

  if (
    normalized.includes("studio backend timed out") ||
    normalized.includes("upstream api timed out")
  ) {
    return "Studio backend timed out. Check that the backend is healthy and retry."
  }

  if (
    normalized.includes("studio backend is unreachable") ||
    normalized.includes("upstream api unreachable") ||
    normalized.includes("failed to fetch") ||
    normalized.includes("network error") ||
    normalized.includes("load failed") ||
    normalized.includes("unable to connect to service")
  ) {
    return "Studio backend is unreachable. Check that the backend is running and retry."
  }

  return null
}

export function presentStudioError(
  message: string | null | undefined,
  fallback: string,
): string {
  const text = typeof message === "string" ? message.trim() : ""
  return normalizeStudioTransportError(text) ?? (text || fallback)
}
