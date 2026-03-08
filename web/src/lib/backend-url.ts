/**
 * Resolve the Python backend base URL, bypassing the Next.js rewrite proxy
 * which buffers SSE responses and breaks streaming.
 */
export function backendUrl(path: string): string {
  const base =
    process.env.NEXT_PUBLIC_PAPERBOT_API_URL ||
    (typeof window !== "undefined"
      ? `${window.location.protocol}//${window.location.hostname}:8000`
      : "http://127.0.0.1:8000")
  return `${base}${path}`
}
