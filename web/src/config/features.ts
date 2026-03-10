// Small feature flag helpers for UI toggles

function parseBool(v?: string | null): boolean {
  if (!v) return false
  const s = String(v).trim().toLowerCase()
  return s === "1" || s === "true" || s === "yes" || s === "on"
}

// Whether to show the "Open Discovery Workspace" entry in Research page.
// Default: false. Can be overridden by:
// - runtime localStorage key:  paperbot.showDiscoveryLink  ("1"/"true")
// - build-time env: NEXT_PUBLIC_SHOW_DISCOVERY ("1"/"true")
export function showDiscoveryLink(): boolean {
  if (typeof window !== "undefined") {
    const v = window.localStorage.getItem("paperbot.showDiscoveryLink")
    if (v != null) return parseBool(v)
  }
  return parseBool(process.env.NEXT_PUBLIC_SHOW_DISCOVERY)
}

