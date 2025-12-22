export const runtime = "nodejs"

function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function GET(_req: Request, ctx: { params: Promise<{ snapshotId: string }> }) {
  const { snapshotId } = await ctx.params
  const upstream = await fetch(`${apiBaseUrl()}/api/runbook/snapshots/${encodeURIComponent(snapshotId)}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  })
  const text = await upstream.text()
  return new Response(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json",
      "Cache-Control": "no-cache",
    },
  })
}

