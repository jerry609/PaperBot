export const runtime = "nodejs"

function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function GET(req: Request) {
  const url = new URL(req.url)
  const upstream = await fetch(`${apiBaseUrl()}/api/runbook/changes?${url.searchParams.toString()}`, {
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

