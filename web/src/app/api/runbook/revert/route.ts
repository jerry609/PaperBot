export const runtime = "nodejs"

function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function POST(req: Request) {
  const body = await req.text()
  const upstream = await fetch(`${apiBaseUrl()}/api/runbook/revert`, {
    method: "POST",
    headers: {
      "Content-Type": req.headers.get("content-type") || "application/json",
      Accept: "application/json",
    },
    body,
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

