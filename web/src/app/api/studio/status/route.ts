export const runtime = "nodejs"

function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function GET() {
  try {
    const upstream = await fetch(`${apiBaseUrl()}/api/studio/status`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    const data = await upstream.json()

    return Response.json(data, {
      status: upstream.status,
    })
  } catch (error) {
    return Response.json(
      {
        claude_cli: false,
        error: "Failed to check Claude CLI status",
        fallback: "anthropic_api"
      },
      { status: 500 }
    )
  }
}
