export const runtime = "nodejs"

function apiBaseUrl() {
  return process.env.PAPERBOT_API_BASE_URL || "http://127.0.0.1:8000"
}

export async function GET() {
  try {
    const upstream = await fetch(`${apiBaseUrl()}/api/studio/cwd`, {
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
    // Return a sensible default if backend is unavailable
    return Response.json(
      {
        cwd: process.env.HOME || "/tmp",
        source: "fallback",
        error: "Failed to get working directory from backend",
      },
      { status: 200 }
    )
  }
}
