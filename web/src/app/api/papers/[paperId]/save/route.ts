import { NextResponse } from "next/server"
import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

// Validate paperId to prevent path traversal attacks
function validatePaperId(paperId: string): number | null {
  const parsed = parseInt(paperId, 10)
  if (isNaN(parsed) || parsed <= 0 || String(parsed) !== paperId) {
    return null
  }
  return parsed
}

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ paperId: string }> }
) {
  const { paperId } = await params
  const validId = validatePaperId(paperId)
  if (validId === null) {
    return NextResponse.json({ error: "Invalid paper ID" }, { status: 400 })
  }
  const url = new URL(req.url)
  const upstream = `${apiBaseUrl()}/api/papers/${validId}/save${url.search}`
  return proxyJson(req, upstream, "DELETE")
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ paperId: string }> }
) {
  const { paperId } = await params
  const validId = validatePaperId(paperId)
  if (validId === null) {
    return NextResponse.json({ error: "Invalid paper ID" }, { status: 400 })
  }
  const upstream = `${apiBaseUrl()}/api/papers/${validId}/save`
  return proxyJson(req, upstream, "POST")
}
