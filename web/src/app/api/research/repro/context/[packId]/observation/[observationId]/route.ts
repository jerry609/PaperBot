import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"
import type { NextRequest } from "next/server"

export const runtime = "nodejs"

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ packId: string; observationId: string }> }
) {
  const { packId, observationId } = await params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/repro/context/${encodeURIComponent(packId)}/observation/${encodeURIComponent(observationId)}`,
    "GET"
  )
}
