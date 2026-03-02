import type { NextRequest } from "next/server"

import { apiBaseUrl, proxyJson } from "../../../../_base"

export async function POST(req: NextRequest, { params }: { params: Promise<{ packId: string }> }) {
  const { packId } = await params
  return proxyJson(req, `${apiBaseUrl()}/api/research/repro/context/${encodeURIComponent(packId)}/session`, "POST")
}
