import { apiBaseUrl, proxyJson } from "../../../../_base"

export async function POST(req: Request, { params }: { params: { packId: string } }) {
  const packId = encodeURIComponent(params.packId)
  return proxyJson(req, `${apiBaseUrl()}/api/research/repro/context/${packId}/session`, "POST")
}
