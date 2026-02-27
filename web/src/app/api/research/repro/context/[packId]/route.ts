import { apiBaseUrl, proxyJson } from "../../../_base"

export async function GET(req: Request, { params }: { params: { packId: string } }) {
  const packId = encodeURIComponent(params.packId)
  return proxyJson(req, `${apiBaseUrl()}/api/research/repro/context/${packId}`, "GET")
}

export async function DELETE(req: Request, { params }: { params: { packId: string } }) {
  const packId = encodeURIComponent(params.packId)
  return proxyJson(req, `${apiBaseUrl()}/api/research/repro/context/${packId}`, "DELETE")
}
