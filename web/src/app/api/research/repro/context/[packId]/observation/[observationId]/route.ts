import { apiBaseUrl, proxyJson } from "../../../../../_base"

export const runtime = "nodejs"

export async function GET(
  req: Request,
  { params }: { params: { packId: string; observationId: string } }
) {
  const { packId, observationId } = params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/repro/context/${encodeURIComponent(packId)}/observation/${encodeURIComponent(observationId)}`,
    "GET"
  )
}
