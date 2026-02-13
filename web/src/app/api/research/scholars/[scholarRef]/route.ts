export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "../../_base"

export async function PATCH(req: Request, context: { params: Promise<{ scholarRef: string }> }) {
  const params = await context.params
  const scholarRef = encodeURIComponent(params.scholarRef)
  return proxyJson(req, `${apiBaseUrl()}/api/research/scholars/${scholarRef}`, "PATCH")
}

export async function DELETE(req: Request, context: { params: Promise<{ scholarRef: string }> }) {
  const params = await context.params
  const scholarRef = encodeURIComponent(params.scholarRef)
  return proxyJson(req, `${apiBaseUrl()}/api/research/scholars/${scholarRef}`, "DELETE")
}
