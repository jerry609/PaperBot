export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

interface Context {
  params: Promise<{ collectionId: string }>
}

export async function PATCH(req: Request, context: Context) {
  const { collectionId } = await context.params
  return proxyJson(req, `${apiBaseUrl()}/api/research/collections/${encodeURIComponent(collectionId)}`, "PATCH")
}
