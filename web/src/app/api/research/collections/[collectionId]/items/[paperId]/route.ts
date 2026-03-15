export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

interface Context {
  params: Promise<{ collectionId: string; paperId: string }>
}

export async function PATCH(req: Request, context: Context) {
  const { collectionId, paperId } = await context.params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/collections/${encodeURIComponent(collectionId)}/items/${encodeURIComponent(paperId)}`,
    "PATCH",
  )
}

export async function DELETE(req: Request, context: Context) {
  const { collectionId, paperId } = await context.params
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/collections/${encodeURIComponent(collectionId)}/items/${encodeURIComponent(paperId)}?${url.searchParams.toString()}`,
    "DELETE",
  )
}
