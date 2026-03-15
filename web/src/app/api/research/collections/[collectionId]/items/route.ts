export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/auth-json-proxy"

interface Context {
  params: Promise<{ collectionId: string }>
}

export async function GET(req: Request, context: Context) {
  const { collectionId } = await context.params
  const url = new URL(req.url)
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/collections/${encodeURIComponent(collectionId)}/items?${url.searchParams.toString()}`,
    "GET",
  )
}

export async function POST(req: Request, context: Context) {
  const { collectionId } = await context.params
  return proxyJson(
    req,
    `${apiBaseUrl()}/api/research/collections/${encodeURIComponent(collectionId)}/items`,
    "POST",
  )
}
