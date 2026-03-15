import { apiBaseUrl, proxyText } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request, ctx: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = await ctx.params
  return proxyText(
    req,
    `${apiBaseUrl()}/api/research/paperscool/sessions/${encodeURIComponent(sessionId)}`,
    "GET",
    {
      cache: "no-store",
      responseContentType: "application/json",
    },
  )
}
