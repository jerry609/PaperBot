export const runtime = "nodejs"

import { apiBaseUrl, proxyBinary } from "@/app/api/_utils/backend-proxy"

export async function GET(req: Request) {
  const url = new URL(req.url)
  const upstream = `${apiBaseUrl()}/api/research/papers/export?${url.searchParams.toString()}`

  return proxyBinary(req, upstream, "GET", {
    accept: "*/*",
    auth: true,
    onError: ({ error }) =>
      Response.json(
        {
          detail: "Upstream API unreachable",
          error: error instanceof Error ? error.message : String(error),
        },
        { status: 502 },
      ),
  })
}
