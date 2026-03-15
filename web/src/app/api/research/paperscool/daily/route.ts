export const runtime = "nodejs"

import { apiBaseUrl, proxyStream } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyStream(req, `${apiBaseUrl()}/api/research/paperscool/daily`, "POST", {
    accept: "text/event-stream, application/json",
    auth: true,
    passthroughNonStreamResponse: true,
    responseContentType: "text/event-stream",
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
