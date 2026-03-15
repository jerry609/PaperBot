export const runtime = "nodejs"

import { apiBaseUrl, proxyStream } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyStream(req, `${apiBaseUrl()}/api/gen-code`, "POST", {
    auth: true,
    responseContentType: "text/event-stream",
  })
}
