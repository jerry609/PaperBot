export const runtime = "nodejs"

import { apiBaseUrl, proxyStream } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyStream(req, `${apiBaseUrl()}/api/studio/chat`, "POST", {
    auth: true,
    responseContentType: "text/event-stream",
  })
}
