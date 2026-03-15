export const runtime = "nodejs"

import { apiBaseUrl, proxyStream } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyStream(req, `${apiBaseUrl()}/api/research/paperscool/analyze`, "POST", {
    accept: "text/event-stream",
    responseContentType: "text/event-stream",
  })
}
