export const runtime = "nodejs"

import { apiBaseUrl, proxyText } from "@/app/api/_utils/backend-proxy"

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
}

export async function GET(
  req: Request,
  { params }: { params: Promise<{ token: string }> },
) {
  const { token } = await params
  return proxyText(
    req,
    `${apiBaseUrl()}/api/newsletter/unsubscribe/${encodeURIComponent(token)}`,
    "GET",
    {
      accept: "text/html",
      responseContentType: "text/html",
      onError: ({ error }) => {
        const detail = error instanceof Error ? error.message : String(error)
        return new Response(
          `<html><body><h2>Error</h2><p>${escapeHtml(detail)}</p></body></html>`,
          { status: 502, headers: { "Content-Type": "text/html" } },
        )
      },
    },
  )
}
