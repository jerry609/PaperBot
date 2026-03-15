import { apiBaseUrl, proxyText } from "@/app/api/_utils/backend-proxy"

export const runtime = "nodejs"

export async function GET(req: Request, ctx: { params: Promise<{ snapshotId: string }> }) {
  const { snapshotId } = await ctx.params
  return proxyText(req, `${apiBaseUrl()}/api/runbook/snapshots/${encodeURIComponent(snapshotId)}`, "GET")
}
