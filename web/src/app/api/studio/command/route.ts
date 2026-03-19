export const runtime = "nodejs"

import { apiBaseUrl, proxyJson } from "@/app/api/_utils/backend-proxy"

export async function POST(req: Request) {
  return proxyJson(req, `${apiBaseUrl()}/api/studio/command`, "POST", {
    auth: true,
    onError: () =>
      Response.json(
        {
          ok: false,
          command: [],
          returncode: 500,
          stdout: "",
          stderr: "Failed to run Studio command",
        },
        { status: 500 },
      ),
  })
}
