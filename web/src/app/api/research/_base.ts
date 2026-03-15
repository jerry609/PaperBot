import {
  apiBaseUrl,
  proxyJson as proxyBackendJson,
  type ProxyMethod,
} from "../_utils/backend-proxy"

export { apiBaseUrl }

export async function proxyJson(
  req: Request,
  upstreamUrl: string,
  method: ProxyMethod,
) {
  return proxyBackendJson(req, upstreamUrl, method, { auth: true })
}
