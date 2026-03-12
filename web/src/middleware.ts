import { auth } from "@/auth"
import { NextResponse } from "next/server"

// Only allow unauthenticated access to explicit auth pages.
// All other paths (including "/") require a valid session.
const PUBLIC_PATHS = ["/login", "/register", "/forgot-password", "/reset-password"]

export default auth((req) => {
  const { pathname } = req.nextUrl

  // Skip all Next internals and API routes
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api/") ||
    pathname === "/favicon.ico"
  ) {
    return NextResponse.next()
  }

  // Public pages
  if (PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"))) {
    return NextResponse.next()
  }

  if (!req.auth) {
    const url = new URL("/login", req.url)
    url.searchParams.set("callbackUrl", req.nextUrl.pathname)
    return NextResponse.redirect(url)
  }

  return NextResponse.next()
})

export const config = {
  matcher: ["/(.*)"],
}
