"use client"

import { Suspense, useState } from "react"
import { useSearchParams } from "next/navigation"
import Link from "next/link"
import { BookOpen, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export default function ForgotPasswordPage() {
  return (
    <Suspense fallback={<ForgotPasswordContent callbackUrl="/login" />}>
      <ForgotPasswordWithSearchParams />
    </Suspense>
  )
}

function ForgotPasswordWithSearchParams() {
  const searchParams = useSearchParams()
  const callbackUrl = searchParams.get("callbackUrl") || "/login"
  return <ForgotPasswordContent callbackUrl={callbackUrl} />
}

function ForgotPasswordContent({ callbackUrl }: { callbackUrl: string }) {
  const [email, setEmail] = useState("")
  const [loading, setLoading] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const loginHref = callbackUrl.startsWith("/login")
    ? callbackUrl
    : `/login?callbackUrl=${encodeURIComponent(callbackUrl)}`
  const studioReturn = callbackUrl.startsWith("/studio")

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await fetch("/api/auth/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      })
      // Always show success to avoid email enumeration
      setSubmitted(true)
    } catch {
      setError("Something went wrong. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-sm space-y-7">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5" />
          <span className="font-semibold tracking-tight">PaperBot</span>
        </div>

        {submitted ? (
          <div className="space-y-4">
            <div className="space-y-1">
              <h1 className="text-2xl font-semibold tracking-tight">Check your inbox</h1>
              <p className="text-sm text-muted-foreground">
                If <span className="font-medium text-foreground">{email}</span> is registered,
                you&apos;ll receive a reset link shortly.
              </p>
            </div>
            <p className="text-sm text-muted-foreground">
              Didn&apos;t get it? Check your spam folder or{" "}
              <button
                className="font-medium text-foreground underline-offset-4 hover:underline"
                onClick={() => setSubmitted(false)}
              >
                try again
              </button>
              .
            </p>
            <Link
              href={loginHref}
              className="block text-sm font-medium text-foreground underline-offset-4 hover:underline"
            >
              Back to sign in
            </Link>
          </div>
        ) : (
          <>
            <div className="space-y-1">
              <h1 className="text-2xl font-semibold tracking-tight">Forgot your password?</h1>
              <p className="text-sm text-muted-foreground">
                {studioReturn
                  ? "Enter your email and we&apos;ll send you a reset link so you can return to the Studio sign-in gate."
                  : "Enter your email and we&apos;ll send you a reset link."}
              </p>
            </div>

            <form onSubmit={onSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="email">Email address</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="name@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  disabled={loading}
                  required
                />
              </div>

              {error && <p className="text-sm text-destructive">{error}</p>}

              <Button type="submit" className="w-full" disabled={loading}>
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                Send reset link
              </Button>
            </form>

            <p className="text-center text-sm text-muted-foreground">
              Remember your password?{" "}
              <Link
                href={loginHref}
                className="font-medium text-foreground underline-offset-4 hover:underline"
              >
                Sign in
              </Link>
            </p>
          </>
        )}
      </div>
    </div>
  )
}
