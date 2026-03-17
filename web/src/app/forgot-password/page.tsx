"use client"

import { Suspense, useMemo, useState } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout"
import { buildAuthGateContext, buildLoginHref } from "@/lib/auth-flow"

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
  const gate = useMemo(() => buildAuthGateContext(callbackUrl), [callbackUrl])
  const loginHref = buildLoginHref(callbackUrl)
  const studioReturn = gate.destination.startsWith("/studio")

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
      setSubmitted(true)
    } catch {
      setError("Something went wrong. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthSplitLayout
      panelEyebrow={gate.eyebrow}
      panelTitle={gate.title}
      panelDescription={gate.description}
      panelSteps={gate.steps}
      destinationLabel={gate.destinationLabel}
      destination={gate.destination}
      cardEyebrow="Password recovery"
      cardTitle={submitted ? "Check your inbox" : "Reset your password"}
      cardDescription={
        submitted
          ? "If the address is registered, you'll receive a reset link shortly."
          : studioReturn
            ? "Enter your email and we'll send a reset link so you can return to the Studio sign-in gate."
            : "Enter your email and we'll send you a reset link."
      }
    >
      {submitted ? (
        <div className="space-y-4">
          <div className="rounded-[22px] border border-slate-200 bg-[#f8faf5] px-4 py-4">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Email
            </p>
            <p className="mt-1 break-all text-sm font-medium text-slate-900">{email}</p>
          </div>

          <p className="text-sm leading-6 text-slate-500">
            Didn&apos;t get it? Check your spam folder or{" "}
            <button
              className="font-medium text-slate-900 underline-offset-4 hover:underline"
              onClick={() => setSubmitted(false)}
            >
              try again
            </button>
            .
          </p>

          <Link
            href={loginHref}
            className="block text-sm font-medium text-slate-900 underline-offset-4 hover:underline"
          >
            Back to sign in
          </Link>
        </div>
      ) : (
        <>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="email">Email address</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="name@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                required
                className="h-11 rounded-2xl border-slate-200 bg-white"
              />
            </div>

            {error ? (
              <p className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                {error}
              </p>
            ) : null}

            <Button
              type="submit"
              className="h-11 w-full rounded-full bg-slate-900 text-white hover:bg-slate-800"
              disabled={loading}
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Send reset link
            </Button>
          </form>

          <p className="text-center text-sm text-slate-500">
            Remember your password?{" "}
            <Link
              href={loginHref}
              className="font-medium text-slate-900 underline-offset-4 hover:underline"
            >
              Sign in
            </Link>
          </p>
        </>
      )}
    </AuthSplitLayout>
  )
}
