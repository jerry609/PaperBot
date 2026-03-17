"use client"

import { Suspense, useMemo, useState } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import Link from "next/link"
import { Eye, EyeOff, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout"
import { buildAuthGateContext, buildLoginHref } from "@/lib/auth-flow"

type Strength = { score: 0 | 1 | 2 | 3 | 4; label: string; color: string }

function getStrength(pw: string): Strength {
  if (!pw) return { score: 0, label: "", color: "" }
  let score = 0
  if (pw.length >= 8) score++
  if (pw.length >= 12) score++
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++
  if (/\d/.test(pw) && /[^A-Za-z0-9]/.test(pw)) score++
  const levels: Strength[] = [
    { score: 0, label: "", color: "" },
    { score: 1, label: "Weak", color: "bg-red-500" },
    { score: 2, label: "Fair", color: "bg-orange-400" },
    { score: 3, label: "Good", color: "bg-yellow-400" },
    { score: 4, label: "Strong", color: "bg-green-500" },
  ]
  return levels[score as 0 | 1 | 2 | 3 | 4]
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<ResetPasswordContent token="" callbackUrl="/login" />}>
      <ResetPasswordWithSearchParams />
    </Suspense>
  )
}

function ResetPasswordWithSearchParams() {
  const searchParams = useSearchParams()
  const token = searchParams.get("token") ?? ""
  const callbackUrl = searchParams.get("callbackUrl") || "/login"
  return <ResetPasswordContent token={token} callbackUrl={callbackUrl} />
}

function ResetPasswordContent({
  token,
  callbackUrl,
}: {
  token: string
  callbackUrl: string
}) {
  const router = useRouter()
  const [password, setPassword] = useState("")
  const [confirm, setConfirm] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const gate = useMemo(() => buildAuthGateContext(callbackUrl), [callbackUrl])
  const loginHref = buildLoginHref(callbackUrl)
  const strength = useMemo(() => getStrength(password), [password])
  const mismatch = confirm.length > 0 && confirm !== password

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirm) {
      setError("Passwords do not match.")
      return
    }
    setError(null)
    setLoading(true)

    try {
      const res = await fetch("/api/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, new_password: password }),
      })
      if (!res.ok) {
        const body = (await res.json().catch(() => null)) as { detail?: string } | null
        setError(body?.detail ?? "Failed to reset password. The link may have expired.")
        return
      }
      setDone(true)
      setTimeout(() => router.push(loginHref), 2500)
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
      cardEyebrow="Password reset"
      cardTitle={
        !token
          ? "Reset link unavailable"
          : done
            ? "Password updated"
            : "Set a new password"
      }
      cardDescription={
        !token
          ? "The reset link is missing or expired. Request a fresh link to continue."
          : done
            ? "Your password has been updated. You'll return to sign in automatically."
            : "Choose a strong password before returning to sign in."
      }
    >
      {!token ? (
        <div className="space-y-4">
          <p className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
            Invalid or missing reset token.
          </p>
          <Link
            href={`/forgot-password?callbackUrl=${encodeURIComponent(callbackUrl)}`}
            className="block text-sm font-medium text-slate-900 underline-offset-4 hover:underline"
          >
            Request a new link
          </Link>
        </div>
      ) : done ? (
        <div className="space-y-4">
          <div className="rounded-[22px] border border-emerald-200 bg-emerald-50 px-4 py-4 text-sm text-emerald-800">
            Password updated successfully. Redirecting you to sign in...
          </div>
          <Link
            href={loginHref}
            className="block text-sm font-medium text-slate-900 underline-offset-4 hover:underline"
          >
            Continue now
          </Link>
        </div>
      ) : (
        <>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="password">New password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="new-password"
                  placeholder="Min. 8 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                  required
                  minLength={8}
                  className="h-11 rounded-2xl border-slate-200 bg-white pr-10"
                />
                <button
                  type="button"
                  tabIndex={-1}
                  onClick={() => setShowPassword((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-slate-700"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>

              {password.length > 0 ? (
                <div className="space-y-1">
                  <div className="flex gap-1">
                    {([1, 2, 3, 4] as const).map((level) => (
                      <div
                        key={level}
                        className={`h-1 flex-1 rounded-full transition-colors duration-300 ${
                          strength.score >= level ? strength.color : "bg-slate-200"
                        }`}
                      />
                    ))}
                  </div>
                  {strength.label ? (
                    <p className="text-xs text-slate-500">
                      Strength:{" "}
                      <span
                        className={`font-medium ${
                          strength.score <= 1
                            ? "text-red-500"
                            : strength.score === 2
                              ? "text-orange-400"
                              : strength.score === 3
                                ? "text-yellow-500"
                                : "text-green-500"
                        }`}
                      >
                        {strength.label}
                      </span>
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="confirm">Confirm password</Label>
              <div className="relative">
                <Input
                  id="confirm"
                  type={showConfirm ? "text" : "password"}
                  autoComplete="new-password"
                  placeholder="Re-enter your password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  disabled={loading}
                  required
                  className={`h-11 rounded-2xl bg-white pr-10 ${
                    mismatch ? "border-destructive focus-visible:ring-destructive/30" : "border-slate-200"
                  }`}
                />
                <button
                  type="button"
                  tabIndex={-1}
                  onClick={() => setShowConfirm((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 transition-colors hover:text-slate-700"
                >
                  {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {mismatch ? (
                <p className="text-xs text-destructive">Passwords do not match.</p>
              ) : null}
            </div>

            {error ? (
              <p className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                {error}
              </p>
            ) : null}

            <Button
              type="submit"
              className="h-11 w-full rounded-full bg-slate-900 text-white hover:bg-slate-800"
              disabled={loading || mismatch}
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Update password
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
