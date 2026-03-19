"use client"

import type { ReactNode } from "react"
import {
  ArrowRight,
  BookOpen,
  FolderOpen,
  MessageSquare,
  ShieldCheck,
} from "lucide-react"
import type { AuthGateStep, AuthGateStepIcon } from "@/lib/auth-flow"

interface AuthSplitLayoutProps {
  panelEyebrow: string
  panelTitle: string
  panelDescription: string
  panelSteps: AuthGateStep[]
  destinationLabel?: string
  destination?: string
  cardEyebrow?: string
  cardTitle: string
  cardDescription: string
  children: ReactNode
}

function stepIcon(step: AuthGateStepIcon) {
  if (step === "workspace") return FolderOpen
  if (step === "chat") return MessageSquare
  return ShieldCheck
}

export function AuthSplitLayout({
  panelEyebrow,
  panelTitle,
  panelDescription,
  panelSteps,
  destinationLabel,
  destination,
  cardEyebrow,
  cardTitle,
  cardDescription,
  children,
}: AuthSplitLayoutProps) {
  return (
    <div className="min-h-screen bg-[#f5f6f1]">
      <div className="grid min-h-screen lg:grid-cols-[minmax(0,1.05fr)_minmax(420px,0.95fr)]">
        <div className="relative hidden overflow-hidden border-r border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(232,237,227,0.95),_rgba(243,245,239,0.98)_58%,_rgba(250,250,247,1)_100%)] lg:flex">
          <div className="absolute inset-0 bg-[linear-gradient(135deg,rgba(255,255,255,0.65),transparent_52%)]" />
          <div className="relative flex w-full flex-col justify-between px-12 py-10">
            <div className="flex items-center gap-2.5 text-slate-800">
              <div className="flex h-9 w-9 items-center justify-center rounded-2xl border border-slate-200 bg-white">
                <BookOpen className="h-4 w-4" />
              </div>
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  PaperBot
                </p>
                <p className="text-sm font-semibold text-slate-900">Authentication</p>
              </div>
            </div>

            <div className="max-w-[34rem] space-y-6">
              <div className="space-y-3">
                <span className="inline-flex rounded-full border border-slate-200 bg-white px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                  {panelEyebrow}
                </span>
                <div className="space-y-2">
                  <h1 className="text-[34px] font-semibold leading-[1.05] tracking-[-0.03em] text-slate-950">
                    {panelTitle}
                  </h1>
                  <p className="max-w-[30rem] text-[15px] leading-7 text-slate-600">
                    {panelDescription}
                  </p>
                </div>
              </div>

              {destinationLabel && destination ? (
                <div className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_40px_rgba(15,23,42,0.06)]">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                        {destinationLabel}
                      </p>
                      <p className="mt-1 text-base font-semibold text-slate-900">{panelTitle}</p>
                    </div>
                    <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] text-emerald-700">
                      callback ready
                    </span>
                  </div>
                  <div className="mt-4 flex items-center gap-2 rounded-2xl border border-slate-200 bg-[#f7f8f4] px-3 py-2.5">
                    <ArrowRight className="h-4 w-4 text-slate-400" />
                    <span className="truncate font-mono text-[11px] text-slate-600">{destination}</span>
                  </div>
                </div>
              ) : null}

              <div className="grid gap-3">
                {panelSteps.map((step, index) => {
                  const Icon = stepIcon(step.icon)
                  return (
                    <div
                      key={step.title}
                      className="flex items-start gap-3 rounded-[24px] border border-slate-200 bg-white/70 px-4 py-4"
                    >
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-700">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="rounded-full border border-slate-200 bg-white px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                            {index + 1}
                          </span>
                          <p className="text-sm font-semibold text-slate-900">{step.title}</p>
                        </div>
                        <p className="mt-1 text-[13px] leading-6 text-slate-600">{step.detail}</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <p className="text-xs text-slate-400">Secure authentication gate for protected PaperBot surfaces.</p>
          </div>
        </div>

        <div className="flex items-center justify-center px-6 py-10">
          <div className="w-full max-w-[430px] space-y-6">
            <div className="flex items-center gap-2 lg:hidden">
              <BookOpen className="h-5 w-5 text-slate-700" />
              <span className="font-semibold tracking-tight text-slate-900">PaperBot</span>
            </div>

            <div className="rounded-[30px] border border-slate-200 bg-white/95 p-6 shadow-[0_20px_44px_rgba(15,23,42,0.06)]">
              <div className="space-y-5">
                {cardEyebrow ? (
                  <span className="inline-flex rounded-full border border-slate-200 bg-[#f7f8f4] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {cardEyebrow}
                  </span>
                ) : null}

                <div className="space-y-1.5">
                  <h2 className="text-[28px] font-semibold tracking-[-0.03em] text-slate-950">
                    {cardTitle}
                  </h2>
                  <p className="text-sm leading-6 text-slate-600">
                    {cardDescription}
                  </p>
                </div>

                {destinationLabel && destination ? (
                  <div className="rounded-[22px] border border-slate-200 bg-[#f8faf5] p-4">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {destinationLabel}
                    </p>
                    <p className="mt-1 break-all font-mono text-[11px] leading-5 text-slate-700">
                      {destination}
                    </p>
                  </div>
                ) : null}

                {children}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
