"use client"

import { useEffect, useState } from "react"

import {
  buildStudioRuntimeInfo,
  type StudioRuntimeCwdResponse,
  type StudioRuntimeInfo,
  type StudioRuntimeStatusResponse,
} from "@/lib/studio-runtime"

interface StudioRuntimeState {
  info: StudioRuntimeInfo
  loading: boolean
  refreshing: boolean
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url, { cache: "no-store" })
    const payload = (await res.json()) as T
    return payload
  } catch {
    return null
  }
}

export function useStudioRuntime(): StudioRuntimeState {
  const [state, setState] = useState<StudioRuntimeState>({
    info: buildStudioRuntimeInfo(),
    loading: true,
    refreshing: false,
  })

  useEffect(() => {
    let active = true
    let intervalId: number | null = null

    const load = async (background = false) => {
      if (background) {
        setState((current) => ({
          ...current,
          refreshing: true,
        }))
      }

      const [status, cwd] = await Promise.all([
        fetchJson<StudioRuntimeStatusResponse>("/api/studio/status"),
        fetchJson<StudioRuntimeCwdResponse>("/api/studio/cwd"),
      ])

      if (!active) return
      setState({
        info: buildStudioRuntimeInfo(status, cwd),
        loading: false,
        refreshing: false,
      })
    }

    void load()

    const handleVisibilityRefresh = () => {
      if (document.visibilityState !== "visible") return
      void load(true)
    }

    const handleOnline = () => {
      void load(true)
    }

    intervalId = window.setInterval(() => {
      void load(true)
    }, 15000)

    window.addEventListener("focus", handleVisibilityRefresh)
    window.addEventListener("online", handleOnline)
    document.addEventListener("visibilitychange", handleVisibilityRefresh)

    return () => {
      active = false
      if (intervalId !== null) {
        window.clearInterval(intervalId)
      }
      window.removeEventListener("focus", handleVisibilityRefresh)
      window.removeEventListener("online", handleOnline)
      document.removeEventListener("visibilitychange", handleVisibilityRefresh)
    }
  }, [])

  return state
}
