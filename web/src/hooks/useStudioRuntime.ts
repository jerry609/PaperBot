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
  })

  useEffect(() => {
    let active = true

    const load = async () => {
      const [status, cwd] = await Promise.all([
        fetchJson<StudioRuntimeStatusResponse>("/api/studio/status"),
        fetchJson<StudioRuntimeCwdResponse>("/api/studio/cwd"),
      ])

      if (!active) return
      setState({
        info: buildStudioRuntimeInfo(status, cwd),
        loading: false,
      })
    }

    void load()

    return () => {
      active = false
    }
  }, [])

  return state
}
