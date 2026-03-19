"use client"

import { useCallback, useEffect, useState } from "react"

import {
  buildStudioSkillCatalogInfo,
  buildStudioSkillDetailInfo,
  type StudioSkillCatalogInfo,
  type StudioSkillCatalogResponse,
  type StudioSkillDetailInfo,
  type StudioSkillDetailResponse,
} from "@/lib/studio-skill-catalog"

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T | null> {
  try {
    const response = await fetch(url, {
      cache: "no-store",
      ...init,
    })
    if (!response.ok) return null
    return (await response.json()) as T
  } catch {
    return null
  }
}

export function useStudioSkillsCatalog() {
  const [catalog, setCatalog] = useState<StudioSkillCatalogInfo>(() => buildStudioSkillCatalogInfo())
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const refresh = useCallback(async (background = false) => {
    if (background) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }

    const payload = await fetchJson<StudioSkillCatalogResponse>("/api/studio/skills")
    setCatalog(buildStudioSkillCatalogInfo(payload))
    setLoading(false)
    setRefreshing(false)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return {
    catalog,
    loading,
    refreshing,
    refresh,
  }
}

export function useStudioSkillDetail(skillKey: string | null) {
  const [detail, setDetail] = useState<StudioSkillDetailInfo>(() => buildStudioSkillDetailInfo())
  const [loading, setLoading] = useState(Boolean(skillKey))

  useEffect(() => {
    let active = true
    if (!skillKey) {
      setDetail(buildStudioSkillDetailInfo())
      setLoading(false)
      return
    }

    setLoading(true)
    void fetchJson<StudioSkillDetailResponse>(`/api/studio/skills/${encodeURIComponent(skillKey)}`).then(
      (payload) => {
        if (!active) return
        setDetail(buildStudioSkillDetailInfo(payload))
        setLoading(false)
      },
    )

    return () => {
      active = false
    }
  }, [skillKey])

  return { detail, loading }
}
