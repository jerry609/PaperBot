"use client"

import { useEffect, useMemo, useState } from "react"
import {
  CheckCircle2,
  KeyRound,
  Loader2,
  Sparkles,
  WandSparkles,
  Wrench,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"

type EmbeddingSettingsItem = {
  provider: string
  base_url?: string | null
  api_key_env: string
  api_key?: string
  api_key_present?: boolean
  key_source?: string
  model: string
  enabled: boolean
  updated_at?: string | null
}

type EmbeddingEnvironment = {
  provider: string
  api_key_env: string
  api_key_present: boolean
  base_url_env: string
  base_url?: string | null
  model_env: string
  model: string
}

type EmbeddingSettingsResponse = {
  item: EmbeddingSettingsItem
  environment: EmbeddingEnvironment
  effective_source: "settings" | "environment" | "none"
}

type FormState = {
  provider: string
  base_url: string
  api_key_env: string
  api_key: string
  model: string
}

type Preset = {
  label: string
  description: string
  values: Omit<FormState, "api_key">
}

const EMPTY_FORM: FormState = {
  provider: "openai",
  base_url: "",
  api_key_env: "PAPERBOT_EMBEDDING_API_KEY",
  api_key: "",
  model: "text-embedding-3-small",
}

const EMBEDDING_PRESETS: Preset[] = [
  {
    label: "OpenAI",
    description: "Default OpenAI embeddings endpoint",
    values: {
      provider: "openai",
      base_url: "https://api.openai.com/v1",
      api_key_env: "PAPERBOT_EMBEDDING_API_KEY",
      model: "text-embedding-3-small",
    },
  },
  {
    label: "NVIDIA NIM",
    description: "Use NVIDIA's OpenAI-compatible embeddings API",
    values: {
      provider: "openai",
      base_url: "https://integrate.api.nvidia.com/v1",
      api_key_env: "NVIDIA_API_KEY",
      model: "nvidia/nv-embed-v1",
    },
  },
]

function toFormState(item: EmbeddingSettingsItem): FormState {
  return {
    provider: item.provider || "openai",
    base_url: item.base_url || "",
    api_key_env: item.api_key_env || "PAPERBOT_EMBEDDING_API_KEY",
    api_key: "",
    model: item.model || "text-embedding-3-small",
  }
}

function effectiveBadge(source: EmbeddingSettingsResponse["effective_source"]) {
  switch (source) {
    case "settings":
      return <Badge>Custom endpoint</Badge>
    case "environment":
      return <Badge variant="secondary">Environment</Badge>
    default:
      return <Badge variant="outline">Unavailable</Badge>
  }
}

export function EmbeddingSettingsPanel() {
  const [data, setData] = useState<EmbeddingSettingsResponse | null>(null)
  const [useCustomEndpoint, setUseCustomEndpoint] = useState(false)
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const savedKeyHint = useMemo(() => {
    if (!data?.item.api_key_present) return "No saved custom key"
    if (data.item.key_source === "keychain") return "Stored in Keychain"
    return "Stored securely"
  }, [data])

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/embedding-settings")
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const payload = (await res.json()) as EmbeddingSettingsResponse
      setData(payload)
      setUseCustomEndpoint(payload.item.enabled)
      setForm(toFormState(payload.item))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load().catch(() => {})
  }, [])

  function applyPreset(preset: Preset) {
    setUseCustomEndpoint(true)
    setForm((prev) => ({
      ...prev,
      ...preset.values,
      api_key: "",
    }))
    setMessage(`Applied ${preset.label} preset.`)
    setError(null)
  }

  async function saveSettings() {
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const payload = useCustomEndpoint
        ? {
            enabled: true,
            provider: form.provider,
            base_url: form.base_url.trim() || null,
            api_key_env: form.api_key_env.trim() || "PAPERBOT_EMBEDDING_API_KEY",
            api_key: form.api_key,
            model: form.model.trim() || "text-embedding-3-small",
          }
        : { enabled: false }

      const res = await fetch("/api/embedding-settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `${res.status} ${res.statusText}`)
      }
      const saved = (await res.json()) as EmbeddingSettingsResponse
      setData(saved)
      setUseCustomEndpoint(saved.item.enabled)
      setForm({ ...toFormState(saved.item), api_key: "" })
      setMessage(saved.item.enabled ? "Embedding endpoint saved." : "Environment fallback restored.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  async function testSettings() {
    setTesting(true)
    setError(null)
    setMessage(null)
    try {
      const payload = useCustomEndpoint
        ? {
            enabled: true,
            provider: form.provider,
            base_url: form.base_url.trim() || null,
            api_key_env: form.api_key_env.trim() || "PAPERBOT_EMBEDDING_API_KEY",
            api_key: form.api_key || undefined,
            model: form.model.trim() || "text-embedding-3-small",
            remote: true,
          }
        : { enabled: false, remote: true }

      const res = await fetch("/api/embedding-settings/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      const body = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(String(body?.detail || `${res.status} ${res.statusText}`))
      setMessage(body?.message || "Embedding test passed.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold">Embedding Endpoint</h3>
          <p className="text-sm text-muted-foreground">
            Controls semantic retrieval for track routing and document evidence.
          </p>
        </div>
        {data ? effectiveBadge(data.effective_source) : null}
      </div>

      <Card>
        <CardContent className="space-y-4 p-5">
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading embedding settings...</p>
          ) : (
            <>
              <div className="rounded-xl border border-border/70 bg-muted/20 p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-1">
                    <p className="text-sm font-medium">Use custom endpoint</p>
                    <p className="text-xs text-muted-foreground">
                      Override environment embeddings with a dedicated provider config.
                    </p>
                  </div>
                  <Switch
                    checked={useCustomEndpoint}
                    onCheckedChange={(checked) => {
                      setUseCustomEndpoint(checked)
                      setError(null)
                      setMessage(null)
                    }}
                  />
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-lg border bg-background p-3">
                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Active model
                    </p>
                    <p className="mt-1 text-sm font-medium break-words">
                      {useCustomEndpoint ? form.model || "Unset" : data?.environment.model || "Unset"}
                    </p>
                  </div>
                  <div className="rounded-lg border bg-background p-3">
                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Base URL
                    </p>
                    <p className="mt-1 text-sm font-medium break-all">
                      {useCustomEndpoint
                        ? form.base_url || "Unset"
                        : data?.environment.base_url || "OpenAI default"}
                    </p>
                  </div>
                  <div className="rounded-lg border bg-background p-3">
                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      Credential source
                    </p>
                    <p className="mt-1 text-sm font-medium">
                      {useCustomEndpoint
                        ? form.api_key_env || "PAPERBOT_EMBEDDING_API_KEY"
                        : data?.environment.api_key_env || "OPENAI_API_KEY"}
                    </p>
                  </div>
                </div>
              </div>

              {!useCustomEndpoint ? (
                <div className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Environment model</p>
                      <p className="mt-1 text-sm font-medium">
                        {data?.environment.model || "text-embedding-3-small"}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {data?.environment.model_env || "default"}
                      </p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Environment base URL</p>
                      <p className="mt-1 text-sm font-medium break-all">
                        {data?.environment.base_url || "OpenAI default"}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {data?.environment.base_url_env || "OPENAI_BASE_URL"}
                      </p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Environment key</p>
                      <p className="mt-1 text-sm font-medium">
                        {data?.environment.api_key_present ? "Detected" : "Missing"}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {data?.environment.api_key_env || "OPENAI_API_KEY"}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 rounded-xl bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
                    <Sparkles className="h-3.5 w-3.5" />
                    Use environment mode only when that endpoint exposes a real <code>/embeddings</code> route.
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-medium">Quick presets</span>
                      {EMBEDDING_PRESETS.map((preset) => (
                        <Button
                          key={preset.label}
                          type="button"
                          variant="secondary"
                          size="sm"
                          onClick={() => applyPreset(preset)}
                        >
                          <WandSparkles className="mr-1.5 h-3.5 w-3.5" />
                          {preset.label}
                        </Button>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      NVIDIA preset uses <code>NVIDIA_API_KEY</code> and the official NIM OpenAI-compatible endpoint.
                    </p>
                  </div>

                  <div className="rounded-xl border border-border/70 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium">Endpoint</p>
                        <p className="text-xs text-muted-foreground">
                          OpenAI-compatible <code>/v1/embeddings</code> base URL.
                        </p>
                      </div>
                      <Button type="button" variant="outline" size="sm" onClick={testSettings} disabled={testing}>
                        {testing ? (
                          <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                        ) : (
                          <Wrench className="mr-1.5 h-4 w-4" />
                        )}
                        Test endpoint
                      </Button>
                    </div>
                    <div className="mt-3 space-y-1">
                      <label className="text-sm font-medium">Base URL</label>
                      <Input
                        value={form.base_url}
                        onChange={(e) => setForm((prev) => ({ ...prev, base_url: e.target.value }))}
                        placeholder="https://your-embedding-endpoint/v1"
                      />
                    </div>
                  </div>

                  <div className="rounded-xl border border-border/70 p-4">
                    <p className="text-sm font-medium">Credentials</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Save a dedicated key, or point to an existing environment variable.
                    </p>
                    <div className="mt-3 grid gap-3 sm:grid-cols-2">
                      <div className="space-y-1">
                        <label className="text-sm font-medium">API Key Env</label>
                        <Input
                          value={form.api_key_env}
                          onChange={(e) =>
                            setForm((prev) => ({ ...prev, api_key_env: e.target.value }))
                          }
                          placeholder="PAPERBOT_EMBEDDING_API_KEY"
                        />
                      </div>
                      <div className="space-y-1">
                        <label className="text-sm font-medium">API Key</label>
                        <Input
                          type="password"
                          value={form.api_key}
                          onChange={(e) => setForm((prev) => ({ ...prev, api_key: e.target.value }))}
                          placeholder={data?.item.api_key_present ? "leave blank to keep" : "sk-..."}
                        />
                        <p className="flex items-center gap-1 text-xs text-muted-foreground">
                          <KeyRound className="h-3 w-3" />
                          {savedKeyHint}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-xl border border-border/70 p-4">
                    <p className="text-sm font-medium">Embedding model</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Set the exact model id used for document and routing embeddings.
                    </p>
                    <div className="mt-3 space-y-1">
                      <label className="text-sm font-medium">Model</label>
                      <Input
                        value={form.model}
                        onChange={(e) => setForm((prev) => ({ ...prev, model: e.target.value }))}
                        placeholder="text-embedding-3-small"
                      />
                    </div>
                  </div>
                </div>
              )}

              {error && <p className="text-sm text-destructive">{error}</p>}
              {message && <p className="text-sm text-green-600">{message}</p>}

              <div className="flex items-center justify-between gap-3 border-t pt-4">
                <div className="text-xs text-muted-foreground">
                  {data?.item.updated_at
                    ? `Updated ${new Date(data.item.updated_at).toLocaleString()}`
                    : "No custom endpoint saved yet."}
                </div>
                <Button onClick={saveSettings} disabled={saving}>
                  {saving ? (
                    <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                  ) : (
                    <CheckCircle2 className="mr-1.5 h-4 w-4" />
                  )}
                  Save
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
