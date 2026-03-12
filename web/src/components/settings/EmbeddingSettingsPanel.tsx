"use client"

import { useEffect, useMemo, useState } from "react"
import { CheckCircle2, KeyRound, Loader2, Sparkles, Wrench } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

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

const EMPTY_FORM: FormState = {
  provider: "openai",
  base_url: "",
  api_key_env: "PAPERBOT_EMBEDDING_API_KEY",
  api_key: "",
  model: "text-embedding-3-small",
}

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
  const [mode, setMode] = useState<"environment" | "custom">("environment")
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const savedKeyHint = useMemo(() => {
    if (!data?.item.api_key_present) return "No saved key"
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
      setMode(payload.item.enabled ? "custom" : "environment")
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

  async function saveSettings() {
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const payload =
        mode === "custom"
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
      setForm({ ...toFormState(saved.item), api_key: "" })
      setMessage(mode === "custom" ? "Embedding endpoint saved." : "Environment fallback restored.")
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
      const payload =
        mode === "custom"
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
        <CardContent className="p-5 space-y-4">
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading embedding settings...</p>
          ) : (
            <>
              <Tabs value={mode} onValueChange={(value) => setMode(value as "environment" | "custom")}>
                <TabsList className="grid w-full max-w-sm grid-cols-2">
                  <TabsTrigger value="environment">Environment</TabsTrigger>
                  <TabsTrigger value="custom">Custom Endpoint</TabsTrigger>
                </TabsList>

                <TabsContent value="environment" className="mt-4 space-y-3">
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Model</p>
                      <p className="mt-1 text-sm font-medium">{data?.environment.model || "text-embedding-3-small"}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{data?.environment.model_env || "default"}</p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">Base URL</p>
                      <p className="mt-1 text-sm font-medium break-all">
                        {data?.environment.base_url || "OpenAI default"}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">{data?.environment.base_url_env || "OPENAI_BASE_URL"}</p>
                    </div>
                    <div className="rounded-xl border p-3">
                      <p className="text-xs text-muted-foreground">API Key</p>
                      <p className="mt-1 text-sm font-medium">
                        {data?.environment.api_key_present ? "Detected" : "Missing"}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">{data?.environment.api_key_env || "OPENAI_API_KEY"}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 rounded-xl bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
                    <Sparkles className="h-3.5 w-3.5" />
                    Use this mode when your chat relay also exposes a real <code>/embeddings</code> endpoint.
                  </div>
                </TabsContent>

                <TabsContent value="custom" className="mt-4 space-y-3">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1">
                      <label className="text-sm font-medium">Provider</label>
                      <Input value="OpenAI Compatible" readOnly disabled />
                    </div>
                    <div className="space-y-1">
                      <label className="text-sm font-medium">Model</label>
                      <Input
                        value={form.model}
                        onChange={(e) => setForm((prev) => ({ ...prev, model: e.target.value }))}
                        placeholder="text-embedding-3-small"
                      />
                    </div>
                  </div>

                  <div className="space-y-1">
                    <label className="text-sm font-medium">Base URL</label>
                    <Input
                      value={form.base_url}
                      onChange={(e) => setForm((prev) => ({ ...prev, base_url: e.target.value }))}
                      placeholder="https://your-embedding-endpoint/v1"
                    />
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1">
                      <label className="text-sm font-medium">API Key Env</label>
                      <Input
                        value={form.api_key_env}
                        onChange={(e) => setForm((prev) => ({ ...prev, api_key_env: e.target.value }))}
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
                </TabsContent>
              </Tabs>

              {error && <p className="text-sm text-destructive">{error}</p>}
              {message && <p className="text-sm text-green-600">{message}</p>}

              <div className="flex items-center justify-between gap-3 border-t pt-4">
                <div className="text-xs text-muted-foreground">
                  {data?.item.updated_at ? `Updated ${new Date(data.item.updated_at).toLocaleString()}` : "No custom endpoint saved yet."}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" onClick={testSettings} disabled={testing}>
                    {testing ? (
                      <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                    ) : (
                      <Wrench className="mr-1.5 h-4 w-4" />
                    )}
                    Test
                  </Button>
                  <Button onClick={saveSettings} disabled={saving}>
                    {saving ? (
                      <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                    ) : (
                      <CheckCircle2 className="mr-1.5 h-4 w-4" />
                    )}
                    Save
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
