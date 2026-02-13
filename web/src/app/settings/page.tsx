"use client"

import { useEffect, useMemo, useState } from "react"
import { CheckCircle2, KeyRound, Loader2, Plus, Trash2, Wrench, Pencil } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

import { ScholarSubscriptionsPanel } from "@/components/settings/ScholarSubscriptionsPanel"

type ModelEndpoint = {
  id: number
  name: string
  vendor: string
  base_url?: string | null
  api_key_env: string
  api_key?: string
  models: string[]
  task_types: string[]
  enabled: boolean
  is_default: boolean
  api_key_present?: boolean
  key_source?: string
}

type FormState = {
  id?: number
  name: string
  vendor: string
  base_url: string
  api_key_env: string
  api_key: string
  models: string
  task_types: string
  enabled: boolean
  is_default: boolean
}

type Preset = {
  label: string
  name: string
  vendor: string
  base_url: string
  api_key_env: string
  models: string[]
  task_types: string[]
}

const QUICK_PRESETS: Preset[] = [
  {
    label: "OpenAI",
    name: "OpenAI",
    vendor: "openai",
    base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    models: ["gpt-4o-mini"],
    task_types: ["default", "summary", "chat"],
  },
  {
    label: "Anthropic",
    name: "Anthropic",
    vendor: "anthropic",
    base_url: "",
    api_key_env: "ANTHROPIC_API_KEY",
    models: ["claude-3-5-sonnet-20241022"],
    task_types: ["reasoning", "analysis"],
  },
  {
    label: "OpenRouter",
    name: "OpenRouter",
    vendor: "openai_compatible",
    base_url: "https://openrouter.ai/api/v1",
    api_key_env: "OPENROUTER_API_KEY",
    models: ["openai/gpt-4o-mini"],
    task_types: ["reasoning", "review"],
  },
  {
    label: "Ollama",
    name: "Local Ollama",
    vendor: "ollama",
    base_url: "http://localhost:11434",
    api_key_env: "OLLAMA_API_KEY",
    models: ["llama3.1"],
    task_types: ["default", "chat"],
  },
]

const EMPTY_FORM: FormState = {
  name: "",
  vendor: "openai_compatible",
  base_url: "",
  api_key_env: "OPENAI_API_KEY",
  api_key: "",
  models: "",
  task_types: "",
  enabled: true,
  is_default: false,
}

function toPayload(form: FormState) {
  return {
    name: form.name.trim(),
    vendor: form.vendor,
    base_url: form.base_url.trim() || null,
    api_key_env: form.api_key_env.trim(),
    api_key: form.api_key,
    models: form.models.split(",").map((x) => x.trim()).filter(Boolean),
    task_types: form.task_types.split(",").map((x) => x.trim()).filter(Boolean),
    enabled: form.enabled,
    is_default: form.is_default,
  }
}

function maskKey(key?: string): string {
  if (!key) return ""
  if (key.length <= 8) return "****"
  return "****" + key.slice(-4)
}

function statusDot(item: ModelEndpoint) {
  if (item.is_default) return "bg-green-500"
  if (!item.api_key_present) return "bg-red-500"
  return "bg-gray-400"
}

export default function SettingsPage() {
  const [items, setItems] = useState<ModelEndpoint[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testingId, setTestingId] = useState<number | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  const editing = useMemo(() => typeof form.id === "number", [form.id])

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("/api/model-endpoints")
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const payload = await res.json()
      setItems(payload.items || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load().catch(() => {}) }, [])

  function openAdd(preset?: Preset) {
    const base = { ...EMPTY_FORM }
    if (preset) {
      base.name = preset.name
      base.vendor = preset.vendor
      base.base_url = preset.base_url
      base.api_key_env = preset.api_key_env
      base.models = preset.models.join(", ")
      base.task_types = preset.task_types.join(", ")
    }
    setForm(base)
    setError(null)
    setMessage(null)
    setDialogOpen(true)
  }

  function openEdit(item: ModelEndpoint) {
    setForm({
      id: item.id,
      name: item.name,
      vendor: item.vendor,
      base_url: item.base_url || "",
      api_key_env: item.api_key_env,
      api_key: "",
      models: (item.models || []).join(", "),
      task_types: (item.task_types || []).join(", "),
      enabled: item.enabled,
      is_default: item.is_default,
    })
    setError(null)
    setMessage(null)
    setDialogOpen(true)
  }

  async function saveItem() {
    setSaving(true)
    setError(null)
    setMessage(null)
    try {
      const payload = toPayload(form)
      if (!payload.name) throw new Error("Name is required")
      if (!payload.models.length) throw new Error("At least one model is required")
      const res = await fetch(editing ? `/api/model-endpoints/${form.id}` : "/api/model-endpoints", {
        method: editing ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const text = await res.text().catch(() => "")
        throw new Error(text || `${res.status} ${res.statusText}`)
      }
      await load()
      setDialogOpen(false)
      setMessage(editing ? "Provider updated." : "Provider created.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  async function deleteItem(id: number) {
    if (!confirm("Delete this provider?")) return
    setError(null)
    setMessage(null)
    try {
      const res = await fetch(`/api/model-endpoints/${id}`, { method: "DELETE" })
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      await load()
      setMessage("Provider removed.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function activateItem(id: number) {
    setError(null)
    setMessage(null)
    try {
      const res = await fetch(`/api/model-endpoints/${id}/activate`, { method: "POST" })
      if (!res.ok) throw new Error(await res.text().catch(() => `${res.status}`))
      await load()
      setMessage("Provider activated.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function testItem(id: number) {
    setTestingId(id)
    setError(null)
    setMessage(null)
    try {
      const res = await fetch(`/api/model-endpoints/${id}/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ remote: false }),
      })
      const payload = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(String(payload?.detail || `${res.status}`))
      setMessage(payload?.message || "Connection test passed.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setTestingId(null)
    }
  }

  return (
    <div className="flex-1 space-y-4 p-8 pt-6 max-w-3xl">
      <h2 className="text-2xl font-bold tracking-tight">Settings</h2>

      <div>
        <h3 className="text-base font-semibold">Model Providers</h3>
        <p className="text-sm text-muted-foreground">Configure LLM providers for paper analysis</p>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {message && <p className="text-sm text-green-600">{message}</p>}

      {/* Provider Cards */}
      <div className="space-y-2">
        {loading ? (
          <p className="text-sm text-muted-foreground">Loading providers...</p>
        ) : !items.length ? (
          <p className="text-sm text-muted-foreground">No providers configured yet.</p>
        ) : (
          items.map((item) => (
            <Card
              key={item.id}
              className={`relative overflow-hidden ${item.is_default ? "border-l-4 border-l-green-500" : ""}`}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className={`h-2.5 w-2.5 rounded-full shrink-0 ${statusDot(item)}`} />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-sm">{item.name}</span>
                        {item.is_default && (
                          <Badge className="text-xs">Default</Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground truncate">
                        {(item.models || []).join(", ") || "no models"} · {item.base_url || "(default URL)"}
                      </p>
                      <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                        <span className="text-xs text-muted-foreground">
                          Key: {item.api_key_present ? maskKey(item.api_key || "present") : "missing"}
                        </span>
                        {item.key_source === "keychain" && (
                          <span className="inline-flex items-center gap-0.5 text-xs text-muted-foreground">
                            <KeyRound className="h-3 w-3" /> Keychain
                          </span>
                        )}
                        {(item.task_types || []).length > 0 && (
                          <span className="text-xs text-muted-foreground">
                            Tasks: {item.task_types.join(", ")}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => testItem(item.id)}
                      disabled={testingId === item.id}
                      title="Test"
                    >
                      {testingId === item.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wrench className="h-3.5 w-3.5" />}
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => openEdit(item)} title="Edit">
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    {!item.is_default && (
                      <Button size="sm" variant="ghost" onClick={() => activateItem(item.id)} title="Set default">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" onClick={() => deleteItem(item.id)} title="Delete" className="text-destructive hover:text-destructive">
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Add Provider + Quick Presets */}
      <div className="space-y-3">
        <Button onClick={() => openAdd()} variant="outline">
          <Plus className="h-4 w-4 mr-1.5" /> Add Provider
        </Button>

        <div>
          <p className="text-xs text-muted-foreground mb-1.5">Quick Presets</p>
          <div className="flex flex-wrap gap-1.5">
            {QUICK_PRESETS.map((preset) => (
              <Button key={preset.label} variant="secondary" size="sm" onClick={() => openAdd(preset)}>
                {preset.label}
              </Button>
            ))}
          </div>
        </div>
      </div>

      <div className="pt-4 border-t">
        <ScholarSubscriptionsPanel />
      </div>

      {/* Add/Edit Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>{editing ? "Edit Provider" : "Add Provider"}</DialogTitle>
            <DialogDescription>
              {editing ? "Update provider configuration." : "Configure a new LLM provider."}
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-sm font-medium">Name</label>
                <Input value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="My Provider" />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium">Vendor</label>
                <select
                  value={form.vendor}
                  onChange={(e) => setForm((p) => ({ ...p, vendor: e.target.value }))}
                  className="h-9 rounded-md border bg-background px-3 text-sm w-full"
                >
                  <option value="openai_compatible">OpenAI Compatible</option>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="ollama">Ollama</option>
                </select>
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-sm font-medium">Base URL</label>
              <Input value={form.base_url} onChange={(e) => setForm((p) => ({ ...p, base_url: e.target.value }))} placeholder="https://api.openai.com/v1" />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-sm font-medium">API Key Env</label>
                <Input value={form.api_key_env} onChange={(e) => setForm((p) => ({ ...p, api_key_env: e.target.value }))} placeholder="OPENAI_API_KEY" />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium">API Key</label>
                <Input
                  type="password"
                  value={form.api_key}
                  onChange={(e) => setForm((p) => ({ ...p, api_key: e.target.value }))}
                  placeholder={editing ? "leave blank to keep" : "sk-..."}
                />
                <p className="text-xs text-muted-foreground flex items-center gap-1">
                  <KeyRound className="h-3 w-3" /> Stored in Keychain
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <label className="text-sm font-medium">Models</label>
                <Input value={form.models} onChange={(e) => setForm((p) => ({ ...p, models: e.target.value }))} placeholder="gpt-4o-mini, gpt-4o" />
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium">Task Routing</label>
                <Input value={form.task_types} onChange={(e) => setForm((p) => ({ ...p, task_types: e.target.value }))} placeholder="default, summary" />
              </div>
            </div>

            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={form.enabled} onChange={(e) => setForm((p) => ({ ...p, enabled: e.target.checked }))} />
                Enabled
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={form.is_default} onChange={(e) => setForm((p) => ({ ...p, is_default: e.target.checked }))} />
                Default
              </label>
            </div>

            {error && <p className="text-sm text-destructive">{error}</p>}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button>
            <Button onClick={saveItem} disabled={saving}>
              {saving && <Loader2 className="h-4 w-4 animate-spin mr-1.5" />}
              {editing ? "Update" : "Create"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
