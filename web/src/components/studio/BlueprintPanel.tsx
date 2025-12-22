"use client"

import { useState } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useStudioStore } from "@/lib/store/studio-store"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChevronDown, ChevronRight, FileText } from "lucide-react"

export function BlueprintPanel() {
  const { paperDraft, setPaperDraft, lastGenCodeResult } = useStudioStore()
  const [showAdvanced, setShowAdvanced] = useState(false)

  return (
    <div className="h-full flex flex-col min-w-0 min-h-0 bg-muted/5">
      <div className="border-b px-4 py-3 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="text-sm font-semibold flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" /> Blueprint / Goals
        </div>
        <div className="text-xs text-muted-foreground">Inputs + success criteria for the current project.</div>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="p-3 space-y-3">
          <Card className="py-4">
            <CardHeader className="px-4">
              <CardTitle className="text-sm">Paper Context</CardTitle>
              <CardDescription className="text-xs">Used by Runbook steps (e.g., Paper2Code).</CardDescription>
            </CardHeader>
            <CardContent className="px-4 space-y-3">
              <div className="space-y-1.5">
                <Label htmlFor="paper-title" className="text-xs">Title</Label>
                <Input
                  id="paper-title"
                  value={paperDraft.title}
                  onChange={(e) => setPaperDraft({ title: e.target.value })}
                  placeholder="Paste the paper title"
                />
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="paper-abstract" className="text-xs">Abstract</Label>
                <Textarea
                  id="paper-abstract"
                  value={paperDraft.abstract}
                  onChange={(e) => setPaperDraft({ abstract: e.target.value })}
                  placeholder="Paste the abstract"
                  className="min-h-[120px]"
                />
              </div>

              <div className="pt-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs"
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  {showAdvanced ? <ChevronDown className="h-3.5 w-3.5 mr-1" /> : <ChevronRight className="h-3.5 w-3.5 mr-1" />}
                  Advanced
                </Button>
              </div>

              {showAdvanced && (
                <div className="space-y-1.5">
                  <Label htmlFor="paper-method" className="text-xs">Method (optional)</Label>
                  <Textarea
                    id="paper-method"
                    value={paperDraft.methodSection}
                    onChange={(e) => setPaperDraft({ methodSection: e.target.value })}
                    placeholder="Optional: method section excerpt"
                    className="min-h-[110px]"
                  />
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="py-4">
            <CardHeader className="px-4">
              <CardTitle className="text-sm">Latest Result</CardTitle>
              <CardDescription className="text-xs">Most recent Runbook output.</CardDescription>
            </CardHeader>
            <CardContent className="px-4">
              {!lastGenCodeResult ? (
                <div className="text-sm text-muted-foreground">No runs yet.</div>
              ) : (
                <div className="space-y-1.5 text-sm">
                  <div>
                    Output: <span className="font-mono text-xs">{lastGenCodeResult.outputDir || "—"}</span>
                  </div>
                  <div>
                    Blueprint:{" "}
                    <span className="font-mono text-xs">
                      {lastGenCodeResult.blueprint?.architectureType || "unknown"} / {lastGenCodeResult.blueprint?.domain || "unknown"}
                    </span>
                  </div>
                  <div>
                    Verification: <span className="font-mono text-xs">{lastGenCodeResult.verificationPassed ? "passed" : "not passed"}</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </ScrollArea>
    </div>
  )
}
