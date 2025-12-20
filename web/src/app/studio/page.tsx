"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Play, Code, Terminal, AlertTriangle } from "lucide-react"

export default function DeepCodeStudioPage() {
    return (
        <div className="flex h-[calc(100vh-theme(spacing.16))] flex-col">
            <div className="border-b bg-background p-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <h2 className="text-xl font-semibold">DeepCode Studio</h2>
                    <Badge variant="outline">Attention Is All You Need</Badge>
                    <Badge variant="secondary">PyTorch 2.0</Badge>
                </div>
                <div className="flex items-center gap-2">
                    <Button size="sm" variant="outline">
                        <Code className="mr-2 h-4 w-4" /> Generate Blueprint
                    </Button>
                    <Button size="sm">
                        <Play className="mr-2 h-4 w-4" /> Run Reproduction
                    </Button>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-12 bg-muted/20">
                {/* Blueprint / Code Viewer */}
                <div className="col-span-8 p-4 border-r bg-background">
                    <Card className="h-full border-dashed shadow-none">
                        <CardHeader>
                            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                                <Code className="h-4 w-4" /> generated_model.py
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <pre className="text-sm font-mono text-muted-foreground">
                                {`class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        ...
# TODO: Implement Multi-Head Attention logic
                        `}
                            </pre>
                        </CardContent>
                    </Card>
                </div>

                {/* Terminal / Logs */}
                <div className="col-span-4 p-4 flex flex-col gap-4">
                    <Card className="h-1/2 bg-black border-zinc-800">
                        <CardHeader className="p-3 border-b border-zinc-800">
                            <CardTitle className="text-xs font-mono text-zinc-400 flex items-center gap-2">
                                <Terminal className="h-3 w-3" /> E2B Sandbox Terminal
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-3 font-mono text-xs text-green-400">
                            <p>$ pip install torch transformers</p>
                            <p className="text-zinc-500">Requirement already satisfied: torch...</p>
                            <p>$ python train.py</p>
                            <p>Epoch 1/10: Loss 2.3412</p>
                            <p>Epoch 2/10: Loss 1.9832</p>
                        </CardContent>
                    </Card>

                    <Card className="h-1/2 border-orange-200 bg-orange-50 dark:bg-orange-950/20 dark:border-orange-900">
                        <CardHeader className="p-3">
                            <CardTitle className="text-xs font-medium text-orange-600 dark:text-orange-400 flex items-center gap-2">
                                <AlertTriangle className="h-3 w-3" /> Self-Healing Debugger
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-3 text-xs">
                            <p className="mb-2">No active errors detected.</p>
                            <p className="text-muted-foreground">Ready to catch exceptions during runtime.</p>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    )
}
