"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useMCP } from "@/lib/mcp"
import { MCPServerConfig } from "@/lib/mcp/client"
import { Plug, Plus, Trash2, CheckCircle2, XCircle, Loader2 } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

export function MCPSettings() {
    const { isConnected, isConnecting, tools, servers, error, connect, disconnect } = useMCP()
    const [newServerUrl, setNewServerUrl] = useState("")
    const [newServerName, setNewServerName] = useState("")
    const [configuredServers, setConfiguredServers] = useState<MCPServerConfig[]>([])

    const addServer = () => {
        if (newServerUrl && newServerName) {
            setConfiguredServers([
                ...configuredServers,
                { name: newServerName, url: newServerUrl }
            ])
            setNewServerUrl("")
            setNewServerName("")
        }
    }

    const removeServer = (index: number) => {
        setConfiguredServers(configuredServers.filter((_, i) => i !== index))
    }

    const handleConnect = async () => {
        if (configuredServers.length > 0) {
            await connect(configuredServers)
        }
    }

    return (
        <div className="p-4 space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold flex items-center gap-2">
                    <Plug className="h-4 w-4" />
                    MCP Servers
                </h3>
                {isConnected ? (
                    <span className="text-xs text-green-600 flex items-center gap-1">
                        <CheckCircle2 className="h-3 w-3" /> Connected
                    </span>
                ) : (
                    <span className="text-xs text-muted-foreground flex items-center gap-1">
                        <XCircle className="h-3 w-3" /> Disconnected
                    </span>
                )}
            </div>

            {error && (
                <div className="text-xs text-red-500 bg-red-50 dark:bg-red-900/20 p-2 rounded">
                    {error}
                </div>
            )}

            {/* Add Server Form */}
            <div className="space-y-2">
                <div className="grid grid-cols-2 gap-2">
                    <div>
                        <Label htmlFor="server-name" className="text-xs">Name</Label>
                        <Input
                            id="server-name"
                            placeholder="filesystem"
                            value={newServerName}
                            onChange={(e) => setNewServerName(e.target.value)}
                            className="h-8 text-xs"
                        />
                    </div>
                    <div>
                        <Label htmlFor="server-url" className="text-xs">URL</Label>
                        <Input
                            id="server-url"
                            placeholder="http://localhost:3001/mcp"
                            value={newServerUrl}
                            onChange={(e) => setNewServerUrl(e.target.value)}
                            className="h-8 text-xs"
                        />
                    </div>
                </div>
                <Button
                    size="sm"
                    variant="outline"
                    className="w-full h-8 text-xs"
                    onClick={addServer}
                    disabled={!newServerUrl || !newServerName}
                >
                    <Plus className="h-3 w-3 mr-1" /> Add Server
                </Button>
            </div>

            {/* Configured Servers List */}
            {configuredServers.length > 0 && (
                <ScrollArea className="h-32">
                    <div className="space-y-2">
                        {configuredServers.map((server, index) => (
                            <div
                                key={index}
                                className="flex items-center justify-between p-2 bg-muted/50 rounded text-xs"
                            >
                                <div>
                                    <p className="font-medium">{server.name}</p>
                                    <p className="text-muted-foreground truncate max-w-[180px]">{server.url}</p>
                                </div>
                                <Button
                                    size="icon"
                                    variant="ghost"
                                    className="h-6 w-6"
                                    onClick={() => removeServer(index)}
                                >
                                    <Trash2 className="h-3 w-3 text-red-500" />
                                </Button>
                            </div>
                        ))}
                    </div>
                </ScrollArea>
            )}

            {/* Connect/Disconnect Button */}
            <div className="flex gap-2">
                {isConnected ? (
                    <Button
                        size="sm"
                        variant="destructive"
                        className="flex-1 h-8 text-xs"
                        onClick={disconnect}
                    >
                        Disconnect
                    </Button>
                ) : (
                    <Button
                        size="sm"
                        className="flex-1 h-8 text-xs"
                        onClick={handleConnect}
                        disabled={configuredServers.length === 0 || isConnecting}
                    >
                        {isConnecting ? (
                            <>
                                <Loader2 className="h-3 w-3 mr-1 animate-spin" /> Connecting...
                            </>
                        ) : (
                            'Connect'
                        )}
                    </Button>
                )}
            </div>

            {/* Available Tools */}
            {isConnected && tools.length > 0 && (
                <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">Available Tools ({tools.length})</p>
                    <ScrollArea className="h-24">
                        <div className="space-y-1">
                            {tools.map((tool, index) => (
                                <div
                                    key={index}
                                    className="text-xs p-1.5 bg-muted/30 rounded"
                                >
                                    <code className="font-mono">{tool.serverName}:{tool.name}</code>
                                </div>
                            ))}
                        </div>
                    </ScrollArea>
                </div>
            )}
        </div>
    )
}
