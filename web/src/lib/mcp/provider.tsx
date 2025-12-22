"use client"

import React, { createContext, useContext, useState, useEffect } from 'react'
import { MCPClientManager, MCPTool, MCPServerConfig, getMCPManager } from './client'

interface MCPContextValue {
    isConnected: boolean;
    isConnecting: boolean;
    tools: MCPTool[];
    servers: MCPServerConfig[];
    error: string | null;
    connect: (configs: MCPServerConfig[]) => Promise<void>;
    disconnect: () => Promise<void>;
    callTool: (serverName: string, toolName: string, args: Record<string, unknown>) => Promise<unknown>;
}

const MCPContext = createContext<MCPContextValue | null>(null);

export function MCPProvider({ children }: { children: React.ReactNode }) {
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);
    const [tools, setTools] = useState<MCPTool[]>([]);
    const [servers, setServers] = useState<MCPServerConfig[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [manager, setManager] = useState<MCPClientManager | null>(null);

    const connect = async (configs: MCPServerConfig[]) => {
        setIsConnecting(true);
        setError(null);

        try {
            const newManager = new MCPClientManager(configs);
            await newManager.connectAll();

            setManager(newManager);
            setTools(newManager.getTools());
            setServers(configs);
            setIsConnected(true);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to connect to MCP servers');
            setIsConnected(false);
        } finally {
            setIsConnecting(false);
        }
    };

    const disconnect = async () => {
        if (manager) {
            await manager.disconnectAll();
            setManager(null);
            setTools([]);
            setIsConnected(false);
        }
    };

    const callTool = async (serverName: string, toolName: string, args: Record<string, unknown>) => {
        if (!manager) {
            throw new Error('MCP not connected');
        }

        const result = await manager.callTool(serverName, toolName, args);
        if (!result.success) {
            throw new Error(result.error || 'Tool call failed');
        }
        return result.content;
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (manager) {
                manager.disconnectAll();
            }
        };
    }, [manager]);

    return (
        <MCPContext.Provider value={{
            isConnected,
            isConnecting,
            tools,
            servers,
            error,
            connect,
            disconnect,
            callTool,
        }}>
            {children}
        </MCPContext.Provider>
    );
}

export function useMCP() {
    const context = useContext(MCPContext);
    if (!context) {
        throw new Error('useMCP must be used within an MCPProvider');
    }
    return context;
}
