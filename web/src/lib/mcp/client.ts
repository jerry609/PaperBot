/**
 * MCP (Model Context Protocol) Client for DeepCode Studio
 * 
 * Connects to MCP servers to expose their tools to the AI.
 * Reference: https://modelcontextprotocol.io/
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';

export interface MCPServerConfig {
    name: string;
    url: string;
    description?: string;
}

export interface MCPTool {
    name: string;
    description: string;
    serverName: string;
    inputSchema: Record<string, unknown>;
}

export interface MCPToolResult {
    success: boolean;
    content: unknown;
    error?: string;
}

/**
 * MCP Client Manager
 * Manages connections to multiple MCP servers
 */
export class MCPClientManager {
    private clients: Map<string, Client> = new Map();
    private serverConfigs: MCPServerConfig[] = [];
    private tools: MCPTool[] = [];

    constructor(configs: MCPServerConfig[]) {
        this.serverConfigs = configs;
    }

    /**
     * Connect to all configured MCP servers
     */
    async connectAll(): Promise<void> {
        for (const config of this.serverConfigs) {
            try {
                await this.connect(config);
            } catch (error) {
                console.error(`Failed to connect to MCP server ${config.name}:`, error);
            }
        }
    }

    /**
     * Connect to a single MCP server
     */
    async connect(config: MCPServerConfig): Promise<void> {
        const transport = new SSEClientTransport(new URL(config.url));
        const client = new Client({
            name: 'deepcode-studio',
            version: '1.0.0',
        });

        await client.connect(transport);

        // Fetch available tools from the server
        const toolsResponse = await client.listTools();

        for (const tool of toolsResponse.tools) {
            this.tools.push({
                name: tool.name,
                description: tool.description || '',
                serverName: config.name,
                inputSchema: tool.inputSchema as Record<string, unknown>,
            });
        }

        this.clients.set(config.name, client);
        console.log(`Connected to MCP server: ${config.name}, tools: ${toolsResponse.tools.length}`);
    }

    /**
     * Get all available tools from all connected servers
     */
    getTools(): MCPTool[] {
        return this.tools;
    }

    /**
     * Call a tool on an MCP server
     */
    async callTool(serverName: string, toolName: string, args: Record<string, unknown>): Promise<MCPToolResult> {
        const client = this.clients.get(serverName);
        if (!client) {
            return {
                success: false,
                content: null,
                error: `Server ${serverName} not connected`,
            };
        }

        try {
            const result = await client.callTool({ name: toolName, arguments: args });
            return {
                success: true,
                content: result.content,
            };
        } catch (error) {
            return {
                success: false,
                content: null,
                error: error instanceof Error ? error.message : 'Unknown error',
            };
        }
    }

    /**
     * Disconnect from all servers
     */
    async disconnectAll(): Promise<void> {
        for (const [name, client] of this.clients) {
            try {
                await client.close();
                console.log(`Disconnected from MCP server: ${name}`);
            } catch (error) {
                console.error(`Failed to disconnect from ${name}:`, error);
            }
        }
        this.clients.clear();
        this.tools = [];
    }
}

// Default MCP server configurations
// Users can configure these in settings
export const DEFAULT_MCP_SERVERS: MCPServerConfig[] = [
    // Example: Local filesystem server
    // {
    //   name: 'filesystem',
    //   url: 'http://localhost:3001/mcp',
    //   description: 'Access to local filesystem',
    // },
    // Example: GitHub server
    // {
    //   name: 'github',
    //   url: 'http://localhost:3002/mcp',
    //   description: 'Access to GitHub repositories',
    // },
];

// Singleton instance
let mcpManager: MCPClientManager | null = null;

export function getMCPManager(): MCPClientManager {
    if (!mcpManager) {
        mcpManager = new MCPClientManager(DEFAULT_MCP_SERVERS);
    }
    return mcpManager;
}

/**
 * Initialize MCP connections
 * Call this on app startup
 */
export async function initializeMCP(): Promise<MCPClientManager> {
    const manager = getMCPManager();
    await manager.connectAll();
    return manager;
}
