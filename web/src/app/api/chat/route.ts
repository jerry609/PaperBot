import { google } from '@ai-sdk/google';
import { anthropic } from '@ai-sdk/anthropic';
import { streamText, convertToModelMessages, UIMessage } from 'ai';
import { z } from 'zod';
import { AVAILABLE_MODELS, DEFAULT_MODEL } from '@/lib/models';
import {
    searchCodebase,
    readFile,
    writeFile,
    editFile,
    listFiles,
    runCommand,
    searchCodebaseSchema,
    readFileSchema,
    writeFileSchema,
    editFileSchema,
    listFilesSchema,
    runCommandSchema,
} from '@/lib/tools';

export const maxDuration = 60;

const SYSTEM_PROMPT = `You are an expert AI coding assistant for PaperBot DeepCode Studio.
You help users reproduce research papers by generating and debugging code.

You have access to the following tools:
- search_codebase: Search for relevant code in the project
- read_file: Read file contents
- write_file: Create or overwrite files
- edit_file: Make targeted edits to files
- list_files: List files in a directory
- run_command: Execute shell commands

When working on a task:
1. First analyze what needs to be done
2. Search or list files to understand the codebase
3. Read relevant files for context
4. Create or edit files as needed
5. Run commands to test your changes

Always explain your reasoning before using a tool.
Use structured tool calls, and report results clearly.`;

function getModel(modelId: string) {
    const config = AVAILABLE_MODELS.find(m => m.id === modelId);
    if (!config) {
        return google(DEFAULT_MODEL);
    }

    switch (config.provider) {
        case 'google':
            return google(config.id);
        case 'anthropic':
            return anthropic(config.id);
        case 'ollama':
            return google(DEFAULT_MODEL);
        default:
            return google(DEFAULT_MODEL);
    }
}

// Define tools using the AI SDK v5 interface
const aiTools = {
    search_codebase: {
        description: 'Search the codebase for relevant code snippets or files',
        inputSchema: searchCodebaseSchema,
        execute: searchCodebase,
    },
    read_file: {
        description: 'Read the contents of a file',
        inputSchema: readFileSchema,
        execute: readFile,
    },
    write_file: {
        description: 'Create or overwrite a file with new content',
        inputSchema: writeFileSchema,
        execute: writeFile,
    },
    edit_file: {
        description: 'Edit a file by finding and replacing content',
        inputSchema: editFileSchema,
        execute: editFile,
    },
    list_files: {
        description: 'List files and directories in a path',
        inputSchema: listFilesSchema,
        execute: listFiles,
    },
    run_command: {
        description: 'Execute a shell command',
        inputSchema: runCommandSchema,
        execute: runCommand,
    },
};

export async function POST(req: Request) {
    const { messages, model: modelId = DEFAULT_MODEL }: { messages: UIMessage[], model?: string } = await req.json();

    const result = streamText({
        model: getModel(modelId),
        system: SYSTEM_PROMPT,
        messages: convertToModelMessages(messages),
        tools: aiTools,
    });

    return result.toUIMessageStreamResponse();
}
