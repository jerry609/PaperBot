import { google } from '@ai-sdk/google';
import { anthropic } from '@ai-sdk/anthropic';
import { streamText, convertToModelMessages, UIMessage, tool } from 'ai';
import { AVAILABLE_MODELS, DEFAULT_MODEL } from '@/lib/models';
import {
    TOOLS,
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

// Build AI SDK tools from our tool definitions
const aiTools = {
    search_codebase: tool({
        description: TOOLS.search_codebase.description,
        parameters: searchCodebaseSchema,
        execute: async (params) => {
            const result = await TOOLS.search_codebase.execute(params);
            return result;
        },
    }),

    read_file: tool({
        description: TOOLS.read_file.description,
        parameters: readFileSchema,
        execute: async (params) => {
            const result = await TOOLS.read_file.execute(params);
            return result;
        },
    }),

    write_file: tool({
        description: TOOLS.write_file.description,
        parameters: writeFileSchema,
        execute: async (params) => {
            const result = await TOOLS.write_file.execute(params);
            return result;
        },
    }),

    edit_file: tool({
        description: TOOLS.edit_file.description,
        parameters: editFileSchema,
        execute: async (params) => {
            const result = await TOOLS.edit_file.execute(params);
            return result;
        },
    }),

    list_files: tool({
        description: TOOLS.list_files.description,
        parameters: listFilesSchema,
        execute: async (params) => {
            const result = await TOOLS.list_files.execute(params);
            return result;
        },
    }),

    run_command: tool({
        description: TOOLS.run_command.description,
        parameters: runCommandSchema,
        execute: async (params) => {
            const result = await TOOLS.run_command.execute(params);
            return result;
        },
    }),
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
