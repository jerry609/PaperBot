/**
 * ChatView Component - Interactive chat interface with streaming AI responses
 */

import React, { useState, useCallback } from 'react';
import { Box, Text, useInput, useApp } from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import { client, StreamEvent } from '../utils/api.js';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export const ChatView: React.FC = () => {
  const { exit } = useApp();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'system',
      content: 'Welcome to PaperBot! Ask me about papers, scholars, or research topics.',
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');

  useInput((inputKey, key) => {
    if (key.ctrl && inputKey === 'c') {
      exit();
    }
  });

  const handleSubmit = useCallback(async (value: string) => {
    if (!value.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: value };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingContent('');

    try {
      const history = messages
        .filter((m) => m.role !== 'system')
        .map((m) => ({ role: m.role, content: m.content }));

      let fullContent = '';

      for await (const event of client.chat(value, history)) {
        if (event.type === 'progress' || event.type === 'result') {
          const data = event.data as { content?: string; delta?: string };
          if (data.delta) {
            fullContent += data.delta;
            setStreamingContent(fullContent);
          } else if (data.content) {
            fullContent = data.content;
            setStreamingContent(fullContent);
          }
        } else if (event.type === 'error') {
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: `Error: ${event.message}` },
          ]);
          break;
        } else if (event.type === 'done') {
          if (fullContent) {
            setMessages((prev) => [
              ...prev,
              { role: 'assistant', content: fullContent },
            ]);
          }
          break;
        }
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        },
      ]);
    } finally {
      setIsLoading(false);
      setStreamingContent('');
    }
  }, [messages, isLoading]);

  return (
    <Box flexDirection="column" marginTop={1}>
      {/* Message History */}
      <Box flexDirection="column" marginBottom={1}>
        {messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}

        {/* Streaming Response */}
        {isLoading && streamingContent && (
          <Box marginY={1}>
            <Text color="green" bold>
              Assistant:{' '}
            </Text>
            <Text>{streamingContent}</Text>
            <Text color="gray"> ▌</Text>
          </Box>
        )}

        {/* Loading Indicator */}
        {isLoading && !streamingContent && (
          <Box marginY={1}>
            <Text color="cyan">
              <Spinner type="dots" />
            </Text>
            <Text color="gray"> Thinking...</Text>
          </Box>
        )}
      </Box>

      {/* Input */}
      <Box borderStyle="round" borderColor="cyan" paddingX={1}>
        <Text color="cyan" bold>
          {'> '}
        </Text>
        <TextInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          placeholder="Ask about papers, scholars, or research..."
        />
      </Box>

      {/* Help */}
      <Box marginTop={1}>
        <Text color="gray">
          Press Ctrl+C to exit • Type your question and press Enter
        </Text>
      </Box>
    </Box>
  );
};

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const roleColors = {
    user: 'blue',
    assistant: 'green',
    system: 'gray',
  } as const;

  const roleLabels = {
    user: 'You',
    assistant: 'Assistant',
    system: 'System',
  };

  return (
    <Box marginY={1} flexDirection="column">
      <Text color={roleColors[message.role]} bold>
        {roleLabels[message.role]}:
      </Text>
      <Box marginLeft={2}>
        <Text wrap="wrap">{message.content}</Text>
      </Box>
    </Box>
  );
};
