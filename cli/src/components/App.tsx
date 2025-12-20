/**
 * Main App Component - Routes to different views based on command
 */

import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { ChatView } from './ChatView.js';
import { TrackView } from './TrackView.js';
import { AnalyzeView } from './AnalyzeView.js';
import { GenCodeView } from './GenCodeView.js';
import { StatusBar } from './StatusBar.js';
import { client } from '../utils/api.js';

export interface AppFlags {
  scholar?: string;
  title?: string;
  doi?: string;
  abstract?: string;
  output?: string;
  stream?: boolean;
}

interface AppProps {
  command: string;
  flags: AppFlags;
}

type ConnectionStatus = 'checking' | 'connected' | 'disconnected';

export const App: React.FC<AppProps> = ({ command, flags }) => {
  const [status, setStatus] = useState<ConnectionStatus>('checking');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      const healthy = await client.health();
      setStatus(healthy ? 'connected' : 'disconnected');
      if (!healthy) {
        setError('Cannot connect to PaperBot API. Make sure the server is running.');
      }
    };

    checkConnection();
  }, []);

  const renderView = () => {
    switch (command) {
      case 'track':
        return <TrackView scholarId={flags.scholar} />;
      case 'analyze':
        return <AnalyzeView title={flags.title} doi={flags.doi} />;
      case 'gen-code':
        return (
          <GenCodeView
            title={flags.title}
            abstract={flags.abstract}
            outputDir={flags.output}
          />
        );
      case 'review':
        return <AnalyzeView title={flags.title} doi={flags.doi} mode="review" />;
      case 'chat':
      default:
        return <ChatView />;
    }
  };

  return (
    <Box flexDirection="column" padding={1}>
      <StatusBar status={status} command={command} />

      {status === 'checking' && (
        <Box marginY={1}>
          <Text color="yellow">Connecting to PaperBot API...</Text>
        </Box>
      )}

      {status === 'disconnected' && error && (
        <Box marginY={1} flexDirection="column">
          <Text color="red">Error: {error}</Text>
          <Text color="gray">
            Start the API server with: python -m uvicorn src.paperbot.api.main:app --reload
          </Text>
        </Box>
      )}

      {status === 'connected' && renderView()}
    </Box>
  );
};
