/**
 * StatusBar Component - Shows connection status and current command
 */

import React from 'react';
import { Box, Text } from 'ink';

interface StatusBarProps {
  status: 'checking' | 'connected' | 'disconnected';
  command: string;
}

export const StatusBar: React.FC<StatusBarProps> = ({ status, command }) => {
  const statusColor = {
    checking: 'yellow',
    connected: 'green',
    disconnected: 'red',
  }[status] as 'yellow' | 'green' | 'red';

  const statusIcon = {
    checking: '○',
    connected: '●',
    disconnected: '○',
  }[status];

  const commandLabels: Record<string, string> = {
    chat: 'Interactive Chat',
    track: 'Scholar Tracking',
    analyze: 'Paper Analysis',
    'gen-code': 'Paper2Code',
    review: 'Deep Review',
  };

  return (
    <Box
      borderStyle="single"
      borderColor="gray"
      paddingX={1}
      justifyContent="space-between"
    >
      <Box>
        <Text color="cyan" bold>
          PaperBot
        </Text>
        <Text color="gray"> │ </Text>
        <Text>{commandLabels[command] || command}</Text>
      </Box>
      <Box>
        <Text color={statusColor}>
          {statusIcon} {status}
        </Text>
      </Box>
    </Box>
  );
};
