/**
 * TrackView Component - Scholar tracking with progress display
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import { client } from '../utils/api.js';

interface TrackViewProps {
  scholarId?: string;
}

interface TrackProgress {
  phase: string;
  message: string;
  percentage?: number;
}

interface Paper {
  title: string;
  year: number;
  citations: number;
  venue?: string;
}

interface TrackResult {
  scholarName: string;
  papers: Paper[];
  influenceScore: number;
}

export const TrackView: React.FC<TrackViewProps> = ({ scholarId: initialScholarId }) => {
  const { exit } = useApp();
  const [scholarInput, setScholarInput] = useState(initialScholarId || '');
  const [isTracking, setIsTracking] = useState(false);
  const [progress, setProgress] = useState<TrackProgress | null>(null);
  const [result, setResult] = useState<TrackResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
    }
  });

  // Auto-start if scholarId provided
  useEffect(() => {
    if (initialScholarId) {
      startTracking(initialScholarId);
    }
  }, [initialScholarId]);

  const startTracking = async (scholarIdOrName: string) => {
    setIsTracking(true);
    setProgress(null);
    setResult(null);
    setError(null);

    try {
      const isId = /^\d+$/.test(scholarIdOrName);

      for await (const event of client.trackScholar({
        scholarId: isId ? scholarIdOrName : undefined,
        scholarName: isId ? undefined : scholarIdOrName,
      })) {
        if (event.type === 'progress') {
          const data = event.data as TrackProgress;
          setProgress(data);
        } else if (event.type === 'result') {
          setResult(event.data as TrackResult);
        } else if (event.type === 'error') {
          setError(event.message || 'Unknown error');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsTracking(false);
    }
  };

  const handleSubmit = (value: string) => {
    if (value.trim()) {
      startTracking(value.trim());
    }
  };

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color="cyan" bold>
        Scholar Tracking
      </Text>

      {/* Input */}
      {!isTracking && !result && (
        <Box marginY={1} flexDirection="column">
          <Text color="gray">Enter scholar name or Semantic Scholar ID:</Text>
          <Box borderStyle="round" borderColor="cyan" paddingX={1} marginTop={1}>
            <Text color="cyan">{'> '}</Text>
            <TextInput
              value={scholarInput}
              onChange={setScholarInput}
              onSubmit={handleSubmit}
              placeholder="e.g., Dawn Song or 1741101"
            />
          </Box>
        </Box>
      )}

      {/* Progress */}
      {isTracking && progress && (
        <Box marginY={1} flexDirection="column">
          <Box>
            <Text color="yellow">
              <Spinner type="dots" />
            </Text>
            <Text color="yellow"> {progress.phase}</Text>
          </Box>
          <Box marginLeft={2}>
            <Text color="gray">{progress.message}</Text>
          </Box>
          {progress.percentage !== undefined && (
            <Box marginTop={1}>
              <ProgressBar percentage={progress.percentage} />
            </Box>
          )}
        </Box>
      )}

      {/* Result */}
      {result && (
        <Box marginY={1} flexDirection="column">
          <Text color="green" bold>
            ✓ Tracking Complete
          </Text>

          <Box marginTop={1} flexDirection="column">
            <Text color="cyan" bold>
              {result.scholarName}
            </Text>
            <Text color="gray">
              Influence Score: {result.influenceScore.toFixed(1)}
            </Text>
          </Box>

          <Box marginTop={1} flexDirection="column">
            <Text bold>Recent Papers:</Text>
            {result.papers.slice(0, 5).map((paper, index) => (
              <Box key={index} marginLeft={2} marginTop={1}>
                <Text color="white">• {paper.title}</Text>
                <Text color="gray">
                  {' '}
                  ({paper.year}, {paper.citations} citations)
                </Text>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      {/* Error */}
      {error && (
        <Box marginY={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      <Box marginTop={1}>
        <Text color="gray">Press Ctrl+C to exit</Text>
      </Box>
    </Box>
  );
};

interface ProgressBarProps {
  percentage: number;
  width?: number;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ percentage, width = 30 }) => {
  const filled = Math.round((percentage / 100) * width);
  const empty = width - filled;

  return (
    <Box>
      <Text color="gray">[</Text>
      <Text color="green">{'█'.repeat(filled)}</Text>
      <Text color="gray">{'░'.repeat(empty)}</Text>
      <Text color="gray">] </Text>
      <Text color="cyan">{percentage.toFixed(0)}%</Text>
    </Box>
  );
};
