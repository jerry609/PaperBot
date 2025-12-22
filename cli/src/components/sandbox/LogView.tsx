/**
 * LogView Component - Real-time log and metrics display
 *
 * Features:
 * - Real-time log streaming
 * - Resource usage display
 * - Cancel/Retry actions
 */

import React, { useState, useEffect, useRef } from 'react';
import { Box, Text, useInput, useApp } from 'ink';
import Spinner from 'ink-spinner';
import { client } from '../../utils/api.js';

interface LogEntry {
  ts: string;
  level: string;
  message: string;
  source: string;
}

interface ResourceMetrics {
  cpu_percent: number;
  memory_mb: number;
  memory_limit_mb: number;
  elapsed_seconds: number;
  timeout_seconds: number;
  status: string;
}

interface LogsResponse {
  logs?: LogEntry[];
}

interface MetricsResponse extends ResourceMetrics {
  error?: string;
}

interface LogViewProps {
  runId: string;
  onBack: () => void;
}

export const LogView: React.FC<LogViewProps> = ({ runId, onBack }) => {
  const { exit } = useApp();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [scrollOffset, setScrollOffset] = useState(0);
  const maxVisibleLogs = 15;

  // Fetch initial logs
  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const data = await client.fetchJson<LogsResponse>(`/api/sandbox/runs/${runId}/logs?limit=100`);
        setLogs(data.logs || []);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch logs');
        setLoading(false);
      }
    };

    fetchLogs();
  }, [runId]);

  // Poll for new logs (SSE not easily supported in Ink)
  useEffect(() => {
    if (!streaming) return;

    const pollLogs = async () => {
      try {
        const data = await client.fetchJson<LogsResponse>(`/api/sandbox/runs/${runId}/logs?limit=200`);
        const newLogs = data.logs || [];
        if (newLogs.length > logs.length) {
          setLogs(newLogs);
          if (autoScroll) {
            setScrollOffset(Math.max(0, newLogs.length - maxVisibleLogs));
          }
        }
      } catch {
        // Ignore polling errors
      }
    };

    const interval = setInterval(pollLogs, 1000);
    return () => clearInterval(interval);
  }, [runId, streaming, logs.length, autoScroll]);

  // Poll for metrics
  useEffect(() => {
    const pollMetrics = async () => {
      try {
        const data = await client.fetchJson<MetricsResponse>(`/api/sandbox/runs/${runId}/metrics`);
        if (data && !data.error) {
          setMetrics(data);
          // Stop streaming if completed
          if (data.status !== 'running') {
            setStreaming(false);
          }
        }
      } catch {
        // Ignore errors
      }
    };

    pollMetrics();
    const interval = setInterval(pollMetrics, 2000);
    return () => clearInterval(interval);
  }, [runId]);

  // Keyboard controls
  useInput(async (input, key) => {
    // Back
    if (key.escape || input === 'q') {
      onBack();
    }

    // Exit
    if (key.ctrl && input === 'c') {
      exit();
    }

    // Scroll
    if (input === 'j' || key.downArrow) {
      setAutoScroll(false);
      setScrollOffset(Math.min(scrollOffset + 1, Math.max(0, logs.length - maxVisibleLogs)));
    }
    if (input === 'k' || key.upArrow) {
      setAutoScroll(false);
      setScrollOffset(Math.max(scrollOffset - 1, 0));
    }

    // Page scroll
    if (key.pageDown) {
      setAutoScroll(false);
      setScrollOffset(Math.min(scrollOffset + maxVisibleLogs, Math.max(0, logs.length - maxVisibleLogs)));
    }
    if (key.pageUp) {
      setAutoScroll(false);
      setScrollOffset(Math.max(scrollOffset - maxVisibleLogs, 0));
    }

    // Jump to end
    if (input === 'G') {
      setAutoScroll(true);
      setScrollOffset(Math.max(0, logs.length - maxVisibleLogs));
    }

    // Jump to start
    if (input === 'g') {
      setAutoScroll(false);
      setScrollOffset(0);
    }

    // Cancel job
    if (input === 'c' && metrics?.status === 'running') {
      try {
        await client.fetchJson(`/api/sandbox/jobs/${runId}/cancel`, { method: 'POST' });
      } catch {
        // Ignore
      }
    }
  });

  const visibleLogs = logs.slice(scrollOffset, scrollOffset + maxVisibleLogs);

  if (loading) {
    return (
      <Box>
        <Text color="yellow">
          <Spinner type="dots" />
        </Text>
        <Text> Loading logs...</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      {/* Header */}
      <Box justifyContent="space-between" marginBottom={1}>
        <Box>
          <Text color="cyan" bold>
            Run: {runId.slice(0, 16)}...
          </Text>
          {streaming && (
            <Text color="yellow">
              {' '}
              <Spinner type="dots" /> Live
            </Text>
          )}
        </Box>
        {metrics && (
          <Text color={metrics.status === 'running' ? 'cyan' : metrics.status === 'completed' ? 'green' : 'red'}>
            {metrics.status.toUpperCase()}
          </Text>
        )}
      </Box>

      {/* Resource Metrics */}
      {metrics && (
        <Box marginBottom={1} flexDirection="column">
          <Box>
            <Text color="gray">CPU: </Text>
            <ProgressBar value={metrics.cpu_percent} max={100} width={15} color="cyan" />
            <Text color="gray"> {metrics.cpu_percent.toFixed(1)}%</Text>
            <Text color="gray">  Memory: </Text>
            <ProgressBar
              value={metrics.memory_mb}
              max={metrics.memory_limit_mb}
              width={15}
              color="magenta"
            />
            <Text color="gray">
              {' '}
              {metrics.memory_mb.toFixed(0)}/{metrics.memory_limit_mb}MB
            </Text>
          </Box>
          <Box>
            <Text color="gray">Time: </Text>
            <ProgressBar
              value={metrics.elapsed_seconds}
              max={metrics.timeout_seconds}
              width={15}
              color={metrics.elapsed_seconds > metrics.timeout_seconds * 0.8 ? 'red' : 'green'}
            />
            <Text color="gray">
              {' '}
              {formatDuration(metrics.elapsed_seconds)} / {formatDuration(metrics.timeout_seconds)}
            </Text>
          </Box>
        </Box>
      )}

      {/* Log Area */}
      <Box
        flexDirection="column"
        borderStyle="single"
        borderColor="gray"
        paddingX={1}
        height={maxVisibleLogs + 2}
      >
        {visibleLogs.length === 0 ? (
          <Text color="gray">No logs yet...</Text>
        ) : (
          visibleLogs.map((log, index) => (
            <LogLine key={`${scrollOffset + index}-${log.ts}`} log={log} />
          ))
        )}
      </Box>

      {/* Scroll indicator */}
      <Box justifyContent="space-between">
        <Text color="gray">
          {logs.length > 0 && (
            <>
              Lines {scrollOffset + 1}-{Math.min(scrollOffset + maxVisibleLogs, logs.length)} of{' '}
              {logs.length}
            </>
          )}
        </Text>
        <Text color={autoScroll ? 'green' : 'gray'}>{autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}</Text>
      </Box>

      {/* Error */}
      {error && (
        <Box marginTop={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      {/* Help */}
      <Box marginTop={1}>
        <Text color="gray">
          j/k: scroll | g/G: top/bottom | q: back |{' '}
          {metrics?.status === 'running' ? 'c: cancel' : ''}
        </Text>
      </Box>
    </Box>
  );
};

// Log Line Component
interface LogLineProps {
  log: LogEntry;
}

const LogLine: React.FC<LogLineProps> = ({ log }) => {
  const getLevelColor = () => {
    switch (log.level.toLowerCase()) {
      case 'error':
        return 'red';
      case 'warning':
        return 'yellow';
      case 'info':
        return 'green';
      case 'debug':
        return 'gray';
      default:
        return 'white';
    }
  };

  const getSourceIcon = () => {
    switch (log.source) {
      case 'stdout':
        return '>';
      case 'stderr':
        return '!';
      case 'executor':
        return '*';
      default:
        return '-';
    }
  };

  const formatTime = (ts: string) => {
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString();
    } catch {
      return '';
    }
  };

  return (
    <Box>
      <Text color="gray">{formatTime(log.ts)} </Text>
      <Text color={log.source === 'stderr' ? 'red' : 'gray'}>{getSourceIcon()} </Text>
      <Text color={getLevelColor()}>
        {log.message.slice(0, 80)}
        {log.message.length > 80 ? '...' : ''}
      </Text>
    </Box>
  );
};

// Progress Bar Component
interface ProgressBarProps {
  value: number;
  max: number;
  width: number;
  color: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ value, max, width, color }) => {
  const percentage = Math.min((value / max) * 100, 100);
  const filled = Math.round((percentage / 100) * width);
  const empty = width - filled;

  return (
    <Box>
      <Text color="gray">[</Text>
      <Text color={color}>{'█'.repeat(filled)}</Text>
      <Text color="gray">{'░'.repeat(empty)}</Text>
      <Text color="gray">]</Text>
    </Box>
  );
};

// Format duration in seconds to human readable
const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${minutes}m${secs}s`;
};
