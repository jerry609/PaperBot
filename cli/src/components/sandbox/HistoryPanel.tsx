/**
 * HistoryPanel Component - Displays run history
 *
 * Features:
 * - Paginated run list
 * - Status filtering
 * - Event timeline view
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import Spinner from 'ink-spinner';
import { client } from '../../utils/api.js';

interface RunInfo {
  run_id: string;
  workflow: string;
  started_at?: string;
  ended_at?: string;
  status: string;
  metadata?: Record<string, unknown>;
}

interface HistoryPanelProps {
  onViewLogs: (runId: string) => void;
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({ onViewLogs }) => {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const pageSize = 10;

  // Fetch runs
  const fetchRuns = async () => {
    try {
      let url = `/api/runs?limit=${pageSize}&offset=${page * pageSize}`;
      if (statusFilter) {
        url += `&status=${statusFilter}`;
      }
      const data = await client.fetchJson(url);
      setRuns(data.runs || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch runs');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, [page, statusFilter]);

  // Keyboard navigation
  useInput((input, key) => {
    // List navigation
    if (input === 'j' || key.downArrow) {
      setSelectedIndex(Math.min(selectedIndex + 1, runs.length - 1));
    }
    if (input === 'k' || key.upArrow) {
      setSelectedIndex(Math.max(selectedIndex - 1, 0));
    }

    // Pagination
    if (input === 'n' || key.pageDown) {
      setPage(page + 1);
      setSelectedIndex(0);
    }
    if (input === 'p' || key.pageUp) {
      if (page > 0) {
        setPage(page - 1);
        setSelectedIndex(0);
      }
    }

    // Status filtering
    if (input === 'f') {
      // Cycle through filters
      const filters = [null, 'running', 'completed', 'failed'];
      const currentIdx = filters.indexOf(statusFilter);
      setStatusFilter(filters[(currentIdx + 1) % filters.length]);
      setPage(0);
      setSelectedIndex(0);
    }

    // View details
    if (key.return) {
      const selectedRun = runs[selectedIndex];
      if (selectedRun) {
        onViewLogs(selectedRun.run_id);
      }
    }

    // Refresh
    if (input === 'R') {
      setLoading(true);
      fetchRuns();
    }
  });

  if (loading) {
    return (
      <Box>
        <Text color="yellow">
          <Spinner type="dots" />
        </Text>
        <Text> Loading history...</Text>
      </Box>
    );
  }

  if (error) {
    return (
      <Box>
        <Text color="red">Error: {error}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      {/* Filter Bar */}
      <Box marginBottom={1}>
        <Text color="gray">Filter: </Text>
        <Text color={statusFilter === null ? 'cyan' : 'gray'}>All</Text>
        <Text color="gray"> | </Text>
        <Text color={statusFilter === 'running' ? 'cyan' : 'gray'}>Running</Text>
        <Text color="gray"> | </Text>
        <Text color={statusFilter === 'completed' ? 'cyan' : 'gray'}>Completed</Text>
        <Text color="gray"> | </Text>
        <Text color={statusFilter === 'failed' ? 'cyan' : 'gray'}>Failed</Text>
        <Text color="gray">  (f to cycle)</Text>
      </Box>

      {/* Run List */}
      <Box flexDirection="column" borderStyle="single" borderColor="gray" paddingX={1}>
        {runs.length === 0 ? (
          <Text color="gray">No runs found</Text>
        ) : (
          runs.map((run, index) => (
            <RunRow key={run.run_id} run={run} selected={index === selectedIndex} />
          ))
        )}
      </Box>

      {/* Pagination */}
      <Box marginTop={1} justifyContent="space-between">
        <Text color="gray">
          Page {page + 1} | {runs.length} runs
        </Text>
        <Text color="gray">n: next | p: prev | R: refresh</Text>
      </Box>

      {/* Help hint */}
      <Box marginTop={1}>
        <Text color="gray">j/k: navigate | Enter: view events | f: filter</Text>
      </Box>
    </Box>
  );
};

// Run Row Component
interface RunRowProps {
  run: RunInfo;
  selected: boolean;
}

const RunRow: React.FC<RunRowProps> = ({ run, selected }) => {
  const getStatusIcon = () => {
    switch (run.status) {
      case 'running':
        return '◐';
      case 'completed':
        return '●';
      case 'failed':
        return '✗';
      default:
        return '○';
    }
  };

  const getStatusColor = () => {
    switch (run.status) {
      case 'running':
        return 'cyan';
      case 'completed':
        return 'green';
      case 'failed':
        return 'red';
      default:
        return 'gray';
    }
  };

  const formatTime = (ts?: string) => {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      return date.toLocaleString();
    } catch {
      return '';
    }
  };

  const duration = () => {
    if (!run.started_at) return '';
    const start = new Date(run.started_at);
    const end = run.ended_at ? new Date(run.ended_at) : new Date();
    const seconds = Math.round((end.getTime() - start.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.round(seconds / 60);
    return `${minutes}m`;
  };

  return (
    <Box>
      <Text color={selected ? 'white' : 'gray'}>{selected ? '>' : ' '} </Text>
      <Text color={getStatusColor()}>{getStatusIcon()} </Text>
      <Text color={selected ? 'white' : 'gray'}>{run.run_id.slice(0, 12)}...</Text>
      <Text color="gray"> | </Text>
      <Text color={selected ? 'cyan' : 'gray'}>{run.workflow || 'unknown'}</Text>
      <Text color="gray"> | </Text>
      <Text color="gray">{formatTime(run.started_at)}</Text>
      {run.started_at && (
        <>
          <Text color="gray"> | </Text>
          <Text color={getStatusColor()}>{duration()}</Text>
        </>
      )}
    </Box>
  );
};
