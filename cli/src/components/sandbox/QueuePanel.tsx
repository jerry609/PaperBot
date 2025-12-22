/**
 * QueuePanel Component - Displays task queue status
 *
 * Features:
 * - Pending/Running/Completed job lists
 * - j/k navigation
 * - Cancel/Retry actions
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import Spinner from 'ink-spinner';
import { client } from '../../utils/api.js';

interface JobInfo {
  job_id: string;
  function: string;
  status: string;
  enqueue_time?: string;
  start_time?: string;
  finish_time?: string;
}

interface QueueData {
  pending: JobInfo[];
  running: JobInfo[];
  completed: JobInfo[];
  stats: {
    total_pending: number;
    total_running: number;
    redis_connected: boolean;
  };
}

interface QueuePanelProps {
  onViewLogs: (runId: string) => void;
}

export const QueuePanel: React.FC<QueuePanelProps> = ({ onViewLogs }) => {
  const [queue, setQueue] = useState<QueueData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [activeSection, setActiveSection] = useState<'pending' | 'running' | 'completed'>('running');
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  // Fetch queue status
  const fetchQueue = async () => {
    try {
      const data = await client.fetchJson('/api/sandbox/queue');
      setQueue(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch queue');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQueue();
    const interval = setInterval(fetchQueue, 3000); // Refresh every 3 seconds
    return () => clearInterval(interval);
  }, []);

  // Get current list based on active section
  const getCurrentList = (): JobInfo[] => {
    if (!queue) return [];
    switch (activeSection) {
      case 'pending':
        return queue.pending;
      case 'running':
        return queue.running;
      case 'completed':
        return queue.completed;
    }
  };

  const currentList = getCurrentList();

  // Keyboard navigation
  useInput(async (input, key) => {
    // Section switching with left/right or h/l
    if (input === 'h' || key.leftArrow) {
      const sections: Array<'pending' | 'running' | 'completed'> = ['pending', 'running', 'completed'];
      const idx = sections.indexOf(activeSection);
      if (idx > 0) {
        setActiveSection(sections[idx - 1]);
        setSelectedIndex(0);
      }
    }
    if (input === 'l' || key.rightArrow) {
      const sections: Array<'pending' | 'running' | 'completed'> = ['pending', 'running', 'completed'];
      const idx = sections.indexOf(activeSection);
      if (idx < sections.length - 1) {
        setActiveSection(sections[idx + 1]);
        setSelectedIndex(0);
      }
    }

    // List navigation with j/k
    if (input === 'j' || key.downArrow) {
      setSelectedIndex(Math.min(selectedIndex + 1, currentList.length - 1));
    }
    if (input === 'k' || key.upArrow) {
      setSelectedIndex(Math.max(selectedIndex - 1, 0));
    }

    // Actions
    const selectedJob = currentList[selectedIndex];
    if (!selectedJob) return;

    // Cancel job
    if (input === 'c' && activeSection === 'pending') {
      try {
        await client.fetchJson(`/api/sandbox/jobs/${selectedJob.job_id}/cancel`, {
          method: 'POST',
        });
        setActionMessage(`Cancelled job ${selectedJob.job_id.slice(0, 8)}...`);
        fetchQueue();
      } catch (err) {
        setActionMessage(`Failed to cancel: ${err}`);
      }
      setTimeout(() => setActionMessage(null), 3000);
    }

    // Retry job
    if (input === 'r' && (activeSection === 'completed' || activeSection === 'pending')) {
      try {
        const result = await client.fetchJson(`/api/sandbox/jobs/${selectedJob.job_id}/retry`, {
          method: 'POST',
        });
        setActionMessage(`Retried job, new ID: ${result.new_job_id?.slice(0, 8)}...`);
        fetchQueue();
      } catch (err) {
        setActionMessage(`Failed to retry: ${err}`);
      }
      setTimeout(() => setActionMessage(null), 3000);
    }

    // View logs
    if (key.return) {
      // Extract run_id from job kwargs if available
      // For now, use job_id as run_id placeholder
      onViewLogs(selectedJob.job_id);
    }
  });

  if (loading) {
    return (
      <Box>
        <Text color="yellow">
          <Spinner type="dots" />
        </Text>
        <Text> Loading queue...</Text>
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
      {/* Section Tabs */}
      <Box marginBottom={1}>
        <SectionTab
          label="Pending"
          count={queue?.pending.length || 0}
          active={activeSection === 'pending'}
        />
        <Text color="gray"> | </Text>
        <SectionTab
          label="Running"
          count={queue?.running.length || 0}
          active={activeSection === 'running'}
        />
        <Text color="gray"> | </Text>
        <SectionTab
          label="Completed"
          count={queue?.completed.length || 0}
          active={activeSection === 'completed'}
        />
      </Box>

      {/* Job List */}
      <Box flexDirection="column" borderStyle="single" borderColor="gray" paddingX={1}>
        {currentList.length === 0 ? (
          <Text color="gray">No jobs in this section</Text>
        ) : (
          currentList.map((job, index) => (
            <JobRow
              key={job.job_id}
              job={job}
              selected={index === selectedIndex}
              section={activeSection}
            />
          ))
        )}
      </Box>

      {/* Action message */}
      {actionMessage && (
        <Box marginTop={1}>
          <Text color="yellow">{actionMessage}</Text>
        </Box>
      )}

      {/* Help hint */}
      <Box marginTop={1}>
        <Text color="gray">
          h/l: sections | j/k: navigate | Enter: view |{' '}
          {activeSection === 'pending' ? 'c: cancel | ' : ''}
          {activeSection !== 'running' ? 'r: retry' : ''}
        </Text>
      </Box>
    </Box>
  );
};

// Section Tab Component
interface SectionTabProps {
  label: string;
  count: number;
  active: boolean;
}

const SectionTab: React.FC<SectionTabProps> = ({ label, count, active }) => {
  return (
    <Text color={active ? 'cyan' : 'gray'} bold={active}>
      {label} ({count})
    </Text>
  );
};

// Job Row Component
interface JobRowProps {
  job: JobInfo;
  selected: boolean;
  section: 'pending' | 'running' | 'completed';
}

const JobRow: React.FC<JobRowProps> = ({ job, selected, section }) => {
  const getStatusIcon = () => {
    switch (section) {
      case 'pending':
        return '◯';
      case 'running':
        return '◐';
      case 'completed':
        return '●';
    }
  };

  const getStatusColor = () => {
    switch (section) {
      case 'pending':
        return 'yellow';
      case 'running':
        return 'cyan';
      case 'completed':
        return 'green';
    }
  };

  const formatTime = (ts?: string) => {
    if (!ts) return '';
    try {
      const date = new Date(ts);
      return date.toLocaleTimeString();
    } catch {
      return '';
    }
  };

  const timeStr =
    section === 'pending'
      ? formatTime(job.enqueue_time)
      : section === 'running'
      ? formatTime(job.start_time)
      : formatTime(job.finish_time);

  return (
    <Box>
      <Text color={selected ? 'white' : 'gray'}>{selected ? '>' : ' '} </Text>
      <Text color={getStatusColor()}>{getStatusIcon()} </Text>
      <Text color={selected ? 'white' : 'gray'}>
        {job.job_id.slice(0, 8)}...
      </Text>
      <Text color="gray"> | </Text>
      <Text color={selected ? 'cyan' : 'gray'}>{job.function}</Text>
      {timeStr && (
        <>
          <Text color="gray"> | </Text>
          <Text color="gray">{timeStr}</Text>
        </>
      )}
    </Box>
  );
};
