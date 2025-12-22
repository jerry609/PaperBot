/**
 * SubmitPanel Component - Manual job submission form
 *
 * Features:
 * - Paper URL/ID input
 * - Executor selection (E2B/Docker)
 * - Advanced options
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import { client } from '../../utils/api.js';

interface SubmitPanelProps {
  onJobSubmitted: (runId: string) => void;
}

type FormField = 'url' | 'executor' | 'timeout';

interface SubmitResponse {
  status: string;
  job_id?: string;
  run_id?: string;
  error?: string;
}

export const SubmitPanel: React.FC<SubmitPanelProps> = ({ onJobSubmitted }) => {
  const [paperUrl, setPaperUrl] = useState('');
  const [executor, setExecutor] = useState<'e2b' | 'docker'>('e2b');
  const [timeoutValue, setTimeoutValue] = useState('600');
  const [activeField, setActiveField] = useState<FormField>('url');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<{ job_id: string; run_id: string } | null>(null);

  // Keyboard navigation
  useInput((input, key) => {
    if (submitting) return;

    // Field navigation with Tab
    if (key.tab || (input === 'j' && activeField !== 'url')) {
      const fields: FormField[] = ['url', 'executor', 'timeout'];
      const idx = fields.indexOf(activeField);
      setActiveField(fields[(idx + 1) % fields.length]);
    }

    // Shift+Tab for reverse navigation
    if (key.shift && key.tab) {
      const fields: FormField[] = ['url', 'executor', 'timeout'];
      const idx = fields.indexOf(activeField);
      setActiveField(fields[(idx - 1 + fields.length) % fields.length]);
    }

    // Toggle executor with space when on executor field
    if (activeField === 'executor' && (input === ' ' || key.return)) {
      setExecutor(executor === 'e2b' ? 'docker' : 'e2b');
    }

    // Submit with Ctrl+Enter
    if (key.ctrl && key.return) {
      handleSubmit();
    }
  });

  const handleSubmit = async () => {
    if (!paperUrl.trim()) {
      setError('Paper URL or ID is required');
      return;
    }

    setSubmitting(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await client.fetchJson<SubmitResponse>('/api/sandbox/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'paper2code',
          paper_url: paperUrl.includes('://') ? paperUrl : undefined,
          paper_id: !paperUrl.includes('://') ? paperUrl : undefined,
          executor: executor,
          options: {
            timeout: parseInt(timeoutValue) || 600,
          },
        }),
      });

      if (result.status === 'enqueued' && result.job_id && result.run_id) {
        setSuccess({ job_id: result.job_id, run_id: result.run_id });
        // Navigate to logs after a short delay
        globalThis.setTimeout(() => {
          onJobSubmitted(result.run_id!);
        }, 1500);
      } else {
        setError(result.error || 'Submission failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box flexDirection="column">
      <Text color="cyan" bold>
        Submit Paper2Code Job
      </Text>

      {/* Paper URL/ID Input */}
      <Box marginTop={1} flexDirection="column">
        <Text color={activeField === 'url' ? 'cyan' : 'gray'}>
          Paper URL or arXiv ID:
        </Text>
        <Box
          borderStyle={activeField === 'url' ? 'single' : undefined}
          borderColor={activeField === 'url' ? 'cyan' : 'gray'}
          paddingX={1}
          marginTop={1}
        >
          <Text color="cyan">{'> '}</Text>
          <TextInput
            value={paperUrl}
            onChange={setPaperUrl}
            placeholder="https://arxiv.org/abs/... or 2301.12345"
            focus={activeField === 'url'}
          />
        </Box>
      </Box>

      {/* Executor Selection */}
      <Box marginTop={1} flexDirection="column">
        <Text color={activeField === 'executor' ? 'cyan' : 'gray'}>
          Executor:
        </Text>
        <Box marginTop={1}>
          <Box
            borderStyle={activeField === 'executor' ? 'single' : undefined}
            borderColor={activeField === 'executor' ? 'cyan' : 'gray'}
            paddingX={1}
          >
            <Text color={executor === 'e2b' ? 'green' : 'gray'}>
              {executor === 'e2b' ? '●' : '○'} E2B (Cloud)
            </Text>
            <Text color="gray">  </Text>
            <Text color={executor === 'docker' ? 'green' : 'gray'}>
              {executor === 'docker' ? '●' : '○'} Docker (Local)
            </Text>
          </Box>
          {activeField === 'executor' && (
            <Text color="gray"> (Space to toggle)</Text>
          )}
        </Box>
      </Box>

      {/* Timeout Input */}
      <Box marginTop={1} flexDirection="column">
        <Text color={activeField === 'timeout' ? 'cyan' : 'gray'}>
          Timeout (seconds):
        </Text>
        <Box
          borderStyle={activeField === 'timeout' ? 'single' : undefined}
          borderColor={activeField === 'timeout' ? 'cyan' : 'gray'}
          paddingX={1}
          marginTop={1}
        >
          <Text color="cyan">{'> '}</Text>
          <TextInput
            value={timeoutValue}
            onChange={setTimeoutValue}
            placeholder="600"
            focus={activeField === 'timeout'}
          />
        </Box>
      </Box>

      {/* Submit Button */}
      <Box marginTop={2}>
        <Box
          borderStyle="round"
          borderColor={submitting ? 'gray' : 'green'}
          paddingX={2}
        >
          {submitting ? (
            <>
              <Text color="yellow">
                <Spinner type="dots" />
              </Text>
              <Text> Submitting...</Text>
            </>
          ) : (
            <Text color="green" bold>
              Ctrl+Enter to Submit
            </Text>
          )}
        </Box>
      </Box>

      {/* Success Message */}
      {success && (
        <Box marginTop={1} flexDirection="column">
          <Text color="green">Job submitted successfully!</Text>
          <Text color="gray">Job ID: {success.job_id}</Text>
          <Text color="gray">Run ID: {success.run_id}</Text>
          <Text color="yellow">Redirecting to logs...</Text>
        </Box>
      )}

      {/* Error Message */}
      {error && (
        <Box marginTop={1}>
          <Text color="red">Error: {error}</Text>
        </Box>
      )}

      {/* Help */}
      <Box marginTop={2}>
        <Text color="gray">Tab: next field | Shift+Tab: prev | Ctrl+Enter: submit</Text>
      </Box>
    </Box>
  );
};
