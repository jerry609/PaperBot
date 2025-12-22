/**
 * SandboxView Component - Main sandbox management interface
 *
 * Features:
 * - Tab navigation (Queue, History, Submit)
 * - System status display
 * - Keyboard shortcuts
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import { client } from '../utils/api.js';
import { QueuePanel } from './sandbox/QueuePanel.js';
import { HistoryPanel } from './sandbox/HistoryPanel.js';
import { SubmitPanel } from './sandbox/SubmitPanel.js';
import { LogView } from './sandbox/LogView.js';

type TabType = 'queue' | 'history' | 'submit';
type ViewMode = 'tabs' | 'logs';

interface SystemStatusData {
  e2b: { status: string; api_key_set?: boolean };
  docker: { status: string; containers_active?: number };
  queue: { status: string; redis_connected?: boolean };
}

interface SandboxViewProps {
  initialTab?: TabType;
  runId?: string;
}

export const SandboxView: React.FC<SandboxViewProps> = ({
  initialTab = 'queue',
  runId: initialRunId,
}) => {
  const { exit } = useApp();
  const [activeTab, setActiveTab] = useState<TabType>(initialTab);
  const [viewMode, setViewMode] = useState<ViewMode>(initialRunId ? 'logs' : 'tabs');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(initialRunId || null);
  const [systemStatus, setSystemStatus] = useState<SystemStatusData | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  // Fetch system status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await client.fetchJson<SystemStatusData>('/api/sandbox/status');
        setSystemStatus(status);
      } catch {
        // Ignore errors
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Keyboard shortcuts
  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
    }

    // Help toggle
    if (input === '?') {
      setShowHelp(!showHelp);
      return;
    }

    // Only handle tab navigation when in tabs mode
    if (viewMode === 'tabs') {
      if (key.tab || input === '\t') {
        // Cycle through tabs
        const tabs: TabType[] = ['queue', 'history', 'submit'];
        const currentIndex = tabs.indexOf(activeTab);
        const nextIndex = (currentIndex + 1) % tabs.length;
        setActiveTab(tabs[nextIndex]);
      }

      // Number shortcuts for tabs
      if (input === '1') setActiveTab('queue');
      if (input === '2') setActiveTab('history');
      if (input === '3') setActiveTab('submit');
    }

    // Back from logs view
    if (viewMode === 'logs' && (key.escape || input === 'q')) {
      setViewMode('tabs');
      setSelectedRunId(null);
    }
  });

  // Handler to view logs for a run
  const handleViewLogs = (runId: string) => {
    setSelectedRunId(runId);
    setViewMode('logs');
  };

  // Handler to go back to tabs
  const handleBack = () => {
    setViewMode('tabs');
    setSelectedRunId(null);
  };

  // Render help overlay
  if (showHelp) {
    return (
      <Box flexDirection="column" padding={1}>
        <Text color="cyan" bold>
          Keyboard Shortcuts
        </Text>
        <Box flexDirection="column" marginTop={1}>
          <Text>
            <Text color="yellow">Tab</Text> - Cycle through tabs
          </Text>
          <Text>
            <Text color="yellow">1/2/3</Text> - Jump to Queue/History/Submit
          </Text>
          <Text>
            <Text color="yellow">j/k</Text> - Navigate up/down in lists
          </Text>
          <Text>
            <Text color="yellow">Enter</Text> - Select/View details
          </Text>
          <Text>
            <Text color="yellow">c</Text> - Cancel selected job
          </Text>
          <Text>
            <Text color="yellow">r</Text> - Retry selected job
          </Text>
          <Text>
            <Text color="yellow">q/Esc</Text> - Back/Exit
          </Text>
          <Text>
            <Text color="yellow">?</Text> - Toggle this help
          </Text>
          <Text>
            <Text color="yellow">Ctrl+C</Text> - Exit
          </Text>
        </Box>
        <Box marginTop={2}>
          <Text color="gray">Press ? to close help</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" padding={1}>
      {/* Header */}
      <Box justifyContent="space-between">
        <Text color="cyan" bold>
          Sandbox Manager
        </Text>
        <SystemStatusBar status={systemStatus} />
      </Box>

      {/* Tab Bar (only show in tabs mode) */}
      {viewMode === 'tabs' && (
        <Box marginY={1}>
          <TabBar
            tabs={[
              { key: 'queue', label: '1. Queue' },
              { key: 'history', label: '2. History' },
              { key: 'submit', label: '3. Submit' },
            ]}
            activeTab={activeTab}
            onSelect={setActiveTab}
          />
        </Box>
      )}

      {/* Content */}
      {viewMode === 'tabs' && (
        <Box flexDirection="column">
          {activeTab === 'queue' && <QueuePanel onViewLogs={handleViewLogs} />}
          {activeTab === 'history' && <HistoryPanel onViewLogs={handleViewLogs} />}
          {activeTab === 'submit' && <SubmitPanel onJobSubmitted={handleViewLogs} />}
        </Box>
      )}

      {viewMode === 'logs' && selectedRunId && (
        <LogView runId={selectedRunId} onBack={handleBack} />
      )}

      {/* Footer */}
      <Box marginTop={1} borderStyle="single" borderColor="gray" paddingX={1}>
        <Text color="gray">
          {viewMode === 'tabs'
            ? 'Tab: switch | 1-3: jump | ?: help | Ctrl+C: exit'
            : 'q/Esc: back | c: cancel | r: retry | ?: help'}
        </Text>
      </Box>
    </Box>
  );
};

// Tab Bar Component
interface TabDef {
  key: TabType;
  label: string;
}

interface TabBarProps {
  tabs: TabDef[];
  activeTab: TabType;
  onSelect: (tab: TabType) => void;
}

const TabBar: React.FC<TabBarProps> = ({ tabs, activeTab }) => {
  return (
    <Box>
      {tabs.map((tab, index) => (
        <React.Fragment key={tab.key}>
          <Box
            paddingX={1}
            borderStyle={tab.key === activeTab ? 'single' : undefined}
            borderColor={tab.key === activeTab ? 'cyan' : undefined}
          >
            <Text color={tab.key === activeTab ? 'cyan' : 'gray'} bold={tab.key === activeTab}>
              {tab.label}
            </Text>
          </Box>
          {index < tabs.length - 1 && <Text color="gray"> </Text>}
        </React.Fragment>
      ))}
    </Box>
  );
};

// System Status Bar Component
interface SystemStatusBarProps {
  status: SystemStatusData | null;
}

const SystemStatusBar: React.FC<SystemStatusBarProps> = ({ status }) => {
  if (!status) {
    return <Text color="gray">Loading status...</Text>;
  }

  const getStatusColor = (s: string) => {
    if (s === 'healthy' || s === 'available') return 'green';
    if (s === 'not_configured' || s === 'unavailable') return 'yellow';
    return 'red';
  };

  return (
    <Box>
      <Text color="gray">E2B:</Text>
      <Text color={getStatusColor(status.e2b?.status || 'unknown')}>
        {status.e2b?.status === 'available' ? '●' : '○'}
      </Text>
      <Text color="gray"> Docker:</Text>
      <Text color={getStatusColor(status.docker?.status || 'unknown')}>
        {status.docker?.status === 'healthy' ? '●' : '○'}
      </Text>
      <Text color="gray"> Redis:</Text>
      <Text color={status.queue?.redis_connected ? 'green' : 'yellow'}>
        {status.queue?.redis_connected ? '●' : '○'}
      </Text>
    </Box>
  );
};
