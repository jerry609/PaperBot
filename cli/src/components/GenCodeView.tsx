/**
 * GenCodeView Component - Paper2Code generation with progress
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import { client } from '../utils/api.js';

interface GenCodeViewProps {
  title?: string;
  abstract?: string;
  outputDir?: string;
}

interface GenCodeProgress {
  phase: string;
  message: string;
  currentFile?: string;
  filesGenerated?: number;
  totalFiles?: number;
}

interface GeneratedFile {
  name: string;
  lines: number;
  purpose: string;
}

interface GenCodeResult {
  success: boolean;
  outputDir: string;
  files: GeneratedFile[];
  blueprint: {
    architectureType: string;
    domain: string;
  };
  verificationPassed: boolean;
}

type InputStep = 'title' | 'abstract' | 'generating';

export const GenCodeView: React.FC<GenCodeViewProps> = ({
  title: initialTitle,
  abstract: initialAbstract,
  outputDir = './output',
}) => {
  const { exit } = useApp();
  const [step, setStep] = useState<InputStep>(
    initialTitle && initialAbstract ? 'generating' : 'title'
  );
  const [title, setTitle] = useState(initialTitle || '');
  const [abstract, setAbstract] = useState(initialAbstract || '');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState<GenCodeProgress | null>(null);
  const [result, setResult] = useState<GenCodeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
    }
  });

  useEffect(() => {
    if (initialTitle && initialAbstract) {
      startGeneration();
    }
  }, []);

  const startGeneration = async () => {
    setIsGenerating(true);
    setProgress(null);
    setResult(null);
    setError(null);

    try {
      for await (const event of client.generateCode({
        title,
        abstract,
        useOrchestrator: true,
        useRag: true,
      })) {
        if (event.type === 'progress') {
          setProgress(event.data as GenCodeProgress);
        } else if (event.type === 'result') {
          setResult(event.data as GenCodeResult);
        } else if (event.type === 'error') {
          setError(event.message || 'Unknown error');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTitleSubmit = (value: string) => {
    if (value.trim()) {
      setTitle(value.trim());
      setStep('abstract');
    }
  };

  const handleAbstractSubmit = (value: string) => {
    if (value.trim()) {
      setAbstract(value.trim());
      setStep('generating');
      startGeneration();
    }
  };

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color="cyan" bold>
        Paper2Code Generation
      </Text>
      <Text color="gray">Generate code implementation from paper</Text>

      {/* Title Input */}
      {step === 'title' && (
        <Box marginY={1} flexDirection="column">
          <Text color="white">Enter paper title:</Text>
          <Box borderStyle="round" borderColor="cyan" paddingX={1} marginTop={1}>
            <Text color="cyan">{'> '}</Text>
            <TextInput
              value={title}
              onChange={setTitle}
              onSubmit={handleTitleSubmit}
              placeholder="e.g., Attention Is All You Need"
            />
          </Box>
        </Box>
      )}

      {/* Abstract Input */}
      {step === 'abstract' && (
        <Box marginY={1} flexDirection="column">
          <Text color="green">✓ Title: {title}</Text>
          <Box marginTop={1}>
            <Text color="white">Enter paper abstract:</Text>
          </Box>
          <Box borderStyle="round" borderColor="cyan" paddingX={1} marginTop={1}>
            <Text color="cyan">{'> '}</Text>
            <TextInput
              value={abstract}
              onChange={setAbstract}
              onSubmit={handleAbstractSubmit}
              placeholder="We propose a new architecture..."
            />
          </Box>
        </Box>
      )}

      {/* Progress */}
      {isGenerating && progress && (
        <Box marginY={1} flexDirection="column">
          <Box>
            <Text color="yellow">
              <Spinner type="dots" />
            </Text>
            <Text color="yellow" bold>
              {' '}
              {progress.phase}
            </Text>
          </Box>
          <Box marginLeft={2}>
            <Text color="gray">{progress.message}</Text>
          </Box>

          {progress.currentFile && (
            <Box marginTop={1} marginLeft={2}>
              <Text color="cyan">Generating: </Text>
              <Text>{progress.currentFile}</Text>
            </Box>
          )}

          {progress.filesGenerated !== undefined &&
            progress.totalFiles !== undefined && (
              <Box marginTop={1}>
                <FileProgressBar
                  current={progress.filesGenerated}
                  total={progress.totalFiles}
                />
              </Box>
            )}
        </Box>
      )}

      {/* Result */}
      {result && (
        <Box marginY={1} flexDirection="column">
          <Box>
            <Text color={result.success ? 'green' : 'red'} bold>
              {result.success ? '✓' : '✗'} Generation{' '}
              {result.success ? 'Complete' : 'Failed'}
            </Text>
          </Box>

          {/* Blueprint Info */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>Blueprint:</Text>
            <Box marginLeft={2}>
              <Text color="gray">Architecture: </Text>
              <Text>{result.blueprint.architectureType}</Text>
            </Box>
            <Box marginLeft={2}>
              <Text color="gray">Domain: </Text>
              <Text>{result.blueprint.domain}</Text>
            </Box>
          </Box>

          {/* Generated Files */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>Generated Files ({result.files.length}):</Text>
            {result.files.map((file, index) => (
              <Box key={index} marginLeft={2}>
                <Text color="cyan">{file.name}</Text>
                <Text color="gray">
                  {' '}
                  ({file.lines} lines) - {file.purpose}
                </Text>
              </Box>
            ))}
          </Box>

          {/* Verification Status */}
          <Box marginTop={1}>
            <Text bold>Verification: </Text>
            <Text color={result.verificationPassed ? 'green' : 'yellow'}>
              {result.verificationPassed ? '✓ Passed' : '⚠ Needs Review'}
            </Text>
          </Box>

          {/* Output Directory */}
          <Box marginTop={1}>
            <Text bold>Output: </Text>
            <Text color="cyan">{result.outputDir}</Text>
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

interface FileProgressBarProps {
  current: number;
  total: number;
  width?: number;
}

const FileProgressBar: React.FC<FileProgressBarProps> = ({
  current,
  total,
  width = 30,
}) => {
  const percentage = total > 0 ? (current / total) * 100 : 0;
  const filled = Math.round((percentage / 100) * width);
  const empty = width - filled;

  return (
    <Box>
      <Text color="gray">[</Text>
      <Text color="green">{'█'.repeat(filled)}</Text>
      <Text color="gray">{'░'.repeat(empty)}</Text>
      <Text color="gray">] </Text>
      <Text color="cyan">
        {current}/{total} files
      </Text>
    </Box>
  );
};
