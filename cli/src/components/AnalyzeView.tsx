/**
 * AnalyzeView Component - Paper analysis and review
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import { client } from '../utils/api.js';

interface AnalyzeViewProps {
  title?: string;
  doi?: string;
  mode?: 'analyze' | 'review';
}

interface AnalysisProgress {
  phase: string;
  message: string;
}

interface AnalysisResult {
  title: string;
  summary: string;
  keyContributions: string[];
  methodology: string;
  strengths: string[];
  weaknesses: string[];
  noveltyScore?: number;
  recommendation?: string;
}

export const AnalyzeView: React.FC<AnalyzeViewProps> = ({
  title: initialTitle,
  doi: initialDoi,
  mode = 'analyze',
}) => {
  const { exit } = useApp();
  const [titleInput, setTitleInput] = useState(initialTitle || '');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState<AnalysisProgress | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
    }
  });

  useEffect(() => {
    if (initialTitle || initialDoi) {
      startAnalysis(initialTitle || '', initialDoi);
    }
  }, [initialTitle, initialDoi]);

  const startAnalysis = async (title: string, doi?: string) => {
    setIsAnalyzing(true);
    setProgress(null);
    setResult(null);
    setError(null);

    try {
      const generator =
        mode === 'review'
          ? client.reviewPaper({ title, abstract: '' })
          : client.analyzePaper({ title, doi });

      for await (const event of generator) {
        if (event.type === 'progress') {
          setProgress(event.data as AnalysisProgress);
        } else if (event.type === 'result') {
          setResult(event.data as AnalysisResult);
        } else if (event.type === 'error') {
          setError(event.message || 'Unknown error');
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSubmit = (value: string) => {
    if (value.trim()) {
      startAnalysis(value.trim());
    }
  };

  const modeLabel = mode === 'review' ? 'Deep Review' : 'Paper Analysis';

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text color="cyan" bold>
        {modeLabel}
      </Text>

      {/* Input */}
      {!isAnalyzing && !result && (
        <Box marginY={1} flexDirection="column">
          <Text color="gray">Enter paper title or DOI:</Text>
          <Box borderStyle="round" borderColor="cyan" paddingX={1} marginTop={1}>
            <Text color="cyan">{'> '}</Text>
            <TextInput
              value={titleInput}
              onChange={setTitleInput}
              onSubmit={handleSubmit}
              placeholder="e.g., Attention Is All You Need"
            />
          </Box>
        </Box>
      )}

      {/* Progress */}
      {isAnalyzing && (
        <Box marginY={1}>
          <Text color="yellow">
            <Spinner type="dots" />
          </Text>
          <Text color="yellow">
            {' '}
            {progress?.phase || 'Analyzing...'}
          </Text>
          {progress?.message && (
            <Text color="gray"> - {progress.message}</Text>
          )}
        </Box>
      )}

      {/* Result */}
      {result && (
        <Box marginY={1} flexDirection="column">
          <Text color="green" bold>
            ✓ Analysis Complete
          </Text>

          {/* Title */}
          <Box marginTop={1}>
            <Text color="cyan" bold>
              {result.title}
            </Text>
          </Box>

          {/* Summary */}
          <Box marginTop={1} flexDirection="column">
            <Text bold>Summary:</Text>
            <Box marginLeft={2}>
              <Text wrap="wrap">{result.summary}</Text>
            </Box>
          </Box>

          {/* Key Contributions */}
          {result.keyContributions.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Text bold>Key Contributions:</Text>
              {result.keyContributions.map((contribution, index) => (
                <Box key={index} marginLeft={2}>
                  <Text color="green">• </Text>
                  <Text wrap="wrap">{contribution}</Text>
                </Box>
              ))}
            </Box>
          )}

          {/* Strengths */}
          {result.strengths.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Text bold color="green">
                Strengths:
              </Text>
              {result.strengths.map((strength, index) => (
                <Box key={index} marginLeft={2}>
                  <Text color="green">+ </Text>
                  <Text wrap="wrap">{strength}</Text>
                </Box>
              ))}
            </Box>
          )}

          {/* Weaknesses */}
          {result.weaknesses.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Text bold color="yellow">
                Weaknesses:
              </Text>
              {result.weaknesses.map((weakness, index) => (
                <Box key={index} marginLeft={2}>
                  <Text color="yellow">- </Text>
                  <Text wrap="wrap">{weakness}</Text>
                </Box>
              ))}
            </Box>
          )}

          {/* Scores */}
          {result.noveltyScore !== undefined && (
            <Box marginTop={1}>
              <Text bold>Novelty Score: </Text>
              <Text color="cyan">{result.noveltyScore}/10</Text>
            </Box>
          )}

          {result.recommendation && (
            <Box marginTop={1}>
              <Text bold>Recommendation: </Text>
              <Text
                color={
                  result.recommendation === 'Accept'
                    ? 'green'
                    : result.recommendation === 'Reject'
                    ? 'red'
                    : 'yellow'
                }
              >
                {result.recommendation}
              </Text>
            </Box>
          )}
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
