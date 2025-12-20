#!/usr/bin/env node
/**
 * PaperBot CLI - Interactive terminal interface for scholar tracking and paper analysis
 * Built with Ink (React for CLI) inspired by Gemini CLI and Claude Code
 */

import React from 'react';
import { render } from 'ink';
import meow from 'meow';
import { App } from './components/App.js';
import { showBanner } from './utils/banner.js';

const cli = meow(`
  Usage
    $ paperbot [command] [options]

  Commands
    track           Track scholars and generate reports
    analyze         Analyze a paper by title/DOI
    gen-code        Generate code from paper (Paper2Code)
    review          Deep review a paper
    chat            Interactive chat mode (default)

  Options
    --scholar, -s   Scholar name or Semantic Scholar ID
    --title, -t     Paper title
    --doi, -d       Paper DOI
    --abstract, -a  Paper abstract
    --output, -o    Output directory
    --stream        Enable streaming output
    --no-banner     Disable startup banner

  Examples
    $ paperbot
    $ paperbot track --scholar "Dawn Song"
    $ paperbot analyze --title "Attention Is All You Need"
    $ paperbot gen-code --title "..." --abstract "..."
`, {
  importMeta: import.meta,
  flags: {
    scholar: {
      type: 'string',
      shortFlag: 's',
    },
    title: {
      type: 'string',
      shortFlag: 't',
    },
    doi: {
      type: 'string',
      shortFlag: 'd',
    },
    abstract: {
      type: 'string',
      shortFlag: 'a',
    },
    output: {
      type: 'string',
      shortFlag: 'o',
      default: './output',
    },
    stream: {
      type: 'boolean',
      default: true,
    },
    banner: {
      type: 'boolean',
      default: true,
    },
  },
});

async function main() {
  // Show banner unless disabled
  if (cli.flags.banner) {
    await showBanner();
  }

  const command = cli.input[0] || 'chat';

  // Render the React app
  render(
    <App
      command={command}
      flags={cli.flags}
    />
  );
}

main().catch(console.error);
