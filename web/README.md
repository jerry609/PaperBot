# PaperBot Web Dashboard

Modern React-based interface for PaperBot, built with Next.js 15, Shadcn UI, and Vercel AI SDK.

## Features

- **Morning Paper Dashboard**: Research activity feed.
- **DeepCode Studio**: Interactive Cloud IDE for reproduction.
- **Scholar Profile**: Radar charts and impact metrics.
- **Topic Workflows** (`/workflows`): Configure topic queries, run papers.cool search, inspect XYFlow read-only DAG status, generate DailyPaper preview, and optionally persist report files.

## Workflows Page

`/workflows` is a parameterized workflow panel (MVP) for topic research operations.

### Current capabilities

- Configure multiple topic queries in one run (one line per query)
- Select sources (currently `papers_cool`) and branches (`arxiv`, `venue`)
- Run topic search and inspect ranked/aggregated results
- Generate DailyPaper report preview (markdown)
- Enable optional LLM analysis (`summary/trends/insight/relevance`) for DailyPaper
- Enable optional LLM-as-Judge scoring (multidimension + recommendation)
- Optionally save DailyPaper artifacts (markdown/json) through backend API

### Related API proxy routes

- `POST /api/research/paperscool/search`
- `POST /api/research/paperscool/daily`

These routes proxy to the FastAPI backend (`PAPERBOT_API_BASE_URL`).

### Why parameterized panel first?

We intentionally ship a parameterized workflow panel before a fully draggable n8n/coze-style canvas:

- Faster MVP delivery and lower maintenance burden
- Easier API contract stabilization
- Leaves room to evolve into node-based orchestration later (Source -> Search -> Rank -> Digest -> Schedule)

## Tech Stack

- **Framework**: [Next.js 15](https://nextjs.org) (App Router)
- **UI**: [Shadcn/ui](https://ui.shadcn.com) + Tailwind CSS
- **AI**: [Vercel AI SDK](https://sdk.vercel.ai/docs)
- **Charts**: Recharts & React Flow

## Getting Started

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Run development server**:
   ```bash
   npm run dev
   ```

3. **Open**: [http://localhost:3000](http://localhost:3000)

## Directory Structure

- `app/`: Next.js App Router pages
- `components/ui/`: Shadcn UI components
- `lib/`: Utilities and API wrappers
