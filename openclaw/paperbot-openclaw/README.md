# paperbot-openclaw

Thin OpenClaw plugin package that bridges PaperBot's FastAPI backend into:

- `registerTool`: `paper_search`, `paper_analyze`, `paper_track`, `gen_code`, `review`, `research`
- `registerHook`: message intent routing and prompt context injection
- `registerCli`: `openclaw paper search|analyze|track|gen-code`
- cron descriptors for `paper-monitor`, `weekly-digest`, `conference-deadlines`, `citation-monitor`

## Configuration

```json
{
  "baseUrl": "http://127.0.0.1:8000",
  "authToken": "",
  "defaultUserId": "default",
  "requestTimeoutMs": 30000,
  "contextTrackId": 0,
  "defaultSearchSources": ["papers_cool"],
  "defaultSearchBranches": ["arxiv", "venue"],
  "enableContextInjection": true,
  "cronQueries": ["llm agents", "retrieval augmented generation"],
  "cronScholarId": ""
}
```

## API Mapping

- `paper_search` -> `POST /api/research/paperscool/search`
- `paper_analyze` -> `POST /api/analyze`
- `paper_track` -> `GET /api/track`
- `gen_code` -> `POST /api/gen-code`
- `review` -> `POST /api/review`
- `research` -> `POST /api/research/context`

## Cron Descriptors

The package exports `DEFAULT_PAPERBOT_CRON_JOBS` so OpenClaw runtime wiring can
map them into its own cron system without hard-coding schedules elsewhere.
