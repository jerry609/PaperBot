# Research Track Context Model

## Purpose

PaperBot now treats `ResearchTrack` as the stable aggregate root for research workspace state.
The goal is to stop rebuilding track state ad hoc in route handlers and frontend pages.

This document describes the current target shape after the `#325` refactor stack:

- `#327` introduced an application-layer track context read model
- `#328` exposed a consolidated track context endpoint
- `#329` migrated `ResearchPageNew` to that endpoint and made context builds send explicit `track_id`
- `#331` wrapped track-scoped memory access behind a dedicated service

## Stable Track Snapshot

Use the consolidated track snapshot when a surface needs to answer:

- what track is active
- which tasks and milestones matter right now
- how much track-scoped memory exists
- what effective feedback already shaped the track
- what saved-paper preview and eval summary should be shown

Current API surface:

- `GET /api/research/tracks/{track_id}/context`

Current backend entry point:

- `paperbot.application.services.research_track_context_service.ResearchTrackContextService`

Snapshot fields:

- `track`
- `tasks`
- `milestones`
- `memory`
- `feedback`
- `saved_papers`
- `eval_summary`

This snapshot is intentionally **stable**. Query-dependent recommendation work still belongs to
`POST /api/research/context`.

## Track Memory Ownership

Track memory mutations and reads should not manually assemble:

- `scope_type="track"`
- `scope_id=str(track_id)`
- active-track fallback logic
- affected-track recompute logic

Current backend entry point:

- `paperbot.application.services.track_memory_service.TrackMemoryService`

Use it for:

- inbox reads
- clear-track operations
- bulk moderate / bulk move flows
- scope resolution for track-bound memory mutations

This keeps route code focused on HTTP concerns, metrics, and response shaping.

## Contributor Guidance

### Backend

When adding a new research-track surface:

1. Start from `ResearchTrackContextService` if the need is a stable snapshot.
2. Start from `TrackMemoryService` if the need is a track-scoped memory read or mutation.
3. Add or widen an application-layer port before reaching into infrastructure from a new service.
4. Prefer reusing existing store projections such as:
   - `list_effective_paper_feedback`
   - `list_saved_papers`
   - `summarize_eval`
5. Do not add new route-local stitching if the same aggregation could live in a service.

### Web

When building track-aware pages:

1. Fetch the consolidated track snapshot from `/api/research/tracks/{track_id}/context`.
2. Treat `track_id` as explicit request state, not hidden server-side activation state.
3. Mutations such as track activation, memory clear, and feedback writes should invalidate or refetch the snapshot.
4. Keep query-time recommendations separate from the stable snapshot; use `POST /api/research/context` for that path.

## Test Guidance

Relevant coverage added in this refactor stack:

- `tests/unit/test_research_track_context_service.py`
- `tests/integration/test_research_track_context_routes.py`
- `tests/unit/test_research_context_route_explicit_track.py`
- `tests/unit/test_track_memory_service.py`
- `tests/integration/test_research_track_memory_routes.py`

Isolation note:

- tests that exercise memory metrics must reset route-level collector/store singletons before creating the client
- route modules in this codebase cache stores aggressively; tests should explicitly clear those globals when switching DB URLs
