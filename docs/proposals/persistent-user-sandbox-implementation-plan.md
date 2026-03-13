# Persistent User Sandbox Implementation Plan

## Scope

Implement and harden Manus-style sandbox behavior in Agent Board:

- One user maps to one long-lived sandbox.
- Multiple tasks/sessions reuse that same sandbox.
- Runtime-installed packages remain available in later commands.
- Verification and auto-repair run against the same persistent environment.

## Current Baseline

Already implemented:

- `PersistentSandboxManager` caches/reuses sandbox executors per `user_id`.
- `BoardSession` persists `user_id`, `sandbox_id`, `sandbox_executor`.
- `E2BExecutor` supports sandbox reuse and reconnect (`attach_sandbox`, `ensure_sandbox`).
- Sandbox verify flow supports bootstrap, missing-module detection, auto-install, and retry.
- Local-module false positives (`src`, `pipeline`) are filtered out from pip auto-install.

## Remaining Work Plan

### Phase 1: Lifecycle Hardening

1. Add explicit lifecycle APIs:
   - release/terminate user sandbox
   - inspect sandbox lease metadata
2. Ensure cleanup on session/archive workflows where appropriate.
3. Record lifecycle events in task/session logs for observability.

### Phase 2: Verification Efficiency

1. Track bootstrap success per session sandbox (or per sandbox id).
2. Skip repeated heavy bootstrap on every verify attempt when already successful.
3. Keep manual override to force bootstrap re-run (env flag).

### Phase 3: Persistence Semantics Across Service Restarts

1. Persist user-to-sandbox lease metadata in durable store (not only memory).
2. On process restart, attempt reconnect using stored `sandbox_id`.
3. Fallback strategy:
   - reconnect fails -> create new sandbox
   - update session/store with new `sandbox_id`

### Phase 4: UX and Debuggability

1. Expose sandbox metadata in API responses/events:
   - `sandbox_id`, executor type, mode (`persistent|ephemeral`)
2. Surface verify/install timeline in Agent Board execution logs.
3. Add clear user-facing error messages for:
   - invalid template
   - API key missing
   - package install timeout/failure

## Edge Cases To Cover

1. User has stored `sandbox_id` but sandbox expired on provider side.
2. API process restarts between task steps.
3. Concurrent sessions for same `user_id` dispatch tasks simultaneously.
4. Verify command fails due to missing package, package install partially fails.
5. Missing module name matches local folder/file and must not be pip-installed.
6. Provider SDK path differences (`connect`, `from_id`, constructor attach).
7. Sandbox unavailable mid-run; command must fail clearly and not hang.

## Functional Test Plan

### A. Persistent Package Behavior

1. Start session for `user_a`.
2. Run `pip install <small-package>` inside sandbox.
3. Run a second task/verify command in same user context.
4. Assert import succeeds without reinstall.

### B. Cross-Session Reuse (Same User)

1. Create session S1 with `user_a`; capture `sandbox_id`.
2. Create session S2 with `user_a`.
3. Assert both sessions resolve the same active sandbox id.

### C. User Isolation

1. Create `user_a` and `user_b` sessions.
2. Install package/file marker only in `user_a` sandbox.
3. Assert it is not visible in `user_b`.

### D. Restart Recovery

1. Persist session with `sandbox_id`.
2. Simulate manager reset/process restart.
3. Trigger new run; assert reconnect attempted and commands execute.

### E. Auto-Install Guardrails

1. Produce verify log with `No module named 'src'` and `'pipeline'`.
2. Assert auto-install list excludes these local modules.
3. Assert third-party module (e.g. `sqlalchemy`) still gets installed.

## Rollout Strategy

1. Keep `PAPERBOT_SANDBOX_MODE` feature flag (`persistent` default).
2. Add metrics/log counters:
   - sandbox reuse rate
   - reconnect success rate
   - verify pass rate after auto-install
3. Roll out in staging first with real E2B template and representative paper projects.
4. Promote to production after functional scenarios A-E are stable.
