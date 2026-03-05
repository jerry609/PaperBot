# Daily Push Ops Runbook

This runbook covers operation and acceptance checks for Daily Push channels and feed delivery.

## 1) Channel configuration

1. Configure `config/push_channels.yaml` and tag production channels with `daily`.
2. Ensure `apprise` is installed (`pip install apprise`).
3. Optional resilience knobs:
   - `PAPERBOT_PUSH_RETRY_ATTEMPTS` (default `3`)
   - `PAPERBOT_PUSH_RETRY_BACKOFF_S` (default `0.8`)
   - `PAPERBOT_PUSH_IDEMPOTENCY_TTL_S` (default `600`)

## 2) Telegram subscription command webhook

- Endpoint: `POST /api/push/telegram/command`
- JSON body:
  - `chat_id`: telegram chat id
  - `text`: command text (`/subscribe`, `/unsubscribe`, `/list`, `/today`)
- Persistent store:
  - `PAPERBOT_TELEGRAM_SUBS_PATH` (default `data/telegram_subscriptions.json`)

## 3) Daily digest quality audit

Use the audit script on generated reports:

```bash
python scripts/audit_daily_push_reports.py \
  --reports-dir reports/dailypaper \
  --sample-size 20 \
  --output-md reports/dailypaper_audit.md \
  --output-csv reports/dailypaper_manual_review.csv
```

Outputs:
- `reports/dailypaper_audit.md`: structural coverage stats
- `reports/dailypaper_manual_review.csv`: manual label sheet for accuracy review

## 4) Feed validation checklist (Feedly/Inoreader)

1. Verify API endpoints return XML:
   - `/api/feed/daily.rss`
   - `/api/feed/daily.atom`
   - `/api/feed/track/{track}.rss`
   - `/api/feed/keyword/{keyword}.rss`
2. Subscribe the URLs in Feedly/Inoreader.
3. Confirm latest items and image enclosure fallback behavior.

## 5) Failure triage

Inspect per-channel payload from push result:
- `ok`: final status
- `attempts`: retry attempts used
- `error_code`: normalized code (`rate_limited`, `timeout`, `auth_failed`, `downstream_unavailable`)

If channels keep failing:
1. Rotate webhook/token.
2. Lower push frequency or increase backoff.
3. Check downstream status page and retry later.
