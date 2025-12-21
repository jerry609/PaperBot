"""
PaperBot API - FastAPI backend for CLI and Web interfaces
Supports Server-Sent Events (SSE) for streaming responses
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import track, analyze, gen_code, review, chat, runs, jobs
from paperbot.infrastructure.event_log.logging_event_log import LoggingEventLog
from paperbot.infrastructure.event_log.composite_event_log import CompositeEventLog
from paperbot.infrastructure.event_log.sqlalchemy_event_log import SqlAlchemyEventLog

app = FastAPI(
    title="PaperBot API",
    description="API for scholar tracking, paper analysis, and code generation",
    version="0.1.0",
)

# CORS for CLI and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}


# Include routers
app.include_router(track.router, prefix="/api", tags=["Scholar Tracking"])
app.include_router(analyze.router, prefix="/api", tags=["Paper Analysis"])
app.include_router(gen_code.router, prefix="/api", tags=["Paper2Code"])
app.include_router(review.router, prefix="/api", tags=["Review"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(runs.router, prefix="/api", tags=["Runs"])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"])

@app.on_event("startup")
async def _startup_eventlog():
    # Phase-0: create a single event log backend and store on app.state.
    # Per-request run_id/trace_id are generated in handlers.
    try:
        app.state.event_log = CompositeEventLog([LoggingEventLog(), SqlAlchemyEventLog()])
    except Exception:
        # If SQLAlchemy isn't available or DB init fails, fall back to logging only.
        app.state.event_log = LoggingEventLog()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
