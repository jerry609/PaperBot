def test_jobs_router_importable():
    # Import should not require Redis to be running.
    from paperbot.api.routes import jobs  # noqa: F401


