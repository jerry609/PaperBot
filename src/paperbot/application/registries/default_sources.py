from __future__ import annotations

from paperbot.application.registries.source_registry import SourceRegistry, SourceDescriptor, AcquisitionMode


def register_default_sources(registry: SourceRegistry) -> SourceRegistry:
    """
    Register default sources aligned with the acquisition matrix.

    Note: This is metadata only (no factory implementations yet).
    """

    registry.register(
        SourceDescriptor(
            name="semantic_scholar",
            version="v1",
            reliability=0.9,
            rate_limit_rps=1.0,
            auth="api_key",
            acquisition_mode=AcquisitionMode.api_first,
            enabled_by_default=True,
        )
    )

    registry.register(
        SourceDescriptor(
            name="arxiv",
            version="v1",
            reliability=0.9,
            rate_limit_rps=1.0,
            auth="none",
            acquisition_mode=AcquisitionMode.api_first,
            enabled_by_default=True,
        )
    )

    registry.register(
        SourceDescriptor(
            name="usenix",
            version="v1",
            reliability=0.8,
            rate_limit_rps=0.5,
            auth="none",
            acquisition_mode=AcquisitionMode.http_static,
            enabled_by_default=True,
        )
    )

    registry.register(
        SourceDescriptor(
            name="ndss",
            version="v1",
            reliability=0.8,
            rate_limit_rps=0.5,
            auth="none",
            acquisition_mode=AcquisitionMode.http_static,
            enabled_by_default=True,
        )
    )

    # Restricted publishers: metadata API-first; fulltext optional only with credentials.
    registry.register(
        SourceDescriptor(
            name="acm_dl",
            version="v1",
            reliability=0.5,
            rate_limit_rps=0.2,
            auth="cookie",
            acquisition_mode=AcquisitionMode.restricted,
            enabled_by_default=False,
            metadata={"default_behavior": "metadata_only"},
        )
    )

    registry.register(
        SourceDescriptor(
            name="ieee_xplore",
            version="v1",
            reliability=0.5,
            rate_limit_rps=0.2,
            auth="cookie",
            acquisition_mode=AcquisitionMode.restricted,
            enabled_by_default=False,
            metadata={"default_behavior": "metadata_only"},
        )
    )

    # Social sources
    registry.register(
        SourceDescriptor(
            name="reddit",
            version="v1",
            reliability=0.7,
            rate_limit_rps=1.0,
            auth="oauth",
            acquisition_mode=AcquisitionMode.api_first,
            enabled_by_default=True,
        )
    )

    registry.register(
        SourceDescriptor(
            name="github",
            version="v1",
            reliability=0.88,
            rate_limit_rps=1.0,
            auth="api_key",
            acquisition_mode=AcquisitionMode.api_first,
            enabled_by_default=True,
            metadata={"feeds": ["commits", "releases", "issues"]},
        )
    )

    registry.register(
        SourceDescriptor(
            name="huggingface",
            version="v1",
            reliability=0.8,
            rate_limit_rps=1.0,
            auth="none",
            acquisition_mode=AcquisitionMode.api_first,
            enabled_by_default=True,
            metadata={"feed": "daily_papers"},
        )
    )

    # X/Twitter official recent-search API. Default off until a bearer token is configured.
    registry.register(
        SourceDescriptor(
            name="twitter_x",
            version="v1",
            reliability=0.62,
            rate_limit_rps=1.0,
            auth="oauth",
            acquisition_mode=AcquisitionMode.api_first,
            enabled_by_default=False,
            metadata={
                "policy": "official_recent_search",
                "requires": "PAPERBOT_X_BEARER_TOKEN",
            },
        )
    )

    return registry


