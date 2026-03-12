"""Email sending for auth flows.

Currently in log mode — no real email is sent.
To switch to Resend, set RESEND_API_KEY and flip _SEND_MODE = "resend".
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_SEND_MODE = os.getenv("EMAIL_SEND_MODE", "log")  # "log" | "resend"


def send_password_reset_email(to_email: str, reset_url: str) -> None:
    if _SEND_MODE == "resend":
        _send_via_resend(to_email, reset_url)
    else:
        _log_email(to_email, reset_url)


def _log_email(to_email: str, reset_url: str) -> None:
    logger.warning(
        "[DEV] Password reset link for %s → %s",
        to_email,
        reset_url,
    )


def _send_via_resend(to_email: str, reset_url: str) -> None:  # pragma: no cover
    import resend  # type: ignore[import]

    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        logger.error("Cannot send email via Resend: RESEND_API_KEY is not set.")
        return
    resend.api_key = api_key
    from_addr = os.getenv("EMAIL_FROM", "PaperBot <noreply@paperbot.app>")

    resend.Emails.send({
        "from": from_addr,
        "to": [to_email],
        "subject": "Reset your PaperBot password",
        "html": f"""
        <p>Hi,</p>
        <p>Click the link below to reset your password. This link expires in <strong>1 hour</strong>.</p>
        <p><a href="{reset_url}">{reset_url}</a></p>
        <p>If you didn't request a password reset, you can ignore this email.</p>
        """,
    })
