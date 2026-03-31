"""
Structured logging for lethes.

Quick start (call once at application startup, before any lethes objects are used):

    from lethes.observability import configure_logging
    configure_logging()                  # JSON → stderr
    configure_logging(fmt="console")     # human-readable coloured output

To route logs to a custom sink (e.g. a future HTTP visualisation server):

    import logging
    from lethes.observability import configure_logging, make_formatter

    class HttpSinkHandler(logging.Handler):
        def emit(self, record):
            payload = self.format(record)   # already serialised JSON string
            # ... POST payload to your server ...

    h = HttpSinkHandler()
    configure_logging(handlers=[h])

You can also mix handlers:

    import logging, sys
    from lethes.observability import configure_logging, make_formatter

    stderr_h = logging.StreamHandler(sys.stderr)
    http_h = HttpSinkHandler(...)
    configure_logging(handlers=[stderr_h, http_h])

All lethes loggers live under the ``lethes`` stdlib logger hierarchy, so the
root application logger is never affected.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog :class:`~structlog.stdlib.BoundLogger` for *name*."""
    return structlog.get_logger(name)


def make_formatter(
    fmt: str = "json",
    extra_processors: list[Any] | None = None,
) -> structlog.stdlib.ProcessorFormatter:
    """
    Build a stdlib :class:`logging.Formatter` that renders structlog events.

    Attach this to any :class:`logging.Handler` to make it receive structured
    output — useful when you want to add your own handler after
    :func:`configure_logging` has already been called.

    Parameters
    ----------
    fmt:
        ``"json"`` (default) or ``"console"``.
    extra_processors:
        Additional structlog processors inserted before the final renderer.
    """
    pre_chain: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        *(extra_processors or []),
    ]
    renderer: Any = (
        structlog.dev.ConsoleRenderer()
        if fmt == "console"
        else structlog.processors.JSONRenderer()
    )
    return structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=pre_chain,
    )


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    handlers: list[logging.Handler] | None = None,
    extra_processors: list[Any] | None = None,
) -> None:
    """
    Configure structured logging for lethes.

    Parameters
    ----------
    level:
        Minimum log level: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        Use ``"DEBUG"`` to see per-step pipeline details.
    fmt:
        Output format — ``"json"`` (default, machine-readable) or ``"console"``
        (human-readable, coloured, great for development).
    handlers:
        stdlib handlers to attach to the ``lethes`` logger.
        Defaults to a single :class:`logging.StreamHandler` writing to *stderr*.
        Any handler that has no formatter already set receives the structlog
        :class:`~structlog.stdlib.ProcessorFormatter` automatically, so plain
        handlers like :class:`logging.handlers.HTTPHandler` or a custom HTTP
        sink work out of the box — they receive the formatted JSON string in
        ``LogRecord.getMessage()``.
    extra_processors:
        Additional structlog processors inserted just before the final renderer,
        useful for injecting custom fields (e.g. service name, environment).

    Examples
    --------
    Minimal JSON setup::

        configure_logging()

    Development console::

        configure_logging(level="DEBUG", fmt="console")

    Send to an HTTP endpoint::

        import logging
        h = logging.handlers.HTTPHandler("logs.example.com", "/ingest", method="POST")
        configure_logging(handlers=[h])
    """
    shared_pre_chain: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        *(extra_processors or []),
    ]

    structlog.configure(
        processors=shared_pre_chain + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = make_formatter(fmt=fmt, extra_processors=extra_processors)

    if handlers is None:
        h: logging.Handler = logging.StreamHandler(sys.stderr)
        h.setFormatter(formatter)
        handlers = [h]
    else:
        for h in handlers:
            if h.formatter is None:
                h.setFormatter(formatter)

    lethes_log = logging.getLogger("lethes")
    lethes_log.handlers.clear()
    for h in handlers:
        lethes_log.addHandler(h)
    lethes_log.setLevel(getattr(logging, level.upper(), logging.INFO))
    lethes_log.propagate = False
