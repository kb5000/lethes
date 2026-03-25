"""
Flag parser — extracts ``!key=value,+persist,-remove`` prefixes from message text.

Ported and refined from ``example.py``.  This is a **pure function** with no
side-effects; callers decide what to do with the stripped message text.
"""

from __future__ import annotations

import re

# Maps flag key → value (or None for boolean flags)
FlagMap = dict[str, str | None]

# Compiled once at import time
_FLAG_PATTERN = re.compile(
    r"""
    (?P<key>[+-]?[\w_.\-]+)          # Key with optional +/- prefix
    (?:
        =
        (?P<value>
            "(?:\\.|[^"\\])*"  |     # Double-quoted (escape support)
            '[^']*'            |     # Single-quoted
            [^,]+                    # Unquoted (up to comma)
        )
    )?
    (?:,|$)                          # Comma separator or end
    """,
    re.VERBOSE,
)


def extract_flags(input_str: str) -> tuple[FlagMap, str]:
    """
    Parse a flag prefix from the beginning of *input_str*.

    A flag block must start with ``!`` and be followed by either a space
    (separating it from the message body) or the end of the string.

    Examples::

        extract_flags("!nosum,full hello world")
        # → ({"nosum": None, "full": None}, "hello world")

        extract_flags('!weight=2.5,+pin=true message')
        # → ({"weight": "2.5", "+pin": "true"}, "message")

        extract_flags("plain message")
        # → ({}, "plain message")

    Returns
    -------
    tuple[FlagMap, str]
        ``(flags, remaining_content)`` where *remaining_content* has the flag
        prefix stripped.  If no flag block is found, *flags* is empty and
        *remaining_content* is the original string.
    """
    if not input_str.startswith("!"):
        return {}, input_str

    # Find the first whitespace outside of quotes to split flag block / body
    in_double = False
    in_single = False
    split_index = -1

    for i, ch in enumerate(input_str[1:], start=1):
        if ch == '"' and input_str[i - 1] != "\\":
            in_double = not in_double
        elif ch == "'" and input_str[i - 1] != "\\":
            in_single = not in_single
        elif ch.isspace() and not in_double and not in_single:
            split_index = i
            break

    if split_index == -1:
        command_part = input_str[1:]
        rest = ""
    else:
        command_part = input_str[1:split_index]
        rest = input_str[split_index:].lstrip()

    if not command_part:
        return {}, rest

    flags: FlagMap = {}
    # Append trailing comma so every flag ends with a separator
    for m in _FLAG_PATTERN.finditer(command_part + ","):
        key = m.group("key")
        if not key:
            continue
        value = m.group("value")
        if value is not None:
            if value.startswith('"') and value.endswith('"'):
                # Remove quotes and interpret Python escape sequences
                value = value[1:-1].encode("utf-8").decode("unicode_escape")
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
        flags[key] = value

    return flags, rest
