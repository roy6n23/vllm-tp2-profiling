"""Flags known to a vLLM CLI (from its `--help=all` text) and flags used in an argv.

The preflight and the contract tests use `missing_flags` to prove that every argv
tpprof generates is accepted by the real vLLM 0.30.0 parser (spec 4.2, 8).
"""
from __future__ import annotations

import re
from collections.abc import Sequence

_OPTION_LINE = re.compile(r"^\s{2,}(-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*)")
_FLAG = re.compile(r"-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*")
_LOG_PREFIXES = ("INFO", "WARNING", "ERROR", "DEBUG", "W0", "I0", "E0")
_NUMBER = re.compile(r"-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
# argparse rewrites -O2 to --optimization-level 2; tpprof never emits it.
_ALIASES = frozenset({"-O2"})


def parse_help_flags(text: str) -> set[str]:
    """Every option string listed in `--help=all` output, e.g. `--tensor-parallel-size` and `-tp`."""
    flags: set[str] = set()
    for line in text.splitlines():
        if line.startswith(_LOG_PREFIXES) or not _OPTION_LINE.match(line):
            continue
        invocation = re.split(r"\s{2,}", line.strip(), maxsplit=1)[0]
        for piece in invocation.split(", "):
            token = piece.split()[0] if piece.split() else ""
            # Description continuation lines can start with `-cc.mode=none` or `--flag.`;
            # only a whole option string counts.
            if token.startswith("-") and _FLAG.fullmatch(token):
                flags.add(token)
    return flags


def flags_in_argv(argv: Sequence[str]) -> list[str]:
    """Option strings in `argv`, canonicalized: no `=value`, `_` -> `-` in long flags, dotted -> base."""
    flags = []
    for token in argv:
        if not token.startswith("-") or _NUMBER.fullmatch(token):
            continue
        flag = token.split("=", 1)[0]
        flag = flag.split(".", 1)[0]
        if flag.startswith("--"):
            flag = "--" + flag[2:].replace("_", "-")
        flags.append(flag)
    return flags


def missing_flags(argv: Sequence[str], help_text: str) -> list[str]:
    """Flags used in `argv` that the CLI behind `help_text` does not know, in argv order."""
    known = parse_help_flags(help_text)
    return [f for f in flags_in_argv(argv) if f not in known and f not in _ALIASES]
