"""Unified bounded SparseMM evaluation API, safe to import on CPU."""

from .api import RunConfig, audit, build_command, run, summarize
from .registry import DEFAULT_METHODS, METHODS, Method

__all__ = ["RunConfig", "Method", "METHODS", "DEFAULT_METHODS", "build_command", "run", "audit", "summarize"]
