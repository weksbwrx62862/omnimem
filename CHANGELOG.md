# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `pyproject.toml` for modern Python packaging and dependency management
- Added `requirements.txt` and `requirements-dev.txt` for dependency installation
- Added GitHub Actions CI workflow for automated testing and linting
- Restructured tests into `tests/` directory with proper pytest configuration
- Added pre-commit hooks configuration (ruff, mypy, trailing-whitespace)
- Added GitHub Issue templates (bug report, feature request)
- Added GitHub Pull Request template
- Added this CHANGELOG.md

### Changed
- Tests now use `omnimem.*` imports instead of `plugins.memory.omnimem.*`

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of OmniMem
- Five-layer memory architecture (L0 Perception → L1 Working → L2 Structured → L3 Deep → L4 Internalized)
- Hybrid retrieval engine (Vector + BM25 + RRF fusion + Knowledge Graph)
- Complete governance engine (Conflict resolution, Temporal decay, Forgetting curve, Privacy levels, Provenance tracking)
- Saga transaction coordination for derived data consistency
- Multi-instance synchronization with vector clocks
- Built-in memory tool compatibility layer
- Context Manager with semantic deduplication and token budget control
- Security features (anti-recursion, input sanitization, Unicode normalization)
- Comprehensive test suite covering all core modules
