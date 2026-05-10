# AION-RLT CLI Agent Instructions

This file is derived from `ai-agent-standards/instructions.md` and adapts the canonical guidance for this repository.

## Purpose
This repository provides a Rust CLI for Aixker-RLT model training and export operations.

## Big picture
- `RLT-CLI` is a consumer project built around the AIXKER-RLT framework.
- The canonical AI agent behavior is defined in `ai-agent-standards/instructions.md`.
- Local agents should treat this file as the project-specific wrapper for those standard instructions.

## Project-specific conventions
- Key files:
  - `Cargo.toml` — Rust package manifest and dependency declarations.
  - `src/main.rs` — CLI entrypoint and command definitions.
  - `README.md` — project overview, build instructions, and usage examples.
  - `LICENSE` — license terms.
- Do not invent build, packaging, or CI steps unless explicit manifest/build files exist in this repo.
- Prefer small, runnable examples and use exact file paths in code snippets.

## How to use these standards
- Option A: Copy `instructions.md` and other standard convention files into the project root.
- Option B: Reference `ai-agent-standards` as a submodule or external template and document any local overrides here.

## Agent behavior rules
- Do not invent project-specific build or CI steps unless you find explicit files such as `Cargo.toml`, `Makefile`, or `.github/workflows` in the repo.
- When the project rules are unclear, ask a clarifying question instead of guessing.
- Inspect these files first when making changes:
  - `README.md`
  - `instructions.md`
  - `assistant-conventions.md`
  - `code-conventions.md`
  - `cicd-conventions.md`
  - `commit-conventions.md`
  - `docs-conventions.md`
  - `repository-conventions.md`

## Documentation requirements
- This project uses the `ai-agent-standards` docs convention.
- A `docs/` folder is required at the repository root.
- `docs/index.md` must exist and list links to other Markdown files in `docs/`.

## Sync note
If the user requests `sync instructions`, refresh this file to remain aligned with the canonical `ai-agent-standards/instructions.md` guidance.
