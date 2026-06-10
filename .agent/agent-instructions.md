# AION-RLT CLI — Agent Instructions

This is the master agent instruction file for this repository, per the
`ai-agent-standards` standard. It MUST live at `.agent/agent-instructions.md`.
Any other agent instruction files MUST be placed inside the `.agent/` folder.

The canonical base conventions are copied under [`.agent/base/`](base/):
`assistant-conventions.md`, `code-conventions.md`, `cicd-conventions.md`,
`commit-conventions.md`, `docs-conventions.md`, `repository-conventions.md`.
Project additions are allowed; overrides of the base standard are not.

## Purpose

This repository provides a Rust CLI for Aixker-RLT model training and export
operations. `RLT-CLI` is a consumer project built around the AIXKER-RLT
framework.

## Big picture

- The canonical AI agent behavior is defined by `ai-agent-standards`
  (`instructions.md` + the six base convention files).
- This file is the project-specific wrapper for those standard instructions.

## Project-specific conventions

- Key files:
  - `Cargo.toml` — Rust package manifest and dependency declarations.
  - `src/main.rs` — CLI entrypoint and command definitions.
  - `README.md` — project overview, build instructions, and usage examples.
  - `LICENSE` — license terms.
- Do not invent build, packaging, or CI steps unless explicit manifest/build
  files exist in this repo.
- Prefer small, runnable examples and use exact file paths in code snippets.

## Files to inspect first

- `README.md`
- `.agent/agent-instructions.md` (this file)
- `.agent/base/assistant-conventions.md`
- `.agent/base/code-conventions.md`
- `.agent/base/cicd-conventions.md`
- `.agent/base/commit-conventions.md`
- `.agent/base/docs-conventions.md`
- `.agent/base/repository-conventions.md`

## Interactive workflow (from assistant-conventions)

1. State the overall task before starting work.
2. Explain each step clearly and concisely.
3. Before writing code, describe the exact code change planned.
4. Ask whether the developer understands or agrees before applying the change.
5. Apply the code only after the developer confirms, then proceed.

## Commit conventions (from commit-conventions)

- Format: `type(scope): short summary`, blank line, optional body, blank line,
  `Refs: #<issue>`.
- Allowed types: `feat`, `fix`, `chore`, `docs`, `ci`, `style`, `refactor`,
  `test`.

## Documentation requirements

- A `docs/` folder is required at the repository root.
- `docs/index.md` must exist and list links to the other Markdown files in
  `docs/`.
- When code or standards change, update `README.md` and the relevant Markdown
  files together.

## Agent behavior rules

- Do not invent project-specific build or CI steps unless you find explicit
  files such as `Cargo.toml`, `Makefile`, or `.github/workflows`.
- When the project rules are unclear, ask a clarifying question instead of
  guessing.

## Sync note

If the user requests `sync instructions`, re-read the canonical base convention
files from `ai-agent-standards`, then update this file (`.agent/agent-instructions.md`)
and the files under `.agent/base/` to remain aligned with the canonical guidance.
