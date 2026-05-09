# AION-RLT CLI Agent Instructions

Base standard: `instructions.md` from the `ai-agent-standards` repository is the canonical source for AI agent guidance.

## Project purpose
This repository hosts a lightweight Rust CLI for Aixker-RLT model training and export workflows. Key files are:
- `Cargo.toml` — Rust package manifest and dependency declarations
- `src/main.rs` — CLI entry point and command handling
- `README.md` — project overview and usage notes
- `LICENSE` — license terms

## Core rule
Do not invent build, packaging, or CI steps unless explicit manifest/build files exist in this repo. Use only actual project files such as `Cargo.toml`, Rust sources, and existing documentation as the implementation basis.

## How to use these standards
Option 1: Copy the standard files into the project root and keep this file aligned with `ai-agent-standards/instructions.md`.

Option 2: Reference `ai-agent-standards` as an external repository or git submodule/template, and document local overrides here.

For the shared canonical file, see:
- `https://github.com/ali-heidari/ai-agent-standards/blob/develop/instructions.md`
