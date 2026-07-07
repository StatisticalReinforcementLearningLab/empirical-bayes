# Empirical Bayes Contextual Bandit — Project Context

## Project
Comparing Empirical Bayes (EB), unpooled Thompson Sampling, and pooled Thompson Sampling in contextual bandit environments with longitudinal data (n participants × T time points). Core question: when is EB worse than unpooled or pooled TS?

## Key Documents
- **Meeting notes (Google Doc):** https://docs.google.com/document/d/1xIpsKtApYGvkglkKidauZ9PPW3ehjBJLrXfUF9ldKrw/edit?usp=sharing
- **Simulation slides:** https://docs.google.com/presentation/d/1_fpqbWhbSr6PqojAD5nI7F5XB7wxYRbpy_gOvZfLmjQ/edit?usp=sharing
- **GitHub (time-variant):** https://github.com/StatisticalReinforcementLearningLab/empirical-bayes/tree/main/2026/contextual-bandit/time-variant/041726

## Repo Structure
- `time-variant/041726/` — time-variant contextual bandit simulations (C++)
- `time-invariant/042326/` — time-invariant simulations (C++)
- `write-up/` — contains date-versioned subfolders named `write-up-MMDDYY/`. **Always use the latest version**: run `ls write-up/` and pick the subfolder with the most recent date. Do not hardcode a date — new versions are added over time.
  - Main contextual bandit write-up: `<latest>/Nora/contextual bandit/write-up.tex` (Sec 1=time-variant, 2=time-invariant, 3=two-context)
  - F25 (non-contextual, Fall 2025): `<latest>/Nora/F25/`
  - Midterm: `<latest>/midterm/midterm.tex`
  - Ziping's write-up: `<latest>/Ziping/write-up.tex`

## Communication Style
- Reply concisely. Prefer short answers over comprehensive ones unless the question is complex.
- No bullet-point padding. No trailing summaries of what you just did.
- For math/stats questions: lead with the formula, then explain briefly.
- For code tasks: just make the edit, don't narrate it.

## Write-up Notes
- Nora edits the write-up live in Overleaf during sessions. The local `write-up/` folder is a snapshot and will NOT reflect in-session edits. When Nora shares LaTeX she has written, treat it as the current state — do not assume the local file matches.

## Key People
- Nora (user) — undergrad student, writes simulations
- Betina — collaborator, also writes simulations
- Susan — advisor/PI
