# Roadmap Tutor Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the A/E modules into a literature-aligned global roadmap and local student tutor whose effects can be tested by mechanism experiments.

**Architecture:** A becomes a multi-source global roadmap with explicit item, sequence, and self channels instead of an early-collapsed relation score. E becomes a student local route selector that ranks A-supported neighbors using interpretable mastery-gap, recent-mastery-gap, reliability, and state-contrast signals. Prediction remains bounded by existing trust-region and monotonic CD constraints.

**Tech Stack:** PyTorch modules in `src/`, smoke tests in `tests/`, experiment launcher in `tools/`, git for local-server sync.

---

### Task 1: A Multi-Source Roadmap Diagnostics

**Files:**
- Modify: `src/model_graph.py`
- Test: `tests/smoke_interpretable_ae.py`

- [ ] Add explicit source-channel diagnostics for item, sequence, self, and receiver-bias contributions.
- [ ] Preserve row-stochastic support behavior and source-balanced top-k.
- [ ] Add smoke assertions that channel weights are non-negative and source diagnostics are finite.

### Task 2: E Local Route Selector

**Files:**
- Modify: `src/model_personal.py`
- Modify if needed: `src/model_structure_forward.py`
- Test: `tests/smoke_interpretable_ae.py`
- Test: `tests/smoke_ae_reliability_features.py`

- [ ] Replace weak residual dominance with explicit route-need components: state contrast, mastery gap, recent mastery gap, and reliability.
- [ ] Keep E support constrained by A; no student-id shortcut and no dense generated graph.
- [ ] Emit component diagnostics so `E_shuffle_student` can be interpreted.

### Task 3: Mechanism Runner Protection

**Files:**
- Modify: `tools/run_mechanism_experiments.py`
- Test: `tests/smoke_mechanism_params.py`

- [ ] Ensure full phase1 uses active E route selector paths.
- [ ] Ensure no_E, E_shuffle_student, A_uniform, and future controls remain semantically distinct.

### Task 4: Verification and Server Run

**Files:**
- No code ownership beyond previous tasks.

- [ ] Run local smoke tests: `python tests/smoke_interpretable_ae.py`, `python tests/smoke_ae_reliability_features.py`, `python tests/smoke_mechanism_params.py`, `python tests/smoke_runtime_regressions.py`.
- [ ] Commit and push code.
- [ ] Sync server with `git pull`.
- [ ] Run phase1 on `assist_09,junyi` with GPUs `0,1,2`.
- [ ] Inspect results. If full is still close to controls, use logged component diagnostics to make one more targeted architecture correction rather than parameter search.
