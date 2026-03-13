# V3 Rescue Plan

## 1. Branch

Current working branch:

- `v3`

Base:

- created from `9934dfb46618ca275edd3f5c95cb9f03e407a575`

Purpose:

- keep the stronger old `A + B/C/E` line as the main rescue path,
- avoid mixing it with the weaker `QRC` rewrite,
- rescue `B`, `C`, `E` with controlled attribution.

## 2. Core Rule

Do **not** first modify `B/C/E` together inside one new `full` model and only look at the final AUC.

Reason:

- if the score goes up, we cannot know which module was rescued;
- if the score goes down, we cannot know which module caused the regression;
- `B`, `C`, `E` currently fail in different ways, so they should not share one first-pass diagnosis.

The correct order is:

1. build one rescue variant for `B`
2. build one rescue variant for `C`
3. build one rescue variant for `E`
4. run them in parallel on the same datasets/seeds
5. only after that, build a combined `B+C`, `B+E`, or `B+C+E` version

## 3. What The Logs Say

### B

Observed state:

- `assist_09`: clearly useful, but too strong and tends to dominate
- `junyi`: weak but still slightly useful

Evidence:

- removing `B` causes the biggest drop on `assist_09`
- in `assist_09 full`, `gate_mean` climbs close to `1.0`
- in `assist_09 full`, `delta_over_irt` climbs to about `0.58`

Interpretation:

- `B` is not dead
- `B` is over-aggressive on `assist_09`
- `B` is under-calibrated across datasets

### C

Observed state:

- small gain on `assist_09`
- slightly negative on `junyi`

Evidence:

- `assist_09`: `no_C` is only slightly below `full`
- `junyi`: `no_C` is slightly above `full`

Interpretation:

- `C` is not structurally broken
- `C` is too weak and too noisy
- current prototype path behaves like a fragile add-on to `B`, not a stable standalone contribution

### E

Observed state:

- `junyi` shows structural collapse
- `assist_09` does not justify keeping `E`

Evidence:

- `junyi full`: `alpha_std` stays at `0`
- graph-related grad norms stay near zero for many epochs
- `no_E` is not worse than `full`

Interpretation:

- `E` is not a hyperparameter-only issue
- `E` needs mechanism-level change, otherwise it should be dropped

## 4. Proposed Rescue Changes

### B Rescue

Goal:

- keep `assist_09` gains,
- stop residual takeover,
- avoid hurting `junyi`.

Planned changes:

1. Add `mf_warmup_epochs`
   - ramp residual contribution from small to full strength
   - avoid early gate saturation

2. Add `lambda_delta_ratio` with a soft cap
   - penalize `delta_over_irt` only when it exceeds a target
   - target can be around `0.20 ~ 0.30`
   - this suppresses the pathological `assist_09` takeover without killing weak but useful `junyi` residuals

3. Make fusion more conservative by default
   - lower initial residual scale
   - slightly lower initial gate bias for the first rescue version

Why this is first priority:

- `B` already has the strongest empirical upside
- it looks rescue-able without rewriting the whole paper story

### C Rescue

Goal:

- make prototype calibration useful only when confident,
- reduce noise on `junyi`,
- increase explainability of prototype usage.

Planned changes:

1. Confidence-thresholded prototype injection
   - if `proto_conf` is too low, reduce or skip prototype correction
   - avoid uniform/meaningless prototype mixing entering the main path

2. Replace fixed `proto_lambda * proto_conf` with a learnable conservative gate
   - initialized small
   - still modulated by confidence

3. Strengthen prototype health constraints
   - keep usage balance
   - keep prototype diversity

Why this is second priority:

- `C` is weak, not fully broken
- its main problem is low signal-to-noise ratio

### E Rescue

Goal:

- determine whether `E` is salvageable at all.

Planned changes:

1. Add `personal_warmup_epochs`
   - do not let personalized graph participate from epoch 1
   - wait until global graph and student representation stabilize

2. Replace the current variance-only encouragement with a direct anti-collapse target
   - instead of only maximizing variance indirectly,
   - penalize `alpha_std` being below a small target

3. Add `personal_delta_scale`
   - current personal branch likely stays too close to the global prior
   - a learnable scale can test whether the branch is failing because the perturbation is too weak

Why this is lower priority:

- current evidence says `E` may still be a dead end
- rescue cost is highest and success probability is lowest

## 5. Experiment Strategy

### Stage 1: Single-Module Rescue

Run these four variants first:

- `full_baseline`
- `full_B_rescue`
- `full_C_rescue`
- `full_E_rescue`

Datasets:

- `assist_09`
- `junyi`

Seeds:

- `42,52,62`

Success rule:

- mean AUC gain over baseline `> 0.003` on at least one dataset
- and no large regression on the other dataset

### Stage 2: Attribution Check

For whichever rescue survives stage 1, run:

- rescued `full`
- rescued `no_B`
- rescued `no_C`
- rescued `no_E`

Purpose:

- confirm the gain really comes from the rescued module

### Stage 3: Combination

Only after stage 1 and stage 2:

- if `B_rescue` works and `C_rescue` works: test `B+C`
- if `B_rescue` works and `E_rescue` works: test `B+E`
- if only one works: keep only that one

## 6. Practical Decision Rule

Recommended priority:

1. rescue `B`
2. rescue `C`
3. test whether `E` is worth one last mechanism-level attempt

Likely paper outcome:

- best-case: `A + B`
- acceptable fallback: `A + B + C`
- low-probability path: `A + B + E`

Current expectation:

- `B` is the most likely module to survive
- `C` is a conditional add-on
- `E` should be removed quickly if the anti-collapse rescue still fails

## 7. Next Implementation Order

Implementation should proceed in this order:

1. add `B` rescue knobs and variant
2. add `C` rescue knobs and variant
3. add `E` rescue knobs and variant
4. extend ablation runner to support rescue variants
5. run stage 1 experiments

