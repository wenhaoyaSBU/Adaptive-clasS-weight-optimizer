# CMOK and adaptive class-weight toy Monte Carlo

Runnable implementation of [the toy-MC plan](docs/adaptive_class_weights_CMOK_toy_MC_plan.md), using the conditional-mean composition in [Lenart Jerala's CMOK presentation](docs/lenartj_ttHcc_MLWithEventWeight_02_07_2026.pdf), especially slides 10–13 and backup slides 93–103.

The study compares plain CE, class-balanced event-weighted CE, CMOK, and CMOK with one-step adaptive class weights for a linear classifier and the requested `4 → 64 → 64 → 32 → 2` MLP. It also trains a fixed balanced CMOK control and a validation-selected manual scan. Each replica has independent generated events, weight noise, and model initialization.

## Run

An existing environment on this machine has the required packages:

```bash
cd /home/tomori/ATLAS/toy_CMOK_clsWgt
CMOK_PYTHON=/home/tomori/miniconda3/envs/upp/bin/python

# Analytic normalization checked against independent MC.
"$CMOK_PYTHON" -m cmok_toy validate-truth --output results/truth_validation.json

# Small execution check: 3 toys, 4,000 B + 200 S, 5 epochs.
"$CMOK_PYTHON" -m cmok_toy run --config configs/smoke.json --output results/my_smoke

# Pilot: 5 toys, 20,000 B + 1,000 S, 30 epochs.
"$CMOK_PYTHON" -m cmok_toy run --config configs/pilot.json --output results/my_pilot

# Planned main study: 50 toys, 1,000,000 B + 50,000 S, 30 epochs.
"$CMOK_PYTHON" -m cmok_toy run --config configs/full.json --output results/full

# Resume completed replicas without retraining them; interrupted replicas restart.
"$CMOK_PYTHON" -m cmok_toy run --config configs/full.json --output results/full --resume

"$CMOK_PYTHON" -m pytest -q
```

For another environment, use Python ≥3.10 and `python -m pip install -e '.[test]'`. Running `python -m cmok_toy` directly from this directory does not require installation. CPU is the default and is the validated backend. `--device cuda` is available if the local PyTorch installation and deterministic CUDA operations support it.

Use a fresh output directory for a new experiment. Resume requires the same resolved configuration, Python/NumPy/PyTorch/uproot versions, and source hashes. CLI overrides include `--replicas`, `--seed`, `--n-background`, `--n-signal`, `--epochs`, `--weight-epochs`, `--threads`, and `--device`. All configuration fields and defaults are in `cmok_toy/config.py`; unrecognized JSON fields are rejected.

The full configuration is intentionally much more expensive than smoke/pilot checks. A few small replicas establish implementation correctness, not the adaptive-gain hypothesis.

## Statistical conventions

**Class order is `[B, S]` everywhere.** The exact means, positive-definite covariance matrices, nonlinear transformation, and conditional mean weights are in `truth.py`. The means and weight coefficients follow the plan; the covariance matrices and noise levels (σ_B=0.4, σ_S=1.0) instantiate its unspecified examples.

The transformation is invertible with unit Jacobian. In Gaussian coordinates, `log w̄ = aᵀu + ½uᵀAu`. With `P = Σ⁻¹ − A` and `h = Σ⁻¹μ + a`,

```text
log Z = ½ [hᵀP⁻¹h − μᵀΣ⁻¹μ − log det Σ − log det P].
```

The code verifies positive definiteness of `P`. This gives exact generation and physical densities, normalizations, likelihood ratios, and posterior probabilities without estimating a normalizer from the test set. For the default law, `Z_B = 1.2494521420` and `Z_S = 1.5659512475`.

Each class is split 70/15/15 into training, validation/meta, and test sets. Feature means/scales and weight-target scales use training data only. The realized training fractions define `rho`; small rounding differences in validation/test class fractions are corrected by importance weights. Test events are used only after every classifier, checkpoint, and scan choice has been selected.

### Losses and prior correction

The plan alternates between sums of class-conditional losses and an ordinary event-average loss. This implementation consistently uses the latter:

```text
L = mean_i [ alpha[y_i] × CE(logits_i, y_i) ].
```

Thus `alpha=1` means unweighted CE, and `alpha ∝ 1/rho` means full balancing. Minibatches sample the natural generated imbalance. Positive class factors are divided by `sum_k rho_k alpha_k` to hold the expected loss scale fixed; this normalization is differentiated in the meta-step. A common factor has no effect on inference odds.

Let `d = z_S − z_B`, `r = log(rho_S/rho_B)`, `h = log(ŵ_S/ŵ_B)`, and `n = log(Z_B/Z_S)`:

| Method | Training | Reported log likelihood ratio |
|---|---|---|
| Plain CE | Ordinary CE | `d − r` (generation LR baseline) |
| Balanced weighted CE | `c_y W × CE`, `c_y = 1/(2 rho_y)` | `d − log(c_S/c_B) − r + n` |
| CMOK | Ordinary CE + separate weight regression | `d − r + h + n` |
| Adaptive CMOK | `alpha_y × CE` + separate weight regression | `d − delta − r + h + n` |

`delta = log(alpha_S/alpha_B)`, with raw log weights `[-delta/2, delta/2]`. Plain CE and CMOK share exactly the same generation classifier and checkpoint; their difference is the inference weight correction. The shared checkpoint is selected using CMOK-composed validation physical CE. This isolates the composition effect, but means Plain CE is not independently optimized for its own validation criterion. All other classifier checkpoints also minimize validation physical CE. Training runs for the configured epoch budget and then restores the best checkpoint and its matching delta.

All methods start from the same initialization within each architecture/replica and see the same training minibatch ordering. Meta sampling has its own random stream. The manual scan includes `delta=0` and exact full balancing in addition to the configured scan grid. `scan_best` is chosen using validation CE, never test MSE/AUC/significance. Near-duplicate grid points such as 3 and log(20) remain distinct intentional scan points.

The classifier optimizer is plain SGD, matching the virtual step in the meta-gradient exactly, including weight decay. The adaptive update differentiates through the virtual parameter step, class-weight normalization, **and the explicit `−log(alpha)` correction**. The fitted weight head is frozen. EMA smoothing, a slow meta learning rate, clipping, warmup, and an optional quadratic delta penalty stabilize the update. History records per-class gradient norms and training CE on deterministic training diagnostic subsets, validation CE, delta, and hypergradients.

The literal one-step objective can move delta partly to improve immediate calibration rather than the converged representation. Changing alpha during training also means the final model need not be at the population optimum for its final alpha. No gain or agreement with the fixed scan is forced. Use the saved scan curves and repeated-toy comparisons to assess these effects. Influence-function methods, reliability shrinkage, and significance-based meta-objectives are outside this prototype.

### Weight regression

There are two independent `4 → 32 → 32 → 1` regressors with GELU and exponential output. Targets are noisy event weights in **linear** weight space, divided by each training class's mean weight. MSE is class balanced by a constant per-class factor. It retains the arithmetic conditional mean; fitting `log W`, Huber loss, or a skew-noise median would change the target. Adam trains the regressors, and class-balanced noisy validation MSE selects their checkpoint.

The same frozen weight heads are shared across architectures and CMOK variants in a replica. Numerical output bounds are `log(shape) ∈ [−15,15]`; saturation is reported. The training head never sees exact conditional means except when the explicit oracle control is selected.

### Metrics

Test metrics use exact positive `w̄_Y(x)` to integrate the physical law, reducing irrelevant evaluation noise. Training and validation use the noisy observed `W`. The physical joint class prior for CE/calibration is `P(Y=k) ∝ rho_k Z_k`. Reported LR scores are converted to posteriors with that prior; the Plain CE baseline therefore receives the same constant physical-prior convention but retains its uncorrected generation shape.

Outputs include physical ROC AUC; physical-weighted Pearson and weighted-midrank Spearman correlations; centered and raw physical log-LR MSE; label CE, exact-posterior CE/KL/MSE and calibration ECE; and binned Asimov significance. Centered MSE subtracts the physical-weighted mean residual **only inside that diagnostic**. No test-derived offset modifies scores, CE, or binning.

Asimov templates use **TrafoD** in `sigmoid(log LR)`, adapted from the workspace's
`VHbbccTransformer/salt-framework/scr/binning_study_utils.py`. Each method's
edges are fitted on its **validation scores and observed noisy event weights**,
then frozen for test evaluation. Signal/background weights are scaled separately
to the configured analysis yields (default S=100, B=10,000), including the same
scaling in the sum of squared background weights. These analysis yields are
independent of the joint-law prior used for CE.

The scan starts at the high-score end and closes a bin when all conditions hold:

```text
z_signal * s/S + z_background * b/B > 1
sqrt(sum of background weights squared) / b < max_background_relative_error
s + b >= min_yield
```

The defaults match the standard local TrafoD settings:

| Configuration field | Default | Meaning |
|---|---:|---|
| `binning_method` | `"trafoD"` | Default for all supplied study configurations |
| `trafo_initial_bins` | 500 | Fine-grid edge points over the validation score range |
| `trafo_z_signal` | 10 | Signal fraction coefficient |
| `trafo_z_background` | 5 | Background fraction coefficient |
| `trafo_max_background_relative_error` | 0.2 | Maximum relative background MC uncertainty (strict) |
| `trafo_min_yield` | 3 | Minimum combined analysis yield per accepted bin |

The final bin count is adaptive, with identical rules across methods but generally
different edges. As in the reference implementation, the low-score remainder is
retained even if it does not meet the acceptance conditions. Outer edges are
extended to `[0,1]`, retaining test events outside the validation score range.
Constant scores or insufficient MC statistics give one bin. Nonpositive signed
validation class totals also fall back to one bin and record the reason. Signed
background uncertainty uses `sum W²`, not a squared sum or absolute-weight yields.

`toy_<seed>/binning.json` records all learned and truth-score edges, settings,
validation yields, and background MC uncertainty; `metric_details.json` records
the same edges with test bin contents. No test event determines an edge. If a
test bin has signal but zero sampled background, its significance remains
unavailable rather than inserting a background floor. Validation-selected edges
and finite MC can still put a learned score's binned significance above the
empirical truth benchmark.

Legacy fixed-binning comparisons remain available with `binning_method="fixed"`
and `bins=<count>`; `bins` has no effect in TrafoD mode. Calibration plots keep
their original probability bins. Existing reports are historical results and
are not recomputed merely by changing the binning configuration.

Across-toy summaries report means, standard errors, percentile bootstrap 95% intervals, and paired **method minus CMOK** differences. Negative MSE/CE differences favor the candidate; positive AUC/significance differences favor it. Each row gives valid/missing replica counts. One-toy uncertainties are unavailable, not zero. Toy bootstraps capture the repeated experiment's variation; they are not a full systematic-error treatment or a profile-likelihood limit.

## Controls

```bash
# Exact weight head; constant-within-class physical weights; signed observed weights.
"$CMOK_PYTHON" -m cmok_toy run --config configs/pilot.json --oracle-weights --output results/oracle
"$CMOK_PYTHON" -m cmok_toy run --config configs/pilot.json --constant-weights --output results/constant
"$CMOK_PYTHON" -m cmok_toy run --config configs/pilot.json --signed-weights --output results/signed

# Matched-seed sweeps with aggregate CSV and performance plots.
"$CMOK_PYTHON" -m cmok_toy sweep --config configs/pilot.json --axis samples --values 1 5 20 --output results/sample_sweep
"$CMOK_PYTHON" -m cmok_toy sweep --config configs/pilot.json --axis capacity --values 0.5 1 2 --output results/capacity_sweep
"$CMOK_PYTHON" -m cmok_toy sweep --config configs/pilot.json --axis noise --values 0.5 1 2 --output results/noise_sweep
```

Constant-weight controls use the original `Z_k` as class constants. The physical and generation class-conditional shapes then coincide. Signed controls use `W=w̄(1+σξ)` with a positive conditional mean. The direct signed empirical CE and signed validation CE need not be bounded or positive; a finite-budget result does not establish their stability. A sampled nonpositive total class/validation weight is rejected explicitly. Validate the positive-weight study before interpreting signed controls.

Capacity sweeps change all MLP widths together and run only the MLP. Sample sweeps multiply both class counts; noise sweeps multiply both σ values. Each variant retains its own config and reports. `--resume` is also supported for sweeps.

## Generated events in ROOT

Every new toy run writes `toy_<seed>/events.root` immediately after generation,
before training. This is always enabled, including when `save_predictions` is
false. It contains **all generated events** from all three splits, stored as a
standard ROOT `TTree` named `Events`. Float branches retain the generator's
64-bit precision; writing is compressed and chunked to limit extra memory use.
The writer uses `uproot`, so a ROOT installation is not needed to run the study.

| Branch | Meaning |
|---|---|
| `event_id` | Unique generated-event ID within this toy; matches `predictions.npz` IDs |
| `class_id` | Background = 0, signal = 1 |
| `split` | Train = 0, validation/meta = 1, test = 2 |
| `x1`, `x2`, `x3`, `x4` | Raw generated observables, before feature standardization |
| `event_weight` | Observed noisy MC weight `W`, including its sign in signed controls |
| `mean_weight` | Exact conditional mean `w̄_Y(x)` for this event's class |
| `mean_weight_background`, `mean_weight_signal` | Exact conditional means for both class hypotheses |
| `exact_generation_log_lr`, `exact_physical_log_lr` | Analytic normalized log likelihood ratios |

The `metadata` TObjString contains JSON with the toy seed, full configuration,
class/split mappings and counts, and truth parameters/normalizers. Events are
stored in train/validation/test order; use `event_id` rather than row number to
match another artifact. Export does not consume random numbers or change training.

For example, in an interactive ROOT session:

```cpp
auto f = TFile::Open("results/my_smoke/toy_1729/events.root");
auto events = f->Get<TTree>("Events");
events->Draw("x1", "class_id == 1");                         // generated signal
events->Draw("x1", "event_weight * (class_id == 1)");        // noisy weighted signal
events->Draw("x1", "mean_weight * (class_id == 1)");         // mean-weighted signal
events->Draw("x2:x1", "class_id == 0 && split == 2", "COLZ"); // test background
```

Or use Python:

```python
import uproot
with uproot.open('results/my_smoke/toy_1729/events.root') as f:
    events = f['Events'].arrays(library='np')
    signal_x1 = events['x1'][events['class_id'] == 1]
```

The existing source/version check on resume still applies: runs made before this
change do not automatically gain ROOT files; use a fresh output directory for a
new run with ROOT export.

## Artifacts and reuse

At the run root:

- `config.json`, `environment.json`: resolved settings, versions, and source hashes.
- `metrics.csv`: one row per model/method/replica, plus exact benchmarks.
- `summary.csv`, `summary.json`, `report.md`: across-toy and paired comparisons.
- `replica_summary.png`: alpha distributions, paired MSE differences, effective-size dependence.

Each `toy_<seed>/` contains selected classifier and weight-head checkpoints, histories, scan objectives and histories, training effective sample sizes, weight-head closure, ROC/calibration coordinates, significance bin contents, timing, and diagnostic plots. `predictions.npz` contains raw test features, labels, observed weights, event IDs, truth log LRs, learned mean weights, and all delivered scores; it is disabled by default for the full study to limit disk usage. `metrics.json` is written last as the completed-replica marker.

```python
import numpy as np
from cmok_toy.inference import load_predict

x = np.load('results/smoke/toy_1729/predictions.npz')['x']
log_lr = load_predict('results/smoke/toy_1729/linear_adaptive.pt', x)
```

Keep `weight_head.pt` beside CMOK classifier checkpoints. The loader restores preprocessing, prior corrections, and the corresponding oracle/constant control when applicable. Plain CE returns its generation LR, as above. Reports can be regenerated without training:

```bash
"$CMOK_PYTHON" -m cmok_toy summarize results/smoke
```

## Validation

`tests/` covers inverse/Jacobian correctness, exact normalization and physical sampling, all population prior corrections, constant-weight closure, noise means, disjoint/reproducible splits, class balancing, finite-difference meta-gradients for linear/MLP models (also against real SGD updates), oracle metrics, tied weighted ranks, sparse Asimov bins, paired uncertainty, end-to-end training/checkpoint reload, resume protection, and oracle/signed controls.

Local implementation validation uses the existing `upp` environment. The 50-replica million-background-event production configuration is provided separately from execution checks; consult the actual output directories for completed runs.

Recorded checks and pilot observations: [implementation validation](docs/implementation_validation.md).
