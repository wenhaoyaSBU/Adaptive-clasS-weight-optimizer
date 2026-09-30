# Implementation validation

Validated on the local CPU using `/home/tomori/miniconda3/envs/upp/bin/python` with two PyTorch threads.

- **16 tests passed.** These include analytic-law checks, finite-difference hypergradients for both architectures, agreement with actual SGD updates, checkpoint inference, replica resumption, and signed/oracle controls.
- **Normalization:** with 200,000 independent generated events per class, analytic and Monte Carlo normalizers differed by 0.59 standard errors for background and −0.28 for signal. See [truth validation](../results/truth_validation.json).
- **Smoke experiment:** three independent toys, each with 4,000 background and 200 signal events, five training epochs, both architectures, all four methods, and fixed/manual-scan controls. See [smoke report](../results/smoke/report.md).
- **Longer pilot:** two toys, each with 20,000 background and 1,000 signal events, 30 epochs, the requested default MLP, all methods, and the full fixed-delta scan. See [pilot report](../results/pilot_validation/report.md).
- **Capacity sweep:** the command completed two small MLP width settings and produced [the aggregate plot](../results/control_validation/capacity_performance.png).
- The generated training/scan diagnostic figure was visually inspected. Numerical test predictions were compared with separately reloaded checkpoints for every method and both architectures.

The pilot's mean adaptive signal/background class-weight ratio was approximately 1.024 for the linear model and 1.000 for the MLP. Adaptive results remained close to ordinary CMOK, while the manual scan favored substantially larger signal weights. This is a useful diagnostic of the literal slow one-step update; the implementation does not force agreement with the scan or modify the specified objective to manufacture a gain.

Two toys are insufficient to interpret performance differences as established effects. The full 50-replica study with one million background events per toy is configured in [full.json](../configs/full.json), but was **not executed** during implementation validation. The smoke and pilot artifacts are execution/diagnostic checks, not the planned production result.

To reproduce these exact execution checks in new output directories:

```bash
cd /home/tomori/ATLAS/toy_CMOK_clsWgt
CMOK_PYTHON=/home/tomori/miniconda3/envs/upp/bin/python
"$CMOK_PYTHON" -m pytest -q
"$CMOK_PYTHON" -m cmok_toy run --config configs/smoke.json --output results/smoke_reproduction
"$CMOK_PYTHON" -m cmok_toy run --config configs/pilot.json --replicas 2 --output results/pilot_reproduction
```

The environment emits Matplotlib/PyParsing deprecation warnings during tests; there were no test failures in the final suite.

## ROOT event export validation

After adding ROOT export, **18 tests passed**. ROOT round-trip checks cover all
generated events and splits, lossless raw features and observed weights, exact
mean weights and log likelihood ratios, signed/constant controls, and output
when NPZ prediction saving is disabled. Resume preserves completed ROOT files.

A fresh smoke toy produced [events.root](../results/root_validation/toy_1729/events.root)
with 4,200 entries and 13 branches in the `Events` TTree. Standalone ROOT C++
successfully read the metadata and drew unweighted signal, weighted signal,
and test-background distributions. The local PyROOT binding also read/drew
successfully but crashed during Python shutdown; the standalone ROOT check
exited successfully. The project writes files with uproot and does not import
PyROOT.

## TrafoD binning validation

After making TrafoD the default significance binning, **28 tests passed**.
The tests compare the implementation with an independent event-level scan of
the local VHbbccTransformer TrafoD recipe for positive and signed weights,
check strict occupancy/MC-error thresholds and the inclusive yield threshold,
and verify that squared noisy background weights determine the MC error.
They also cover the low-score remainder, constant scores, sparse samples,
validation-fitted edges, and inclusion of test scores beyond the validation range.

A complete smoke toy using both architectures and all methods produced
[the TrafoD report](../results/trafod_validation/report.md). It selected 8–11
bins depending on the score. The frozen validation edges and their diagnostics
are in [binning.json](../results/trafod_validation/toy_1729/binning.json);
reported bin counts and test histogram edges were checked against this file.
ROOT event export also completed. Existing historical fixed-bin reports have
not been relabeled or recomputed.
