# Claim-to-evidence ledger

## high_volume: EWMA MF-MPC versus reactive control

Claim: Strict matched-budget result: EWMA MF-MPC reduced p95 simulated waiting by 56.7% relative to reactive control (95% paired bootstrap CI [52.8%, 60.6%]; mean resource fractions 0.274 vs 0.273). The paired interval excludes zero.

Metric definition: paired per-seed p95 simulated waiting time; percent reduction is 100 * (comparator - treatment) / comparator.

Configuration: `configs/paper.yaml`; primary empirical-service experiment. Scenario: `high_volume`.

Artifact source: `results/controller_metrics_per_seed.csv`, `results/paired_controller_comparisons.csv`, and `results/selected_hyperparameters.json`.

Seeds: 10. Matching: treatment=matched; comparator=reference; configured tolerance=2%.

Point estimate: percent reduction = 56.739%; 95% CI = [52.775%, 60.551%].

Resource fractions: treatment=0.2735; comparator=0.2734; relative mismatch=0.064%.

Allowed wording: Strict matched-budget result: EWMA MF-MPC reduced p95 simulated waiting by 56.7% relative to reactive control (95% paired bootstrap CI [52.8%, 60.6%]; mean resource fractions 0.274 vs 0.273). The paired interval excludes zero.

Forbidden stronger wording: production superiority, equal-cost superiority when unmatched, or any statement that uses test data for tuning.

Matching status was determined on validation only; reported resource fractions below are test-set means.

## high_volume: GP-UCB MF-MPC versus reactive control

Claim: Descriptive, not strict matched-budget: GP-UCB MF-MPC reduced p95 simulated waiting by 94.4% relative to reactive control (95% paired bootstrap CI [93.5%, 95.3%]; mean resource fractions 0.758 vs 0.273). The paired interval excludes zero.

Metric definition: paired per-seed p95 simulated waiting time; percent reduction is 100 * (comparator - treatment) / comparator.

Configuration: `configs/paper.yaml`; primary empirical-service experiment. Scenario: `high_volume`.

Artifact source: `results/controller_metrics_per_seed.csv`, `results/paired_controller_comparisons.csv`, and `results/selected_hyperparameters.json`.

Seeds: 10. Matching: treatment=unmatched; comparator=reference; configured tolerance=2%.

Point estimate: percent reduction = 94.399%; 95% CI = [93.485%, 95.273%].

Resource fractions: treatment=0.7580; comparator=0.2734; relative mismatch=177.289%.

Allowed wording: Descriptive, not strict matched-budget: GP-UCB MF-MPC reduced p95 simulated waiting by 94.4% relative to reactive control (95% paired bootstrap CI [93.5%, 95.3%]; mean resource fractions 0.758 vs 0.273). The paired interval excludes zero.

Forbidden stronger wording: production superiority, equal-cost superiority when unmatched, or any statement that uses test data for tuning.

## high_volume: GP-UCB MF-MPC versus GP-mean MF-MPC

Claim: Descriptive, not strict matched-budget: GP-UCB MF-MPC reduced p95 simulated waiting by 93.5% relative to GP-mean MF-MPC (95% paired bootstrap CI [93.0%, 93.8%]; mean resource fractions 0.758 vs 0.383). The paired interval excludes zero.

Metric definition: paired per-seed p95 simulated waiting time; percent reduction is 100 * (comparator - treatment) / comparator.

Configuration: `configs/paper.yaml`; primary empirical-service experiment. Scenario: `high_volume`.

Artifact source: `results/controller_metrics_per_seed.csv`, `results/paired_controller_comparisons.csv`, and `results/selected_hyperparameters.json`.

Seeds: 10. Matching: treatment=unmatched; comparator=matched; configured tolerance=2%.

Point estimate: percent reduction = 93.453%; 95% CI = [92.966%, 93.827%].

Resource fractions: treatment=0.7580; comparator=0.3835; relative mismatch=97.673%.

Allowed wording: Descriptive, not strict matched-budget: GP-UCB MF-MPC reduced p95 simulated waiting by 93.5% relative to GP-mean MF-MPC (95% paired bootstrap CI [93.0%, 93.8%]; mean resource fractions 0.758 vs 0.383). The paired interval excludes zero.

Forbidden stronger wording: production superiority, equal-cost superiority when unmatched, or any statement that uses test data for tuning.

## bursty: GP-UCB MF-MPC versus GP-mean MF-MPC

Claim: Descriptive, not strict matched-budget: GP-UCB MF-MPC showed no clear p95 waiting-time difference from GP-mean MF-MPC; the paired percent reduction was -2.5% (95% paired bootstrap CI [-14.8%, 8.9%]). The paired interval includes zero.

Metric definition: paired per-seed p95 simulated waiting time; percent reduction is 100 * (comparator - treatment) / comparator.

Configuration: `configs/paper.yaml`; primary empirical-service experiment. Scenario: `bursty`.

Artifact source: `results/controller_metrics_per_seed.csv`, `results/paired_controller_comparisons.csv`, and `results/selected_hyperparameters.json`.

Seeds: 10. Matching: treatment=unmatched; comparator=unmatched; configured tolerance=2%.

Point estimate: percent reduction = -2.520%; 95% CI = [-14.760%, 8.949%].

Resource fractions: treatment=0.1237; comparator=0.1172; relative mismatch=5.570%.

Allowed wording: Descriptive, not strict matched-budget: GP-UCB MF-MPC showed no clear p95 waiting-time difference from GP-mean MF-MPC; the paired percent reduction was -2.5% (95% paired bootstrap CI [-14.8%, 8.9%]). The paired interval includes zero.

Forbidden stronger wording: production superiority, equal-cost superiority when unmatched, or any statement that uses test data for tuning.
