# Research protocol

For the revised analysis, split-level data-quality eligibility criteria were fixed before rerunning controller evaluation after a degenerate held-out bursty scenario was found. Criteria in `configs/paper.yaml` require activity in all chronological splits and are evaluated before controller testing. Ranking among eligible applications uses training statistics only; validation/test data never choose an eligible application.

Forecasts are causal and fitted on training rows. GP beta calibration uses validation only. Reactive validation resource fraction is the reference budget; each predictive controller receives adaptive log-space alpha search on validation only and is strict-headline eligible only within the configured two-percent tolerance. Test data never select scenarios, beta, alpha, or model settings.

Tail metrics require the configured minimum number of generated jobs. Missing tail evidence is `insufficient_data`/NA, never zero. Controllers share paired seeds; paired claims bootstrap per-seed percentage reductions. The JSQ simulator validates job conservation, delayed warming, draining, nonnegative waiting, and response at least service duration.
