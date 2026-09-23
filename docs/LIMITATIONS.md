# Limitations

- Arrival time is reconstructed as `end_timestamp - duration`.
- Workload shapes are production-derived, but simulation intensities are scaled; finite-$N$ simulation is not Azure deployment.
- The controller uses a stationary/exponential-service mean-field proxy and a quasi-stationary approximation, not a full transient controlled mean-field ODE.
- Empirical service simulation is intentional model mismatch.
- GP fitting may subsample for tractability, and only three representative scenarios are studied.
- Discrete capacity can prevent exact resource matching. Unmatched GP-UCB comparisons cannot establish equal-cost superiority.
