# Mean-Field Model Predictive Control for Serverless Workloads

An independent, Docker-reproducible simulation study of capacity control for bursty, production-derived serverless workloads. It combines Gaussian-process demand forecasting, a mean-field queueing model, and finite-server queue simulation.

## Selected findings

- In the high-volume scenario, EWMA MF-MPC reduced p95 simulated waiting time by 56.7% versus reactive control (95% paired bootstrap CI: 52.8%–60.6%). Validation alone determined matching status; the reported resource fractions are test-set means (0.2735 vs 0.2734).
- In the same scenario, GP-UCB MF-MPC reduced p95 simulated waiting time by 94.4% versus reactive control (95% paired bootstrap CI: 93.5%–95.3%), but used a substantially higher mean resource fraction (0.758 vs 0.273). Treat this as descriptive evidence, not an equal-budget comparison.
- In the bursty scenario, GP-UCB did not show a clear p95 waiting-time difference from GP-mean MF-MPC (paired percent reduction: −2.5%; 95% CI: −14.8%–8.9%).

These are simulation results, not measurements of Azure Functions latency or evaluations of Azure's autoscaler. Workload shapes are derived from a public Azure Functions trace; raw data are not included.

## Reports

- [English report (PDF)](paper/report.pdf)
- [日本語レポート (Markdown)](paper/report_ja.md)
- [Claims and evidence](docs/CLAIMS.md)

## Reproduce

Requires Docker Desktop with Compose. From the repository root:

```powershell
docker compose build research
docker compose run --rm research uv run python -m mfcontrol reproduce --profile paper
```

The reproduction command downloads the public trace into ignored `data/raw/`, preprocesses it into ignored `data/processed/`, and writes results and figures. See [reproducibility notes](docs/REPRODUCIBILITY.md) and [research protocol](docs/RESEARCH_PROTOCOL.md) for details.

## Scope and limitations

This project is an independent computational study, not a peer-reviewed publication or deployment evaluation. Reconstructed invocation-demand event times are not measured user-request arrival times. The mean-field queue model is an idealized controller model and is not a claim about Azure's internal behavior. See [limitations](docs/LIMITATIONS.md) and [methodological deviations](docs/DEVIATIONS.md).

## License and citation

Code is released under the [MIT License](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff).
