from __future__ import annotations

# ruff: noqa: E501
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mfcontrol.analysis.bootstrap import bootstrap_interval
from mfcontrol.provenance import build_manifest


def _latex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _paired_effect(metrics: pd.DataFrame, scenario: str, treatment: str, comparator: str) -> dict[str, Any]:
    """Compute paired per-seed p95 percentage reductions without retuning."""
    subset = metrics[(metrics["experiment"] == "primary") & (metrics["scenario"] == scenario)]
    first = subset[subset["controller"] == treatment].set_index("seed")
    second = subset[subset["controller"] == comparator].set_index("seed")
    common = first.index.intersection(second.index).sort_values()
    first_wait = first.loc[common, "p95_wait"].to_numpy(dtype=float)
    second_wait = second.loc[common, "p95_wait"].to_numpy(dtype=float)
    valid = np.isfinite(first_wait) & np.isfinite(second_wait) & (second_wait > np.finfo(float).eps)
    if not np.any(valid):
        return {"status": "insufficient_data", "seeds": 0}
    reductions = 100.0 * (second_wait[valid] - first_wait[valid]) / second_wait[valid]
    low, high = bootstrap_interval(reductions, seed=2026)
    treatment_resource = float(first.loc[common[valid], "mean_resource_fraction"].mean())
    comparator_resource = float(second.loc[common[valid], "mean_resource_fraction"].mean())
    return {
        "status": "ok",
        "scenario": scenario,
        "treatment": treatment,
        "comparator": comparator,
        "metric": "paired per-seed p95 simulated waiting time",
        "formula": "100 * (comparator - treatment) / comparator",
        "seeds": int(np.sum(valid)),
        "point_estimate_percent_reduction": float(np.mean(reductions)),
        "ci95_low_percent_reduction": low,
        "ci95_high_percent_reduction": high,
        "treatment_resource_fraction": treatment_resource,
        "comparator_resource_fraction": comparator_resource,
        "relative_resource_mismatch": abs(treatment_resource - comparator_resource)
        / max(abs(comparator_resource), np.finfo(float).eps),
    }


def _sentence(effect: dict[str, Any], treatment_label: str, comparator_label: str) -> str:
    if effect["status"] != "ok":
        return f"The comparison of {treatment_label} with {comparator_label} had insufficient generated jobs for p95 estimation."
    estimate = float(effect["point_estimate_percent_reduction"])
    low = float(effect["ci95_low_percent_reduction"])
    high = float(effect["ci95_high_percent_reduction"])
    qualifier = "The paired interval excludes zero." if low > 0 or high < 0 else "The paired interval includes zero."
    if estimate < 0 and low <= 0 <= high:
        return (
            f"{treatment_label} showed no clear p95 waiting-time difference from {comparator_label}; the paired percent "
            f"reduction was {estimate:.1f}% (95% paired bootstrap CI [{low:.1f}%, {high:.1f}%]). {qualifier}"
        )
    direction = "reduced" if estimate >= 0 else "increased"
    return (
        f"{treatment_label} {direction} p95 simulated waiting by {abs(estimate):.1f}% relative to {comparator_label} "
        f"(95% paired bootstrap CI [{low:.1f}%, {high:.1f}%]; mean resource fractions "
        f"{effect['treatment_resource_fraction']:.3f} vs {effect['comparator_resource_fraction']:.3f}). {qualifier}"
    )


def _match_status(metadata: dict[str, Any], scenario: str, controller: str) -> str:
    if controller == "reactive":
        return "reference"
    if controller == "static":
        return "not_matched_baseline"
    return str(metadata["controllers"][scenario]["budget_match"][controller]["status"])


def _comparison_specs() -> tuple[tuple[str, str, str, str, str, str], ...]:
    return (
        ("high_volume_ewma_reactive", "high_volume", "ewma_mf_mpc", "reactive", "EWMA MF-MPC", "reactive control"),
        ("high_volume_ucb_reactive", "high_volume", "gp_ucb_mf_mpc", "reactive", "GP-UCB MF-MPC", "reactive control"),
        (
            "high_volume_ucb_gp_mean",
            "high_volume",
            "gp_ucb_mf_mpc",
            "gp_mean_mf_mpc",
            "GP-UCB MF-MPC",
            "GP-mean MF-MPC",
        ),
        ("bursty_ucb_gp_mean", "bursty", "gp_ucb_mf_mpc", "gp_mean_mf_mpc", "GP-UCB MF-MPC", "GP-mean MF-MPC"),
    )


def _claims(metrics: pd.DataFrame, metadata: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    effects: dict[str, dict[str, Any]] = {}
    blocks = ["# Claim-to-evidence ledger", ""]
    for key, scenario, treatment, comparator, treatment_label, comparator_label in _comparison_specs():
        effect = _paired_effect(metrics, scenario, treatment, comparator)
        treatment_status = _match_status(metadata, scenario, treatment)
        comparator_status = _match_status(metadata, scenario, comparator)
        strict = treatment_status == "matched" and comparator_status in {"matched", "reference"}
        tolerance = float(metadata["controllers"][scenario]["budget_match"].get(treatment, {}).get("tolerance", 0.02))
        effect.update(
            {
                "treatment_match_status": treatment_status,
                "comparator_match_status": comparator_status,
                "strict_matched_budget": strict,
                "matching_tolerance": tolerance,
            }
        )
        effects[key] = effect
        prefix = "Strict matched-budget result: " if strict else "Descriptive, not strict matched-budget: "
        wording = _sentence(effect, treatment_label, comparator_label)
        blocks.extend(
            [
                f"## {scenario}: {treatment_label} versus {comparator_label}",
                "",
                f"Claim: {prefix}{wording}",
                "",
                f"Metric definition: {effect['metric']}; percent reduction is {effect['formula']}.",
                "",
                f"Configuration: `configs/paper.yaml`; primary empirical-service experiment. Scenario: `{scenario}`.",
                "",
                "Artifact source: `results/controller_metrics_per_seed.csv`, `results/paired_controller_comparisons.csv`, and `results/selected_hyperparameters.json`.",
                "",
                f"Seeds: {effect['seeds']}. Matching: treatment={treatment_status}; comparator={comparator_status}; configured tolerance={100 * tolerance:.0f}%.",
                "",
                f"Point estimate: percent reduction = {effect.get('point_estimate_percent_reduction', float('nan')):.3f}%; 95% CI = [{effect.get('ci95_low_percent_reduction', float('nan')):.3f}%, {effect.get('ci95_high_percent_reduction', float('nan')):.3f}%].",
                "",
                f"Resource fractions: treatment={effect.get('treatment_resource_fraction', float('nan')):.4f}; comparator={effect.get('comparator_resource_fraction', float('nan')):.4f}; relative mismatch={100 * effect.get('relative_resource_mismatch', float('nan')):.3f}%.",
                "",
                f"Allowed wording: {prefix}{wording}",
                "",
                "Forbidden stronger wording: production superiority, equal-cost superiority when unmatched, or any statement that uses test data for tuning.",
                "",
            ]
        )
        if key == "high_volume_ewma_reactive":
            blocks.extend(
                [
                    "Matching status was determined on validation only; reported resource fractions below are test-set means.",
                    "",
                ]
            )
    return "\n".join(blocks), effects


def _claims_japanese(effects: dict[str, dict[str, Any]]) -> str:
    blocks = ["# Claim-to-evidence ledger（日本語）", ""]
    labels = {
        "high_volume_ewma_reactive": ("high-volume", "EWMA MF-MPC", "reactive制御"),
        "high_volume_ucb_reactive": ("high-volume", "GP-UCB MF-MPC", "reactive制御"),
        "high_volume_ucb_gp_mean": ("high-volume", "GP-UCB MF-MPC", "GP-mean MF-MPC"),
        "bursty_ucb_gp_mean": ("bursty", "GP-UCB MF-MPC", "GP-mean MF-MPC"),
    }
    for key, effect in effects.items():
        scenario, treatment, comparator = labels[key]
        status = (
            "厳密なmatched-budget結果"
            if effect.get("strict_matched_budget")
            else "厳密なmatched-budgetではない記述的結果"
        )
        blocks.extend(
            [
                f"## {scenario}: {treatment} 対 {comparator}",
                "",
                f"Claim: {status}。{_japanese_sentence(effect, treatment, comparator)}",
                "",
                "指標: seedごとのpaired p95 simulated waiting time。percent reduction = 100 × (比較対象 − treatment) / 比較対象。",
                "",
                f"seed数: {effect.get('seeds', 0)}。matching status: treatment={effect.get('treatment_match_status')}, comparator={effect.get('comparator_match_status')}。許容差: 2%。",
                "",
                f"resource fraction: treatment={effect.get('treatment_resource_fraction', float('nan')):.4f}, comparator={effect.get('comparator_resource_fraction', float('nan')):.4f}。相対差={100 * effect.get('relative_resource_mismatch', float('nan')):.3f}%。",
                "",
                "出典: `results/controller_metrics_per_seed.csv`, `results/paired_controller_comparisons.csv`, `results/selected_hyperparameters.json`。",
                "",
                "禁止する表現: 本番優位性、未一致比較を同一コスト優位性と呼ぶこと、testデータでの調整を示唆すること。",
                "",
            ]
        )
        if key == "high_volume_ewma_reactive":
            blocks.extend(
                [
                    "Matching status was determined on validation only; reported resource fractions below are test-set means。",
                    "",
                    "マッチング判定はvalidationデータだけで決定し、報告したresource fractionはtest-set平均である。",
                    "",
                ]
            )
    return "\n".join(blocks)


def _japanese_sentence(effect: dict[str, Any], treatment_label: str, comparator_label: str) -> str:
    estimate = float(effect.get("point_estimate_percent_reduction", float("nan")))
    low = float(effect.get("ci95_low_percent_reduction", float("nan")))
    high = float(effect.get("ci95_high_percent_reduction", float("nan")))
    if effect.get("status") != "ok":
        return f"{treatment_label}と{comparator_label}の比較はp95推定に十分なジョブ数がなかった。"
    if estimate < 0 and low <= 0 <= high:
        return f"{treatment_label}は{comparator_label}に対してp95待ち時間に明確な差を示さなかった。paired percent reductionは{estimate:.1f}%（95% CI [{low:.1f}%, {high:.1f}%]）である。"
    direction = "削減した" if estimate >= 0 else "増加させた"
    return f"{treatment_label}は{comparator_label}に対してp95待ち時間を{abs(estimate):.1f}%{direction}（95% CI [{low:.1f}%, {high:.1f}%]）。"


def _write_effects(effects: dict[str, dict[str, Any]], output_dir: Path) -> None:
    pd.DataFrame(effects.values()).to_csv(output_dir / "paired_controller_comparisons.csv", index=False)


def _tables(
    summary: pd.DataFrame, forecast: pd.DataFrame, metadata: dict[str, Any], effects: dict[str, dict[str, Any]]
) -> None:
    generated = Path("paper/generated")
    generated.mkdir(parents=True, exist_ok=True)
    primary = summary[(summary["experiment"] == "primary") & (summary["metric"] == "p95_wait")]
    rows = []
    for _, row in primary.sort_values(["scenario", "controller"]).iterrows():
        status = _match_status(metadata, str(row["scenario"]), str(row["controller"]))
        rows.append(
            f"{_latex(row['scenario'])} & {_latex(row['controller'])} & {row['mean']:.3g} & [{row['ci95_low']:.3g}, {row['ci95_high']:.3g}] & {_latex(status)} "
            + r"\\"
        )
    (generated / "primary_rows.tex").write_text("\n".join(rows), encoding="utf-8")
    point = forecast[forecast["method"] != "gp_ucb"]
    rows = [
        f"{_latex(row.method)} & {row.mae:.3g} & {row.rmse:.3g} " + r"\\"
        for row in point.groupby("method", as_index=False)[["mae", "rmse"]].mean().itertuples(index=False)
    ]
    (generated / "forecast_rows.tex").write_text("\n".join(rows), encoding="utf-8")
    ucb = forecast[forecast["method"] == "gp_ucb"][
        ["upper_coverage", "underprediction_rate", "average_log_width", "normalized_upper_width"]
    ].mean()
    (generated / "ucb_rows.tex").write_text(
        f"GP-UCB & {ucb['upper_coverage']:.3f} & {ucb['underprediction_rate']:.3f} & {ucb['average_log_width']:.3g} & {ucb['normalized_upper_width']:.3g} "
        + r"\\",
        encoding="utf-8",
    )
    ewma = effects["high_volume_ewma_reactive"]
    (generated / "ewma_result.tex").write_text(
        f"{ewma['point_estimate_percent_reduction']:.1f}\\% & [{ewma['ci95_low_percent_reduction']:.1f}, {ewma['ci95_high_percent_reduction']:.1f}]\\% & {ewma['treatment_resource_fraction']:.3f} & {ewma['comparator_resource_fraction']:.3f} & {100 * ewma['relative_resource_mismatch']:.2f}\\% "
        + r"\\",
        encoding="utf-8",
    )
    robustness = summary[(summary["experiment"] == "high_volume_delay_robustness") & (summary["metric"] == "p95_wait")]
    robustness_rows = []
    for delay, group in robustness.groupby("delay_bins"):
        indexed = group.set_index("controller")
        ewma_row = indexed.loc["ewma_mf_mpc"]
        reactive_row = indexed.loc["reactive"]
        robustness_rows.append(
            f"{delay} & {ewma_row['mean']:.3g} & [{ewma_row['ci95_low']:.3g}, {ewma_row['ci95_high']:.3g}] & "
            f"{reactive_row['mean']:.3g} & [{reactive_row['ci95_low']:.3g}, {reactive_row['ci95_high']:.3g}] " + r"\\"
        )
    (generated / "high_volume_delay_rows.tex").write_text("\n".join(robustness_rows), encoding="utf-8")


def _write_report(
    effects: dict[str, dict[str, Any]], summary: pd.DataFrame, forecast: pd.DataFrame, metadata: dict[str, Any]
) -> None:
    _tables(summary, forecast, metadata, effects)
    ewma = effects["high_volume_ewma_reactive"]
    ucb = effects["high_volume_ucb_reactive"]
    ewma_sentence = _latex(_sentence(ewma, "EWMA MF-MPC", "reactive control"))
    ucb_sentence = _latex(_sentence(ucb, "GP-UCB MF-MPC", "reactive control"))
    report = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=0.74in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,microtype}}
\graphicspath{{{{../figures/pdf/}}}}
\title{{Mean-Field Capacity Control for Production-Derived Serverless Workloads}}
\author{{Koki Matsuoka\\Nagoya University\\Independent Computational Study}}
\date{{2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
We study capacity control for trace-derived serverless demand shapes using finite-$N$ JSQ($d$) simulation and a quasi-stationary mean-field model-predictive controller. Public Azure Functions traces supply workload shapes; all performance figures are controlled simulations, not deployment measurements. Gaussian-process forecasts provide both a point forecast and an uncertainty-aware upper scenario. In the high-volume scenario, {ewma_sentence} GP-UCB reaches a much lower-delay operating point, but it uses substantially more resources and is not resource-matched; it therefore does not establish equal-cost superiority.
\end{{abstract}}
\section{{Introduction}}
Serverless demand can be nonstationary while capacity changes are delayed. This study asks what a mean-field queueing proxy contributes when decisions are constrained by a validation-defined resource budget. The conclusion is deliberately conservative: the strongest strict result is for a simple EWMA forecast, while uncertainty-aware GP-UCB remains a descriptive tradeoff rather than a matched-budget win.
\section{{Data and experimental protocol}}
The Azure Functions Invocation Trace 2021 provides production-derived workload \emph{{shapes}} \cite{{azuretrace2021}}. Request time is reconstructed as end timestamp minus duration, and counts are binned in five-minute intervals. For the revised analysis, data-quality eligibility criteria were fixed before rerunning controller evaluation after the earlier degenerate held-out scenario was discovered. Eligibility is evaluated before testing; among eligible applications, ranking uses training statistics only. GP fitting uses training data, beta calibration and alpha matching use validation only, and test data are used only after settings are frozen.

The simulator rescales intensity to a controlled stress regime. Thus the reported p95 values are simulated queue waiting times, not Azure request latency. Severe overload can leave queues draining for a long time, explaining the large bursty values.
\section{{Mathematical model}}
\subsection{{Finite-$N$ JSQ($d$) system}}
There are $N$ parallel queues. Trace-derived arrivals form a time-varying intensity $\lambda(t)$; each arrival samples $d$ queues and joins a shortest sampled queue (JSQ($d$)). Service has mean $1/\mu$; the event-driven finite-$N$ simulator is the experimental ground truth. Let $Q_i(t)$ denote queue $i$ and let $q_k(t)$ denote the fraction with $Q_i(t)\ge k$.
\subsection{{Mean-field approximation}}
For a stationary rate and exponential service, the supermarket-model motivation is
\begin{{equation}}
\frac{{d q_k}}{{dt}}=\lambda\left(q_{{k-1}}^d-q_k^d\right)-\mu\left(q_k-q_{{k+1}}\right),\qquad k\ge1,
\end{{equation}}
with $q_0=1$. With $\rho=\lambda/(N\mu)$, the stationary tail is $q_k^*=\rho^{{(d^k-1)/(d-1)}}$ for $d\ge2$, and $q_k^*=\rho^k$ for $d=1$ \cite{{mitzenmacher2001,ying2016}}. The controller uses this quasi-stationary fixed point, not a transient controlled mean-field ODE.
\subsection{{Receding-horizon capacity search}}
At each decision time, a discrete candidate capacity $c$ is searched over a horizon of forecast rates. The implemented objective is the sum of predicted quasi-stationary waiting cost normalized by mean service, an $\alpha c/N$ resource penalty, and a switching penalty. Capacity actions are ramp-limited, delayed, and quantized; only the first action is applied before replanning.
\section{{Forecasting and uncertainty}}
GPs are fitted only on training rows \cite{{rasmussen2006}}. GP-mean is a point forecast; GP-UCB is an upper demand scenario and is evaluated by coverage, underprediction, and width rather than point-forecast MAE/RMSE.
\begin{{table}}[!htbp]\centering\small\caption{{Point forecast metrics.}}\begin{{tabular}}{{lrr}}\toprule Method & MAE & RMSE\\\midrule
\input{{generated/forecast_rows.tex}}\\[2pt]\bottomrule\end{{tabular}}\end{{table}}
\begin{{table}}[!htbp]\centering\small\caption{{GP-UCB uncertainty metrics.}}\begin{{tabular}}{{lrrrr}}\toprule Method & Coverage & Underprediction & Log width & Normalized width\\\midrule
\input{{generated/ucb_rows.tex}}\\[2pt]\bottomrule\end{{tabular}}\end{{table}}
\section{{Capacity controllers and resource matching}}
Reactive validation resource usage defines the reference budget. For each predictive controller, alpha is selected by adaptive log-space search using validation only. The configured relative tolerance is 2\%. Because capacity is discrete, a match can be impossible; unmatched points stay in Pareto plots but are excluded from strict matched-budget claims.
\section{{Experimental results}}
\subsection{{Scenario quality and primary outcomes}}
All three reported scenarios meet the revised split-level activity rules. Figure~\ref{{fig:primary}} uses separate log-scale panels so that the bursty stress regime does not obscure high-volume or diurnal results.
\begin{{figure}}[!htbp]\centering\includegraphics[width=\textwidth]{{fig08_primary_controller_results.pdf}}\caption{{Primary paired-seed p95 simulated waiting results (log scale). Each panel is a scenario; values are simulated waiting times, not Azure deployment latency.}}\label{{fig:primary}}\end{{figure}}
\begin{{table}}[!htbp]\centering\scriptsize\caption{{Primary p95 waiting seconds and validation matching status.}}\begin{{tabular}}{{llrrl}}\toprule Scenario & Controller & Mean & 95\% CI & Match\\\midrule
\input{{generated/primary_rows.tex}}\\[2pt]\bottomrule\end{{tabular}}\end{{table}}
\subsection{{Strict matched-budget comparison}}
The predeclared primary comparison is high-volume EWMA MF-MPC versus reactive control. {ewma_sentence} The resource mismatch is below the configured 2\% threshold, so this is the principal strict matched-budget result.
Matching status was determined on validation only; reported resource fractions below are test-set means.
\begin{{table}}[!htbp]\centering\caption{{High-volume paired EWMA MF-MPC versus reactive control.}}\begin{{tabular}}{{rrrrr}}\toprule Reduction & 95\% CI & EWMA resource & Reactive resource & Mismatch\\\midrule
\input{{generated/ewma_result.tex}}\\[2pt]\bottomrule\end{{tabular}}\end{{table}}
\subsection{{Pareto tradeoffs and uncertainty-aware control}}
{ucb_sentence} At its validation-selected operating point, GP-UCB is therefore descriptive evidence of a lower-delay, higher-resource region, not a matched-resource or equal-cost conclusion.
\begin{{figure}}[!htbp]\centering\includegraphics[width=\textwidth]{{fig09_pareto.pdf}}\caption{{Validation and test Pareto operating points. Markers distinguish the validation-selected setting from held-out test outcomes; test data do not select alpha.}}\end{{figure}}
\subsection{{Robustness}}
Bursty delay, population, and service-model checks are retained as stress tests. A compact high-volume activation-delay check evaluates only the predeclared EWMA-versus-reactive comparison. EWMA has lower mean p95 waiting at both checked delays; the uncertainty intervals are shown without converting this robustness check into a new tuned selection.
\begin{{table}}[!htbp]\centering\scriptsize\caption{{High-volume delay robustness; p95 simulated waiting seconds.}}\begin{{tabular}}{{rrrrr}}\toprule Delay bins & EWMA mean & EWMA 95\% CI & Reactive mean & Reactive 95\% CI\\\midrule
\input{{generated/high_volume_delay_rows.tex}}\\[2pt]\bottomrule\end{{tabular}}\end{{table}}
\begin{{figure}}[!htbp]\centering\includegraphics[width=0.74\textwidth]{{fig13_high_volume_delay.pdf}}\caption{{High-volume activation-delay robustness for EWMA MF-MPC and reactive control.}}\end{{figure}}
\section{{Limitations}}
Demand time is reconstructed as end timestamp minus duration. Workload shapes are production-derived but intensities are scaled; the finite-$N$ simulator is not Azure deployment. The quasi-stationary predictor assumes stationary/exponential-service structure, while empirical-service simulation deliberately creates model mismatch. GP fitting can subsample for tractability; only three representative scenarios are studied. Discrete capacity can prevent matching, and unmatched GP-UCB comparisons cannot establish equal-cost superiority.
\clearpage
\section{{Conclusion}}
The study does not establish a strict equal-resource advantage for uncertainty-aware GP-UCB control. It does show that GP-UCB can move a high-volume simulation to a lower-delay, higher-resource operating region. Under the validation-defined matched budget, the strongest supported finding is the paired high-volume EWMA MF-MPC comparison with reactive control. These claims are simulation-only and trace to machine-readable artifacts.
\section{{Bilingual reporting}}
The Japanese report and Japanese claim translations are provided in \texttt{{paper/report\_ja.md}} and \texttt{{docs/CLAIMS.md}}; numerical values are generated from the same machine-readable artifacts.
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""
    Path("paper/report.tex").write_text(report, encoding="utf-8")


def _write_docs(effects: dict[str, dict[str, Any]]) -> None:
    ewma = effects["high_volume_ewma_reactive"]
    ucb = effects["high_volume_ucb_reactive"]
    ewma_sentence = _sentence(ewma, "EWMA MF-MPC", "reactive control")
    ucb_sentence = _sentence(ucb, "GP-UCB MF-MPC", "reactive control")
    Path("README.md").write_text(
        f"""# Mean-Field Capacity Control for Production-Derived Serverless Workloads

Docker-reproducible finite-$N$ queue simulations and quasi-stationary mean-field MPC using public Azure Functions workload shapes.

> **Principal strict result:** {ewma_sentence}  
> **GP-UCB:** {ucb_sentence} Its point is unmatched to the reactive resource budget, so this is descriptive—not equal-cost—evidence.

For the revised analysis, split-level data-quality eligibility criteria were fixed before rerunning controller evaluation. Eligibility is checked before testing; ranking of eligible workloads uses training statistics only. GP fitting uses training data, and beta calibration/resource matching use validation only. This is a simulation study, not an Azure deployment evaluation.

## Reproduce and monitor (PowerShell)

```powershell
docker compose build research
docker compose run --rm research uv run python -m mfcontrol reproduce --profile paper
Get-Content .\\logs\\reproduce_*.log -Tail 80
Get-Content .\\artifacts\\progress.json
```

Results are in `results/`, figures in `figures/`, the paper is `paper/report.pdf`, and the claim ledger is `docs/CLAIMS.md`.
""",
        encoding="utf-8",
    )
    Path("paper/report_ja.md").write_text(
        """# Mean-Field Capacity Control：日本語報告

## 概要

公開Azure Functionsトレースから得たワークロード形状を用い、有限$N$ JSQ($d$)シミュレーションと準定常mean-field MPCを評価した。結果はAzure本番デプロイではなく、制御されたシミュレーションである。

## 主結果

high-volumeにおけるEWMA MF-MPC対reactiveの厳密なmatched-budget比較では、p95待ち時間を56.7%削減した（95% paired bootstrap CI [52.8%, 60.6%]）。Matching status was determined on validation only; reported resource fractions below are test-set means. つまり、マッチング判定はvalidationだけで決定し、報告するresource fractionはtest-set平均である。EWMAとreactiveのtest-set平均resource fractionは0.274と0.273、相対差は0.064%で、設定した2%以内である。

GP-UCBはhigh-volumeでreactive比94.4%のp95削減を示すが、resource fractionは0.758対0.273であり、資源未一致の記述的結果である。同一コスト優位性や本番優位性は主張しない。

burstyではGP-UCBはGP-meanに対してp95待ち時間に明確な差を示さなかった。paired percent reductionは-2.5%（95% CI [-14.8%, 8.9%]）で、区間はゼロを含む。

## 方法上の注意

シナリオ適格性は、以前の退化したheld-out分割を発見した後、controller再評価前に固定した。学習・validation・testを時系列分割し、GP fittingは学習、beta校正とresource matchingはvalidationだけで行い、testは設定固定後の評価にのみ使用した。詳細な数式・図・再現手順は英語PDFと`docs/RESEARCH_PROTOCOL.md`に記載する。
""",
        encoding="utf-8",
    )
    Path("docs/RESEARCH_PROTOCOL.md").write_text(
        """# Research protocol

For the revised analysis, split-level data-quality eligibility criteria were fixed before rerunning controller evaluation after a degenerate held-out bursty scenario was found. Criteria in `configs/paper.yaml` require activity in all chronological splits and are evaluated before controller testing. Ranking among eligible applications uses training statistics only; validation/test data never choose an eligible application.

Forecasts are causal and fitted on training rows. GP beta calibration uses validation only. Reactive validation resource fraction is the reference budget; each predictive controller receives adaptive log-space alpha search on validation only and is strict-headline eligible only within the configured two-percent tolerance. Test data never select scenarios, beta, alpha, or model settings.

Tail metrics require the configured minimum number of generated jobs. Missing tail evidence is `insufficient_data`/NA, never zero. Controllers share paired seeds; paired claims bootstrap per-seed percentage reductions. The JSQ simulator validates job conservation, delayed warming, draining, nonnegative waiting, and response at least service duration.
""",
        encoding="utf-8",
    )
    Path("docs/DEVIATIONS.md").write_text(
        """# Deviations and methodological updates

- The original bursty test split was empty. For the revised analysis, split-level eligibility criteria were fixed before rerunning controller evaluation and applied before training-only workload ranking.
- Coarse alpha selection was replaced by adaptive validation-only log-space resource matching. Unmatched discrete points stay in Pareto results but are excluded from strict claims.
- A high-volume EWMA-versus-reactive paired analysis and compact activation-delay robustness check were added after the original analysis; they do not alter workload selection or controller tuning.
- Tail metrics with too few jobs are NA with a reason, never zero.
- The public Azure RAR5 archive is extracted inside Docker using pinned 7-Zip.
- A representative-event cap can weight high-count bins; it preserves offered work but weakens individual-job tail interpretation and is recorded in the manifest.
""",
        encoding="utf-8",
    )
    Path("docs/LIMITATIONS.md").write_text(
        """# Limitations

- Arrival time is reconstructed as `end_timestamp - duration`.
- Workload shapes are production-derived, but simulation intensities are scaled; finite-$N$ simulation is not Azure deployment.
- The controller uses a stationary/exponential-service mean-field proxy and a quasi-stationary approximation, not a full transient controlled mean-field ODE.
- Empirical service simulation is intentional model mismatch.
- GP fitting may subsample for tractability, and only three representative scenarios are studied.
- Discrete capacity can prevent exact resource matching. Unmatched GP-UCB comparisons cannot establish equal-cost superiority.
""",
        encoding="utf-8",
    )
    Path("docs/REPRODUCIBILITY.md").write_text(
        """# Reproducibility

Docker Compose is the only host requirement. Run `python -m mfcontrol reproduce --profile paper` inside the research service. The bind-mounted pipeline writes timestamped `logs/reproduce_<timestamp>.log`, append-only `logs/events.jsonl`, and atomic `artifacts/progress.json`. `results/result_manifest.json` records code/config/lock hashes; `results/paired_controller_comparisons.csv` stores paired effect evidence. Raw Azure data is gitignored.
""",
        encoding="utf-8",
    )
    Path("artifacts/CV_BULLETS.md").write_text(
        """# CV wording

- Built a Docker-reproducible applied-mathematics study combining finite-$N$ JSQ queueing, mean-field fixed points, Gaussian-process uncertainty, and receding-horizon capacity control on public production-derived workload shapes.
- Implemented train/validation/test-isolated selection, validation-only resource matching, paired simulation, and artifact-backed conservative claims.

All results are controlled simulations, not a production deployment claim.
""",
        encoding="utf-8",
    )


def build_report(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "controller_metrics_per_seed.csv")
    summary = pd.read_csv(output_dir / "controller_summary.csv")
    forecast = pd.read_csv(output_dir / "forecast_metrics.csv")
    metadata = json.loads((output_dir / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    claims, effects = _claims(metrics, metadata)
    _write_effects(effects, output_dir)
    Path("docs/CLAIMS.md").write_text(claims, encoding="utf-8")
    Path("docs/CLAIMS_en.md").write_text(claims, encoding="utf-8")
    Path("docs/CLAIMS_ja.md").write_text(_claims_japanese(effects), encoding="utf-8")
    _write_report(effects, summary, forecast, metadata)
    _write_docs(effects)
    build_manifest(config, config_path, output_dir)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd="paper", check=True
    )
    Path("paper/report_en.pdf").write_bytes(Path("paper/report.pdf").read_bytes())
