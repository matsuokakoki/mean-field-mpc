# Deviations and methodological updates

- The original bursty test split was empty. For the revised analysis, split-level eligibility criteria were fixed before rerunning controller evaluation and applied before training-only workload ranking.
- Coarse alpha selection was replaced by adaptive validation-only log-space resource matching. Unmatched discrete points stay in Pareto results but are excluded from strict claims.
- A high-volume EWMA-versus-reactive paired analysis and compact activation-delay robustness check were added after the original analysis; they do not alter workload selection or controller tuning.
- Tail metrics with too few jobs are NA with a reason, never zero.
- The public Azure RAR5 archive is extracted inside Docker using pinned 7-Zip.
- A representative-event cap can weight high-count bins; it preserves offered work but weakens individual-job tail interpretation and is recorded in the manifest.
