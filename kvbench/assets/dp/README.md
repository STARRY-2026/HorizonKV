# DP K=500 capacity table

`per_head_allocation_k500.csv` is a byte-identical copy of
`hermes/results/myobservations/two_bank_grid_dp_k500_floor375_20260925/per_head_allocation.csv`.
SHA-256: `fa164606d409a30acda7a6c881a9af24a2e956d7d494ff0417fc03aa90acff84`.

The explicit `kvbench --method dp` run loads `capacity_n` for all 28 × 4 KV
heads (56,000 visual slots at average K=500). Its online eviction is the
SparseMM attention selector. Although the CSV also includes `recent_r` and
`history_q`, this mode validates but does not execute that split. The full
two-bank OVO implementation remains in `sparsemm_native.run_ovobench --arm
dp_dual`, with its separate protocol and audit.

`per_head_allocation_dual_k500.csv` is a byte-identical copy of
`hermes/results/myobservations/two_bank_grid_dp_k500_floor375_recent32_20261020/per_head_allocation_fixed_capacity.csv`.
SHA-256: `957a8d89b79cae86d1390478a451e658932de5e4a990b9fa4813c353c815b317`.
The explicit `kvbench --method dp_dual` run executes `recent_r` and
`history_q` online using the native two-bank selector. Its audit checks
recorded token identities, bank quotas and the protected recent window.
