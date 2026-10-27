# Recent one/four-frame capped-history allocation source

`per_head_allocations.csv` is a byte-identical copy of
`hermes/results/myobservations/lukv_samecalib_two_bank_k1000_20261022/per_head_allocations.csv`.
SHA-256: `acd6ab409a6b83dfa42535a4298e714565c1d7ecba401b9c2033d5da7bbb6fdf`.

The bounded `kvbench run-ovo --method recent1or4_history_cap4000` worker points
the unchanged `sparsemm_native.capped_history` allocator at this local copy.
The cap is 4,000 history tokens per head, with exactly 108,069 history slots
over 112 heads. The 56 one-frame and 56 four-frame heads are selected from
the frozen recent-share ordering. The official OVO annotation and model
weights remain external resources.
