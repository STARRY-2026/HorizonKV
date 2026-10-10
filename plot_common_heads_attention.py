"""Plot full attention density distributions for the 8 common top-10 heads.

These heads appear in BOTH Dual and Ours top 10:
  kv_head=0: L2H0, L22H0
  kv_head=1: L19H1, L27H1
  kv_head=2: L14H2, L18H2, L19H2, L21H2

Data: KV-head-level attention bins from by_kv_id.csv (aggregated across all 28 layers).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

base = "/opt/data/private/yujun/myprogram/hermes"
obs_dir = f"{base}/results/observations/token_readout_attention_eager_gpu3_n2_o8_by_kvhead"

df_kv = pd.read_csv(f"{obs_dir}/whole_cache_attention_by_kv_id.csv")

# Each of the 8 common heads maps to a KV head
head_info = [
    {"kv_name": "kv0", "title": "KV Head 0", "layers": ["L2H0", "L22H0"],
     "scores": ["0.785 / 0.00342", "1.010 / 0.00441"]},
    {"kv_name": "kv1", "title": "KV Head 1", "layers": ["L19H1", "L27H1"],
     "scores": ["0.870 / 0.00379", "0.870 / 0.00379"]},
    {"kv_name": "kv2", "title": "KV Head 2", "layers": ["L14H2", "L18H2", "L19H2", "L21H2"],
     "scores": ["0.795 / 0.00347", "0.831 / 0.00362", "0.832 / 0.00363", "0.803 / 0.00350"]},
]

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for idx, info in enumerate(head_info):
    ax = axes[idx]
    df_h = df_kv[df_kv["name"] == info["kv_name"]]
    x = df_h["x"].values
    local = df_h["local_density"].values
    global_d = df_h["global_density"].values

    ax.loglog(x, local, '-', color='#2196F3', linewidth=2.2, alpha=0.9, label='This KV head')
    ax.loglog(x, global_d, '--', color='gray', linewidth=1.2, alpha=0.7, label='Global baseline')

    ax.fill_between(x, local, global_d, where=(local >= global_d),
                    color='#2196F3', alpha=0.12, interpolate=True)
    ax.fill_between(x, local, global_d, where=(local < global_d),
                    color='#FF5722', alpha=0.12, interpolate=True)

    ax.set_xlabel('Normalized KV position (0=recent, 1=distant)', fontsize=12)
    ax.set_ylabel('Attention density', fontsize=12)
    ax.set_title(f"{info['title']}  ({', '.join(info['layers'])})", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both', linestyle=':')

    # Annotate which heads & scores
    text_lines = [f"{info['layers'][i]}: Dual={info['scores'][i].split('/')[0].strip()}"
                  for i in range(len(info['layers']))]
    ax.text(0.02, 0.02, '\n'.join(text_lines),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9, color='#1565C0',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.85))

    # Compute stats
    total_mass = np.trapezoid(local, x)
    mask_recent = x <= 0.1
    mass_recent = np.trapezoid(local[mask_recent], x[mask_recent])
    peak_idx = np.argmax(local)
    print(f"\n--- {info['title']} ({', '.join(info['layers'])}) ---")
    print(f"  Total mass: {total_mass:.6f}")
    print(f"  Recent 10% mass: {mass_recent:.6f} ({100*mass_recent/total_mass:.1f}%)")
    print(f"  Peak density: {local[peak_idx]:.2e} at x={x[peak_idx]:.4f}")
    print(f"  Layers: {', '.join(info['layers'])}")
    print(f"  Dual scores: {', '.join([s.split('/')[0].strip() for s in info['scores']])}")

plt.suptitle('Common Top-10 Heads: Full Attention Density Distribution (Both Dual & Ours)', fontsize=16, y=1.02)
plt.tight_layout()
output_path = f"{base}/results/observations/common_heads_attention_distribution.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")
plt.close()
