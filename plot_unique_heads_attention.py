"""Plot full attention density distributions for the 4 unique top-10 heads.

Heads:
- Only Dual: L0H2, L0H3  (per-layer bins from by_selected_kv_heads.csv)
- Only Ours: L24H3, L2H1 (KV-head-level bins from by_kv_id.csv, aggregated across layers)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

base = "/opt/data/private/yujun/myprogram/hermes"
obs_dir = f"{base}/results/observations/token_readout_attention_eager_gpu3_n2_o8_by_kvhead"

# ---- Load per-layer per-head attention bins (selected top/bottom heads) ----
df_sel = pd.read_csv(f"{obs_dir}/whole_cache_attention_by_selected_kv_heads.csv")

# ---- Load KV-head-level attention bins (aggregated across all layers) ----
df_kv = pd.read_csv(f"{obs_dir}/whole_cache_attention_by_kv_id.csv")

# ------------------------------------------------------------------
# Define the 4 target heads
# ------------------------------------------------------------------
targets = [
    {"layer": 0, "kv_head": 2, "label": "Layer 0, Head 2\n(Dual only, score=1.01)", "sel_name": "top_L0_KV2"},
    {"layer": 0, "kv_head": 3, "label": "Layer 0, Head 3\n(Dual only, score=0.857)", "sel_name": "top_L0_KV3"},
    {"layer": 24, "kv_head": 3, "label": "Layer 24, Head 3\n(Ours only, score=0.00339)", "sel_name": None, "kv_name": "kv3"},
    {"layer": 2, "kv_head": 1, "label": "Layer 2, Head 1\n(Ours only, score=0.00337)", "sel_name": None, "kv_name": "kv1"},
]

# ------------------------------------------------------------------
# Create 2x2 plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Get the overall global density for reference
df_global = pd.read_csv(f"{obs_dir}/whole_cache_token_attention_bins.csv")

for idx, t in enumerate(targets):
    ax = axes[idx]

    # Get per-head data
    if t["sel_name"] is not None and t["sel_name"] in df_sel["name"].values:
        df_head = df_sel[df_sel["name"] == t["sel_name"]].copy()
        source = "per-layer bins"
    elif t.get("kv_name") is not None:
        df_head = df_kv[df_kv["name"] == t["kv_name"]].copy()
        source = "KV-head-level bins (aggregated across layers)"
    else:
        ax.text(0.5, 0.5, f"No data available", transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(t["label"], fontsize=13)
        continue

    x = df_head["x"].values
    local = df_head["local_density"].values
    global_d = df_head["global_density"].values

    # Plot in log-log
    ax.loglog(x, local, '-', color='#2196F3', linewidth=2, alpha=0.9, label='This head (local)')
    ax.loglog(x, global_d, '--', color='gray', linewidth=1.2, alpha=0.7, label='Global baseline')

    # Fill between to show excess/deficit
    ax.fill_between(x, local, global_d, where=(local >= global_d),
                    color='#2196F3', alpha=0.15, interpolate=True, label='Above global')
    ax.fill_between(x, local, global_d, where=(local < global_d),
                    color='#FF5722', alpha=0.15, interpolate=True, label='Below global')

    ax.set_xlabel('Normalized KV position (0=recent, 1=distant)', fontsize=11)
    ax.set_ylabel('Attention density', fontsize=11)
    ax.set_title(t["label"] + f'\n[{source}]', fontsize=13)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Add text about the score
    ax.text(0.98, 0.02, f"Dual score: {df_head.get('mean_density', pd.Series([np.nan]*len(df_head))).values[0]:.3e}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.suptitle('Full Attention Density Distribution: Unique Top-10 Heads', fontsize=16, y=1.01)
plt.tight_layout()
output_path = f"{base}/results/observations/unique_heads_attention_distribution.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {output_path}")
plt.close()

# ------------------------------------------------------------------
# Also print key statistics for each head
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print("Key Statistics for the 4 unique heads:")
print("-" * 80)
for t in targets:
    print(f"\n--- {t['label'].split(chr(10))[0]} ---")
    if t["sel_name"] is not None and t["sel_name"] in df_sel["name"].values:
        df_h = df_sel[df_sel["name"] == t["sel_name"]]
        # Compute total mass
        local = df_h["local_density"].values
        total_mass = np.trapezoid(local, df_h["x"].values)
        # Peak
        peak_idx = np.argmax(local)
        print(f"  Total attention mass: {total_mass:.6f}")
        print(f"  Peak density: {local[peak_idx]:.2e} at position {df_h['x'].values[peak_idx]:.4f}")
        # Mass in first 10% vs last 90%
        mask_recent = df_h["x"].values <= 0.1
        mass_recent = np.trapezoid(local[mask_recent], df_h["x"].values[mask_recent])
        mask_distant = df_h["x"].values > 0.1
        mass_distant = np.trapezoid(local[mask_distant], df_h["x"].values[mask_distant])
        print(f"  Mass in recent 10% (x<=0.1): {mass_recent:.6f} ({100*mass_recent/total_mass:.1f}%)")
        print(f"  Mass in distant 90% (x>0.1):  {mass_distant:.6f} ({100*mass_distant/total_mass:.1f}%)")
    elif t.get("kv_name") is not None:
        df_h = df_kv[df_kv["name"] == t["kv_name"]]
        local = df_h["local_density"].values
        total_mass = np.trapezoid(local, df_h["x"].values)
        peak_idx = np.argmax(local)
        print(f"  Total attention mass: {total_mass:.6f}")
        print(f"  Peak density: {local[peak_idx]:.2e} at position {df_h['x'].values[peak_idx]:.4f}")
        print(f"  Note: KV-head-level data (aggregated across all 28 layers)")
