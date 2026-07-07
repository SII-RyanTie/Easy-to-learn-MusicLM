import json
import csv
from pathlib import Path

"""
Evaluation Metrics Aggregator
-----------------------------
This script reads evaluation results from two separate pipelines:
1. Audiobox Aesthetics (JSONL): Computes CE, CU, PC, PQ for acoustic quality.
2. MuLan Similarity (CSV): Computes text-audio semantic alignment.

It aggregates the averages and generates a formatted Markdown table 
with 🥇 (Best) and 🥈 (Second Best) medals for easy pasting into README.md.
"""

# ==========================================
# TODO: Configuration & Model List
# Add your custom model names to this list.
# ==========================================
base_dir = Path(".")
audiobox_dir = base_dir / "audiobox_aesthetics"
mulan_dir = base_dir / "mulan_sim"

models = [
    "musicgen_small",
    "musicgen_medium",
    "musicgen_large",
    "stable_audio_open",
    "MusicTransformer_small",
    "MusicTransformer_small2"
]

metrics = ["CE", "CU", "PC", "PQ", "MuLan"]
results_data = []

# ==========================================
# 1. Parse Data from Evaluation Directories
# ==========================================
for model in models:
    jsonl_path = audiobox_dir / f"results_{model}.jsonl"
    csv_path = mulan_dir / f"{model}.csv"
    
    avgs = {}
    
    # --- A. Parse JSONL (Audiobox Metrics: CE, CU, PC, PQ) ---
    if jsonl_path.exists():
        totals = {"CE": 0.0, "CU": 0.0, "PC": 0.0, "PQ": 0.0}
        count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    for key in ["CE", "CU", "PC", "PQ"]:
                        totals[key] += data.get(key, 0.0)
                    count += 1
        if count > 0:
            for k in ["CE", "CU", "PC", "PQ"]:
                avgs[k] = totals[k] / count

    # --- B. Parse CSV (MuLan Semantic Similarity) ---
    if csv_path.exists():
        mulan_total = 0.0
        mulan_count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'similarity' in row:
                    mulan_total += float(row['similarity'])
                    mulan_count += 1
        if mulan_count > 0:
            avgs["MuLan"] = mulan_total / mulan_count

    # --- C. Store Aggregated Data ---
    if avgs:
        results_data.append({"model": model, "avgs": avgs})
    else:
        results_data.append({"model": model, "error": "Missing files"})

# ==========================================
# 2. Ranking Logic (Assigning 🥇 and 🥈)
# ==========================================
rankings = {}
for m in metrics:
    valid_vals = [item["avgs"][m] for item in results_data if "avgs" in item and m in item["avgs"]]
    if valid_vals:
        # Remove duplicates and sort descending
        sorted_vals = sorted(list(set(valid_vals)), reverse=True)
        best = sorted_vals[0] if len(sorted_vals) > 0 else None
        second_best = sorted_vals[1] if len(sorted_vals) > 1 else None
        rankings[m] = {"best": best, "second": second_best}
    else:
        rankings[m] = {"best": None, "second": None}

# ==========================================
# 3. Generate and Print Markdown Table
# ==========================================
print("| **Model Name**             | **CE Avg**   | **CU Avg**   | **PC Avg**   | **PQ Avg**   | **MuLan Avg**   |")
print("| :------------------------- | :----------- | :----------- | :----------- | :----------- | :-------------- |")

def format_val(val_dict, metric):
    """Formats the float value and appends a medal if applicable."""
    if metric not in val_dict:
        return "-          "
        
    val = val_dict[metric]
    # MuLan usually requires more precision (4 decimals), others use 2
    val_str = f"{val:.4f}" if metric == "MuLan" else f"{val:.2f}"
    
    if rankings[metric]["best"] is not None and val == rankings[metric]["best"]:
        return f"{val_str} 🥇"
    elif rankings[metric]["second"] is not None and val == rankings[metric]["second"]:
        return f"{val_str} 🥈"
    return f"{val_str}   "

for item in results_data:
    model_name = item["model"]
    if "error" in item:
        err = item["error"]
        print(f"| {model_name:<26} | {err:<12} | {'-':<12} | {'-':<12} | {'-':<12} | {'-':<15} |")
    else:
        avgs = item["avgs"]
        ce_str = format_val(avgs, "CE")
        cu_str = format_val(avgs, "CU")
        pc_str = format_val(avgs, "PC")
        pq_str = format_val(avgs, "PQ")
        mulan_str = format_val(avgs, "MuLan")
        
        print(f"| {model_name:<26} | {ce_str:<12} | {cu_str:<12} | {pc_str:<12} | {pq_str:<12} | {mulan_str:<15} |")
