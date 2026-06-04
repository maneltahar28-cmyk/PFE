import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "comparison")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

METRICS = [
    "avg_reward",
    "avg_assigned",
    "avg_distance",
    "avg_search_time",
    "avg_occupancy",
    "avg_congestion",
    "avg_imbalance",
]


def find_latest_summary():
    pattern = os.path.join(RESULTS_DIR, "comparison_summary_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier comparison_summary_*.json trouvé dans {RESULTS_DIR}"
        )
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_summary(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_method_name(name):
    mapping = {
        "Random": "Random",
        "ACO": "ACO",
        "RL_Local": "RL Local",
        "Federated_RL": "Federated RL",
    }
    return mapping.get(name, name)


def safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def build_metric_table(summary):
    raw_methods = list(summary.keys())
    methods = [normalize_method_name(k) for k in raw_methods]

    metric_table = {metric: [] for metric in METRICS}

    for raw_name in raw_methods:
        data = summary.get(raw_name, {})
        for metric in METRICS:
            metric_table[metric].append(safe_float(data.get(metric, 0.0)))

    return methods, metric_table


def save_bar_plot(methods, values, metric_name, output_dir):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    plt.bar(x, values)
    plt.xticks(x, methods, rotation=15)
    plt.title(f"Comparaison des méthodes - {metric_name}")
    plt.ylabel(metric_name)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{metric_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot sauvegardé: {out_path}")


def save_global_summary_plot(methods, metric_table, output_dir):
    metrics_lower_better = {
        "avg_distance",
        "avg_search_time",
        "avg_congestion",
        "avg_imbalance",
    }

    normalized_scores = {m: [] for m in methods}

    for metric, values in metric_table.items():
        arr = np.array(values, dtype=float)

        if np.allclose(arr.max(), arr.min()):
            norm = np.ones_like(arr) * 0.5
        else:
            norm = (arr - arr.min()) / (arr.max() - arr.min())

        if metric in metrics_lower_better:
            norm = 1.0 - norm

        for i, method in enumerate(methods):
            normalized_scores[method].append(float(norm[i]))

    plt.figure(figsize=(12, 7))
    x = np.arange(len(METRICS))
    width = 0.18

    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        plt.bar(x + offset, normalized_scores[method], width=width, label=method)

    plt.xticks(x, METRICS, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Score normalisé")
    plt.title("Résumé global comparatif des méthodes")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "global_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot global sauvegardé: {out_path}")


def print_table(methods, metric_table):
    print("\n===== TABLEAU RÉSUMÉ =====")
    header = ["Méthode"] + METRICS
    print(" | ".join(header))
    print("-" * 140)

    for i, method in enumerate(methods):
        row = [method]
        for metric in METRICS:
            row.append(f"{metric_table[metric][i]:.4f}")
        print(" | ".join(row))


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    latest_json = find_latest_summary()
    print(f"[INFO] Fichier utilisé: {latest_json}")

    summary = load_summary(latest_json)
    methods, metric_table = build_metric_table(summary)

    print_table(methods, metric_table)

    for metric in METRICS:
        save_bar_plot(
            methods=methods,
            values=metric_table[metric],
            metric_name=metric,
            output_dir=PLOTS_DIR,
        )

    save_global_summary_plot(
        methods=methods,
        metric_table=metric_table,
        output_dir=PLOTS_DIR,
    )

    print("\nTous les graphiques ont été générés avec succès.")


if __name__ == "__main__":
    main()