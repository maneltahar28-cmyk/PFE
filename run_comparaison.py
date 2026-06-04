import json
import os
import sys
import traceback
from datetime import datetime

# ============================================================
# Racine du projet
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from baselines.random import RandomBaselineRunner
from baselines.aco import ACOStrategyRunner
from baselines.rl_local import RLLocalRunner
from baselines.madina_runner import MadinaSystemRunner # Votre système direct

METRICS = [
    "avg_reward",
    "avg_assigned",
    "avg_distance",
    "avg_search_time",
    "avg_occupancy",
    "avg_congestion",
    "avg_imbalance",
]

def safe_float(value, default=0.0):
    """Conversion robuste en float."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)

def empty_metrics():
    """Retourne un dictionnaire de métriques initialisé à zéro."""
    return {metric: 0.0 for metric in METRICS}

def extract_final_metrics(summary):
    """Normalise les sorties des différentes méthodes."""
    if not isinstance(summary, dict):
        return empty_metrics()

    # Si c'est le résultat de MadinaSystemRunner (dictionnaire plat)
    if "avg_reward" in summary and "final" not in summary:
        return {metric: safe_float(summary.get(metric, 0.0)) for metric in METRICS}

    # Si c'est le résultat des Baselines (dictionnaire avec clé "final")
    final = summary.get("final", {})
    return {metric: safe_float(final.get(metric, 0.0)) for metric in METRICS}

def build_success_result(method_name, summary):
    return {
        "status": "ok",
        "method": method_name,
        "metrics": extract_final_metrics(summary),
        "raw_summary": summary,
    }

def build_error_result(method_name, exc):
    return {
        "status": "error",
        "method": method_name,
        "metrics": empty_metrics(),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }

def run_method(method_name, runner_callable):
    print(f"\n================ {method_name} ================\n")
    try:
        summary = runner_callable()
        print(f"[OK] {method_name} terminé.")
        return build_success_result(method_name, summary)
    except Exception as e:
        print(f"[ERREUR] {method_name} a échoué : {e}")
        return build_error_result(method_name, e)

def build_plot_summary(all_results):
    compact = {}
    for method_name, payload in all_results.items():
        compact[method_name] = payload.get("metrics", empty_metrics())
    return compact

def check_required_paths(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Les fichiers suivants sont introuvables :\n" + "\n".join(missing))

def main():
    # Injection des paramètres S5 pour toutes les méthodes
    os.environ["MADINA_BEGIN"] = "21600"
    os.environ["MADINA_STEPS"] = "57600"

    sumo_cfg = os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "dua.static.sumocfg")
    agents_json = os.path.join(PROJECT_ROOT, "config", "agents_weighted_kmeans_balanced.json")
    parkings_xml = os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "parkings_min300.add.xml")

    check_required_paths(sumo_cfg, agents_json, parkings_xml)

    all_results = {}

    all_results["Random"] = run_method(
        "RANDOM",
        lambda: RandomBaselineRunner(
            sumo_cfg=sumo_cfg,
            agents_json=agents_json,
            parkings_xml=parkings_xml,
            max_rounds=1,
            steps_per_round=57600,
        ).run(use_gui=False)
    )

    all_results["ACO"] = run_method(
        "ACO",
        lambda: ACOStrategyRunner(
            sumo_cfg=sumo_cfg,
            agents_json=agents_json,
            parkings_xml=parkings_xml,
            max_rounds=1,
            steps_per_round=57600,
        ).run(use_gui=False)
    )

    all_results["RL_Local"] = run_method(
        "RL LOCAL",
        lambda: RLLocalRunner(
            sumo_cfg=sumo_cfg,
            agents_json=agents_json,
            parkings_xml=parkings_xml,
            episodes=1,
            steps_per_episode=57600,
        ).run(use_gui=False)
    )

    all_results["MADINA_SYSTEM"] = run_method(
        "MADINA SYSTEM",
        lambda: MadinaSystemRunner(PROJECT_ROOT).run()
    )

    out_dir = os.path.join(PROJECT_ROOT, "results", "comparison")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Rapport complet
    full_out_path = os.path.join(out_dir, f"comparison_full_{ts}.json")
    with open(full_out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # Rapport compact pour plot.py
    compact_summary = build_plot_summary(all_results)
    compact_out_path = os.path.join(out_dir, f"comparison_summary_{ts}.json")
    with open(compact_out_path, "w", encoding="utf-8") as f:
        json.dump(compact_summary, f, indent=4, ensure_ascii=False)

    print("\n================ RÉSUMÉ FINAL ================\n")
    print(json.dumps(all_results, indent=4, ensure_ascii=False))
    print(f"\nRapport complet sauvegardé dans : {full_out_path}")
    print(f"Résumé compact sauvegardé dans : {compact_out_path}")

if __name__ == "__main__":
    main()