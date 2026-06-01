import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.client_flower import ParkingFlowerClient


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_last(values, n=3):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values[-n:]))


def mean_first(values, n=3):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values[:n]))


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def build_env_config(top_k: int):
    parkings_xml = os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "parkings_min300.add.xml")
    if not os.path.exists(parkings_xml):
        parkings_xml = os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "parkings.add.xml")

    return {
        "sumo_cfg": os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "dua.static.sumocfg"),
        "parkings_xml": parkings_xml,
        "agents_json": os.path.join(PROJECT_ROOT, "config", "agents_weighted_kmeans_balanced.json"),
        "top_k": int(top_k),
        "max_steps": 3000,
        "agent_detection_radius": 1500.0,
    }


def pick_metric(metrics: dict, main_key: str, aliases=None, default=0.0):
    if aliases is None:
        aliases = []

    if main_key in metrics:
        return metrics[main_key]

    for key in aliases:
        if key in metrics:
            return metrics[key]

    return default


def run_one_experiment(
    top_k: int,
    seed: int,
    agent_name: str = "agent_1",
    num_rounds: int = 2,
    steps_per_round: int = 1500,
    eval_steps: int = 200,
    warmup_steps: int = 150,
):
    set_seed(seed)

    env_config = build_env_config(top_k=top_k)

    client = ParkingFlowerClient(
        env_config=env_config,
        client_id=f"topk_{top_k}_seed_{seed}",
        agent_name=agent_name,
    )

    params = client.get_parameters(config={})

    train_rewards = []
    train_assigned = []
    train_losses = []
    train_useful_steps = []

    eval_rewards = []
    eval_assigned = []
    eval_useful_steps = []

    for rnd in range(num_rounds):
        epsilon = max(0.10, 0.85 * (0.75 ** (rnd - 1)))

        params, _, fit_metrics = client.fit(
            params,
            {
                "epsilon": float(epsilon),
                "gamma": 0.99,
                "batch_size": 64,
                "steps_per_round": int(steps_per_round),
                "warmup_steps": int(warmup_steps),
                "min_useful_steps": 20,
                "max_total_sim_steps": int(steps_per_round * 4),
            },
        )

        train_reward = pick_metric(
            fit_metrics,
            "reward",
            aliases=["avg_reward", "total_reward"],
            default=0.0,
        )
        train_assigned_value = pick_metric(
            fit_metrics,
            "assigned",
            aliases=["assigned_count"],
            default=0.0,
        )
        train_loss = pick_metric(
            fit_metrics,
            "loss",
            aliases=["avg_loss"],
            default=0.0,
        )
        train_steps = pick_metric(
            fit_metrics,
            "effective_steps",
            aliases=["useful_steps"],
            default=0.0,
        )

        print(f"[DEBUG FIT] K={top_k} seed={seed} round={rnd + 1} metrics={fit_metrics}")

        train_rewards.append(float(train_reward))
        train_assigned.append(float(train_assigned_value))
        train_losses.append(float(train_loss))
        train_useful_steps.append(float(train_steps))

        _, _, eval_metrics = client.evaluate(
            params,
            {
                "eval_steps": int(eval_steps),
                "warmup_steps": int(warmup_steps),
                "max_total_sim_steps": int(eval_steps * 4),
            },
        )

        eval_reward = pick_metric(
            eval_metrics,
            "reward",
            aliases=["avg_reward", "total_reward"],
            default=0.0,
        )
        eval_assigned_value = pick_metric(
            eval_metrics,
            "assigned",
            aliases=["assigned_count"],
            default=0.0,
        )
        eval_steps_value = pick_metric(
            eval_metrics,
            "effective_eval_steps",
            aliases=["useful_steps"],
            default=0.0,
        )

        print(f"[DEBUG EVAL] K={top_k} seed={seed} round={rnd + 1} metrics={eval_metrics}")

        eval_rewards.append(float(eval_reward))
        eval_assigned.append(float(eval_assigned_value))
        eval_useful_steps.append(float(eval_steps_value))

    if (
        all(v == 0.0 for v in train_rewards)
        and all(v == 0.0 for v in eval_rewards)
        and all(v == 0.0 for v in train_assigned)
        and all(v == 0.0 for v in eval_assigned)
    ):
        print(f"[WARNING] Toutes les métriques sont nulles pour K={top_k}, seed={seed}.")

    summary = {
        "top_k": int(top_k),
        "seed": int(seed),
        "agent_name": agent_name,
        "num_rounds": int(num_rounds),
        "train_reward_first3_mean": mean_first(train_rewards, 3),
        "train_reward_last3_mean": mean_last(train_rewards, 3),
        "train_reward_gain": mean_last(train_rewards, 3) - mean_first(train_rewards, 3),
        "train_reward_std": safe_std(train_rewards),
        "eval_reward_first3_mean": mean_first(eval_rewards, 3),
        "eval_reward_last3_mean": mean_last(eval_rewards, 3),
        "eval_reward_gain": mean_last(eval_rewards, 3) - mean_first(eval_rewards, 3),
        "eval_reward_std": safe_std(eval_rewards),
        "train_assigned_last3_mean": mean_last(train_assigned, 3),
        "eval_assigned_last3_mean": mean_last(eval_assigned, 3),
        "train_loss_last3_mean": mean_last(train_losses, 3),
        "train_loss_std": safe_std(train_losses),
        "train_useful_steps_last3_mean": mean_last(train_useful_steps, 3),
        "eval_useful_steps_last3_mean": mean_last(eval_useful_steps, 3),
    }

    curves = {
        "train_rewards": train_rewards,
        "train_assigned": train_assigned,
        "train_losses": train_losses,
        "train_useful_steps": train_useful_steps,
        "eval_rewards": eval_rewards,
        "eval_assigned": eval_assigned,
        "eval_useful_steps": eval_useful_steps,
    }

    return summary, curves


def normalize_column(values, higher_is_better=True):
    arr = np.array(values, dtype=float)
    if np.allclose(arr.max(), arr.min()):
        scores = np.ones_like(arr)
    else:
        scores = (arr - arr.min()) / (arr.max() - arr.min())
    if not higher_is_better:
        scores = 1.0 - scores
    return scores


def build_final_ranking(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("top_k", as_index=False)
        .agg(
            train_reward_last3_mean=("train_reward_last3_mean", "mean"),
            eval_reward_last3_mean=("eval_reward_last3_mean", "mean"),
            train_reward_gain=("train_reward_gain", "mean"),
            eval_reward_gain=("eval_reward_gain", "mean"),
            train_reward_std=("train_reward_std", "mean"),
            eval_reward_std=("eval_reward_std", "mean"),
            train_assigned_last3_mean=("train_assigned_last3_mean", "mean"),
            eval_assigned_last3_mean=("eval_assigned_last3_mean", "mean"),
            train_useful_steps_last3_mean=("train_useful_steps_last3_mean", "mean"),
            eval_useful_steps_last3_mean=("eval_useful_steps_last3_mean", "mean"),
        )
        .copy()
    )

    grouped["score_train_perf"] = normalize_column(grouped["train_reward_last3_mean"], True)
    grouped["score_eval_perf"] = normalize_column(grouped["eval_reward_last3_mean"], True)
    grouped["score_train_conv"] = normalize_column(grouped["train_reward_gain"], True)
    grouped["score_eval_conv"] = normalize_column(grouped["eval_reward_gain"], True)
    grouped["score_train_stability"] = normalize_column(grouped["train_reward_std"], False)
    grouped["score_eval_stability"] = normalize_column(grouped["eval_reward_std"], False)
    grouped["score_assigned"] = normalize_column(grouped["eval_assigned_last3_mean"], True)

    grouped["composite_score"] = (
        0.25 * grouped["score_eval_perf"] +
        0.20 * grouped["score_train_perf"] +
        0.15 * grouped["score_eval_conv"] +
        0.10 * grouped["score_train_conv"] +
        0.10 * grouped["score_train_stability"] +
        0.10 * grouped["score_eval_stability"] +
        0.10 * grouped["score_assigned"]
    )

    grouped = grouped.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return grouped


def plot_metric_by_k(df_grouped: pd.DataFrame, column: str, ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.bar(df_grouped["top_k"].astype(str), df_grouped[column])
    plt.xlabel("K")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_learning_curves(all_curves, out_dir: Path):
    plt.figure(figsize=(9, 5))
    for top_k, curves_list in all_curves.items():
        max_len = max(len(c["train_rewards"]) for c in curves_list)
        mat = []
        for c in curves_list:
            arr = c["train_rewards"]
            if len(arr) < max_len and len(arr) > 0:
                arr = arr + [arr[-1]] * (max_len - len(arr))
            mat.append(arr)
        mean_curve = np.mean(np.array(mat), axis=0)
        plt.plot(range(1, len(mean_curve) + 1), mean_curve, label=f"K={top_k}")
    plt.xlabel("Round local")
    plt.ylabel("Train reward moyenne")
    plt.title("Convergence des rewards d'entraînement selon K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_reward_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for top_k, curves_list in all_curves.items():
        max_len = max(len(c["eval_rewards"]) for c in curves_list)
        mat = []
        for c in curves_list:
            arr = c["eval_rewards"]
            if len(arr) < max_len and len(arr) > 0:
                arr = arr + [arr[-1]] * (max_len - len(arr))
            mat.append(arr)
        mean_curve = np.mean(np.array(mat), axis=0)
        plt.plot(range(1, len(mean_curve) + 1), mean_curve, label=f"K={top_k}")
    plt.xlabel("Round local")
    plt.ylabel("Eval reward moyenne")
    plt.title("Convergence des rewards d'évaluation selon K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eval_reward_curves.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare expérimentalement K = 3, 5, 7 pour top-K candidats."
    )
    parser.add_argument("--agent", type=str, default="agent_1", help="Agent testé")
    parser.add_argument("--repeats", type=int, default=3, help="Nombre de répétitions par K")
    parser.add_argument("--rounds", type=int, default=6, help="Nombre de rounds locaux")
    parser.add_argument("--steps", type=int, default=120, help="Nombre de steps utiles par round")
    parser.add_argument("--eval_steps", type=int, default=200, help="Nombre de steps utiles en évaluation")
    parser.add_argument("--warmup_steps", type=int, default=150, help="Warmup SUMO avant mesure")
    parser.add_argument("--out", type=str, default="outputs/topk_comparison", help="Dossier de sortie")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    topk_values = [3, 5, 7]
    seeds = [42 + i for i in range(args.repeats)]

    all_rows = []
    all_curves = {k: [] for k in topk_values}

    for top_k in topk_values:
        print(f"\n=== TEST K = {top_k} ===")
        for seed in seeds:
            print(f"  -> repeat seed={seed}")
            summary, curves = run_one_experiment(
                top_k=top_k,
                seed=seed,
                agent_name=args.agent,
                num_rounds=args.rounds,
                steps_per_round=args.steps,
                eval_steps=args.eval_steps,
                warmup_steps=args.warmup_steps,
            )
            all_rows.append(summary)
            all_curves[top_k].append(curves)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "topk_raw_results.csv", index=False)

    grouped = build_final_ranking(df)
    grouped.to_csv(out_dir / "topk_summary.csv", index=False)

    plot_metric_by_k(
        grouped,
        "train_reward_last3_mean",
        "Train reward finale moyenne",
        "Performance finale d'entraînement selon K",
        out_dir / "train_reward_by_k.png",
    )
    plot_metric_by_k(
        grouped,
        "eval_reward_last3_mean",
        "Eval reward finale moyenne",
        "Performance finale d'évaluation selon K",
        out_dir / "eval_reward_by_k.png",
    )
    plot_metric_by_k(
        grouped,
        "train_reward_gain",
        "Gain de reward",
        "Convergence d'entraînement selon K",
        out_dir / "train_convergence_by_k.png",
    )
    plot_metric_by_k(
        grouped,
        "eval_reward_gain",
        "Gain de reward",
        "Convergence d'évaluation selon K",
        out_dir / "eval_convergence_by_k.png",
    )
    plot_metric_by_k(
        grouped,
        "train_reward_std",
        "Écart-type reward train",
        "Stabilité d'entraînement selon K",
        out_dir / "train_stability_by_k.png",
    )
    plot_metric_by_k(
        grouped,
        "eval_reward_std",
        "Écart-type reward eval",
        "Stabilité d'évaluation selon K",
        out_dir / "eval_stability_by_k.png",
    )
    plot_metric_by_k(
        grouped,
        "composite_score",
        "Score composite",
        "Classement final des valeurs de K",
        out_dir / "composite_score_by_k.png",
    )

    plot_learning_curves(all_curves, out_dir)

    best_k = int(grouped.iloc[0]["top_k"])

    report = {
        "tested_k": topk_values,
        "repeats": args.repeats,
        "rounds": args.rounds,
        "steps_per_round": args.steps,
        "eval_steps": args.eval_steps,
        "warmup_steps": args.warmup_steps,
        "best_k": best_k,
        "ranking": grouped.to_dict(orient="records"),
    }

    with open(out_dir / "topk_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n=== RÉSULTAT FINAL ===")
    print(grouped[[
        "top_k",
        "train_reward_last3_mean",
        "eval_reward_last3_mean",
        "train_reward_gain",
        "eval_reward_gain",
        "train_reward_std",
        "eval_reward_std",
        "eval_assigned_last3_mean",
        "train_useful_steps_last3_mean",
        "eval_useful_steps_last3_mean",
        "composite_score",
    ]])

    print(f"\nK optimal proposé = {best_k}")
    print(f"Résultats enregistrés dans : {out_dir}")


if __name__ == "__main__":
    main()