import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return 0.0
    return float(s.mean())


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "distance_m",
        "price",
        "reward",
        "step",
        "free_slots",
        "capacity",
        "real_occupancy",
        "incoming_count",
        "predicted_occupancy",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "reason" in df.columns:
        df["assigned_flag"] = (
            df["reason"].astype(str).str.strip().str.lower() == "assigned"
        ).astype(int)
    elif "assigned" in df.columns:
        if df["assigned"].dtype == object:
            df["assigned_flag"] = (
                df["assigned"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["1", "true", "assigned", "yes"])
            ).astype(int)
        else:
            df["assigned_flag"] = (
                pd.to_numeric(df["assigned"], errors="coerce").fillna(0).astype(int)
            )
    else:
        df["assigned_flag"] = 0

    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip().str.lower()

    if "agent_name" in df.columns:
        df["agent_name"] = df["agent_name"].astype(str).str.strip()

    if "parking_id" in df.columns:
        df["parking_id"] = df["parking_id"].astype(str).str.strip()

    # Occupation observée
    if "real_occupancy" in df.columns and "capacity" in df.columns:
        df["occupancy_rate"] = df["real_occupancy"] / df["capacity"].replace(0, pd.NA)
    elif "free_slots" in df.columns and "capacity" in df.columns:
        df["occupancy_rate"] = (
            (df["capacity"] - df["free_slots"]) / df["capacity"].replace(0, pd.NA)
        )
    else:
        df["occupancy_rate"] = pd.NA

    # Occupation prédite
    if "predicted_occupancy" in df.columns and "capacity" in df.columns:
        df["predicted_occupancy_rate"] = (
            df["predicted_occupancy"] / df["capacity"].replace(0, pd.NA)
        )
    else:
        df["predicted_occupancy_rate"] = pd.NA

    # Efficacité
    if "reward" in df.columns and "distance_m" in df.columns:
        df["efficiency_score"] = df["reward"] / (df["distance_m"] + 1.0)
    else:
        df["efficiency_score"] = pd.NA

    return df


def save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" in df.columns:
        mode_summary = (
            df.dropna(subset=["mode"])
            .groupby("mode", dropna=True)
            .agg(
                decisions=("mode", "size"),
                avg_distance_m=("distance_m", lambda s: safe_mean(s)),
                avg_price=("price", lambda s: safe_mean(s)),
                avg_reward=("reward", lambda s: safe_mean(s)),
                success_rate=("assigned_flag", "mean"),
                avg_occupancy_rate=("occupancy_rate", lambda s: safe_mean(s)),
                avg_predicted_occupancy_rate=("predicted_occupancy_rate", lambda s: safe_mean(s)),
                avg_efficiency=("efficiency_score", lambda s: safe_mean(s)),
            )
            .reset_index()
        )

        mode_summary["success_rate"] = mode_summary["success_rate"] * 100.0
        mode_summary["avg_occupancy_rate"] = mode_summary["avg_occupancy_rate"] * 100.0
        mode_summary["avg_predicted_occupancy_rate"] = (
            mode_summary["avg_predicted_occupancy_rate"] * 100.0
        )
        mode_summary.to_csv(out_dir / "summary_by_mode.csv", index=False)

    overall_metrics = {
        "total_decisions": len(df),
        "overall_success_rate_percent": float(df["assigned_flag"].mean() * 100.0),
        "avg_distance_m": safe_mean(df["distance_m"]) if "distance_m" in df.columns else 0.0,
        "avg_price": safe_mean(df["price"]) if "price" in df.columns else 0.0,
        "avg_reward": safe_mean(df["reward"]) if "reward" in df.columns else 0.0,
        "avg_occupancy_rate_percent": safe_mean(df["occupancy_rate"]) * 100.0,
        "avg_predicted_occupancy_rate_percent": safe_mean(df["predicted_occupancy_rate"]) * 100.0,
        "avg_efficiency_score": safe_mean(df["efficiency_score"]),
    }

    overall = pd.DataFrame({
        "metric": list(overall_metrics.keys()),
        "value": list(overall_metrics.values()),
    })
    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    if "parking_id" in df.columns:
        parking_summary = (
            df.groupby("parking_id")
            .agg(
                decisions=("parking_id", "size"),
                avg_distance_m=("distance_m", lambda s: safe_mean(s)),
                avg_price=("price", lambda s: safe_mean(s)),
                avg_reward=("reward", lambda s: safe_mean(s)),
                avg_occupancy_rate=("occupancy_rate", lambda s: safe_mean(s)),
                avg_predicted_occupancy_rate=("predicted_occupancy_rate", lambda s: safe_mean(s)),
            )
            .reset_index()
            .sort_values(by="decisions", ascending=False)
        )
        parking_summary["avg_occupancy_rate"] = parking_summary["avg_occupancy_rate"] * 100.0
        parking_summary["avg_predicted_occupancy_rate"] = (
            parking_summary["avg_predicted_occupancy_rate"] * 100.0
        )
        parking_summary.to_csv(out_dir / "summary_by_parking.csv", index=False)

    if "agent_name" in df.columns:
        agent_summary = (
            df.groupby("agent_name")
            .agg(
                decisions=("agent_name", "size"),
                avg_distance_m=("distance_m", lambda s: safe_mean(s)),
                avg_price=("price", lambda s: safe_mean(s)),
                avg_reward=("reward", lambda s: safe_mean(s)),
                success_rate=("assigned_flag", "mean"),
                avg_occupancy_rate=("occupancy_rate", lambda s: safe_mean(s)),
                avg_predicted_occupancy_rate=("predicted_occupancy_rate", lambda s: safe_mean(s)),
            )
            .reset_index()
        )
        agent_summary["success_rate"] = agent_summary["success_rate"] * 100.0
        agent_summary["avg_occupancy_rate"] = agent_summary["avg_occupancy_rate"] * 100.0
        agent_summary["avg_predicted_occupancy_rate"] = (
            agent_summary["avg_predicted_occupancy_rate"] * 100.0
        )
        agent_summary.to_csv(out_dir / "summary_by_agent.csv", index=False)


def plot_distance_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    if "distance_m" not in df.columns:
        return
    plot_df = df.dropna(subset=["distance_m"])
    if plot_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(plot_df["distance_m"], bins=30)
    plt.xlabel("Distance au parking (m)")
    plt.ylabel("Nombre de décisions")
    plt.title("Histogramme des distances")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_distance.png", dpi=150)
    plt.close()


def plot_success_rate(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" not in df.columns:
        return

    success_by_mode = (
        df.dropna(subset=["mode"])
        .groupby("mode")["assigned_flag"]
        .mean()
        .sort_index()
        * 100.0
    )

    if success_by_mode.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(success_by_mode.index, success_by_mode.values)
    plt.xlabel("Mode")
    plt.ylabel("Taux de succès (%)")
    plt.title("Taux de succès par mode")
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate_by_mode.png", dpi=150)
    plt.close()


def plot_reward_over_time(df: pd.DataFrame, out_dir: Path) -> None:
    if "step" not in df.columns or "reward" not in df.columns:
        return

    reward_df = df.dropna(subset=["step", "reward"]).copy()
    if reward_df.empty:
        return

    reward_by_step = reward_df.groupby("step", as_index=False)["reward"].mean()

    plt.figure(figsize=(9, 5))
    plt.plot(reward_by_step["step"], reward_by_step["reward"])
    plt.xlabel("Step")
    plt.ylabel("Reward moyenne")
    plt.title("Évolution de la reward dans le temps")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_over_time.png", dpi=150)
    plt.close()


def plot_occupancy_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" not in df.columns or "occupancy_rate" not in df.columns:
        return

    occ = (
        df.dropna(subset=["mode", "occupancy_rate"])
        .groupby("mode")["occupancy_rate"]
        .mean()
        .sort_index()
        * 100.0
    )

    if occ.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(occ.index, occ.values)
    plt.xlabel("Mode")
    plt.ylabel("Taux d'occupation moyen (%)")
    plt.title("Occupation réelle moyenne par mode")
    plt.tight_layout()
    plt.savefig(out_dir / "occupancy_by_mode.png", dpi=150)
    plt.close()


def plot_predicted_occupancy_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" not in df.columns or "predicted_occupancy_rate" not in df.columns:
        return

    occ = (
        df.dropna(subset=["mode", "predicted_occupancy_rate"])
        .groupby("mode")["predicted_occupancy_rate"]
        .mean()
        .sort_index()
        * 100.0
    )

    if occ.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(occ.index, occ.values)
    plt.xlabel("Mode")
    plt.ylabel("Occupation prédite moyenne (%)")
    plt.title("Occupation prédite moyenne par mode")
    plt.tight_layout()
    plt.savefig(out_dir / "predicted_occupancy_by_mode.png", dpi=150)
    plt.close()


# =========================
# ANALYSE AJOUTÉE : AGENTS + PRIX + PARKINGS
# =========================

def plot_decisions_by_agent(df: pd.DataFrame, out_dir: Path) -> None:
    if "agent_name" not in df.columns:
        return

    stats = df.groupby("agent_name").size().sort_index()
    if stats.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats.values)
    plt.xlabel("Agent")
    plt.ylabel("Nombre de décisions")
    plt.title("Charge de travail par agent")
    plt.tight_layout()
    plt.savefig(out_dir / "decisions_by_agent.png", dpi=150)
    plt.close()


def plot_distance_by_agent(df: pd.DataFrame, out_dir: Path) -> None:
    if "agent_name" not in df.columns or "distance_m" not in df.columns:
        return

    stats = (
        df.dropna(subset=["agent_name", "distance_m"])
        .groupby("agent_name")["distance_m"]
        .mean()
        .sort_index()
    )
    if stats.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats.values)
    plt.xlabel("Agent")
    plt.ylabel("Distance moyenne (m)")
    plt.title("Distance moyenne par agent")
    plt.tight_layout()
    plt.savefig(out_dir / "distance_by_agent.png", dpi=150)
    plt.close()


def plot_reward_by_agent(df: pd.DataFrame, out_dir: Path) -> None:
    if "agent_name" not in df.columns or "reward" not in df.columns:
        return

    stats = (
        df.dropna(subset=["agent_name", "reward"])
        .groupby("agent_name")["reward"]
        .mean()
        .sort_index()
    )
    if stats.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats.values)
    plt.xlabel("Agent")
    plt.ylabel("Reward moyenne")
    plt.title("Reward moyenne par agent")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_by_agent.png", dpi=150)
    plt.close()


def plot_price_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" not in df.columns or "price" not in df.columns:
        return

    stats = (
        df.dropna(subset=["mode", "price"])
        .groupby("mode")["price"]
        .mean()
        .sort_index()
    )
    if stats.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats.values)
    plt.xlabel("Mode")
    plt.ylabel("Prix moyen")
    plt.title("Prix moyen par mode")
    plt.tight_layout()
    plt.savefig(out_dir / "price_by_mode.png", dpi=150)
    plt.close()


def plot_distance_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" not in df.columns or "distance_m" not in df.columns:
        return

    stats = (
        df.dropna(subset=["mode", "distance_m"])
        .groupby("mode")["distance_m"]
        .mean()
        .sort_index()
    )
    if stats.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats.values)
    plt.xlabel("Mode")
    plt.ylabel("Distance moyenne (m)")
    plt.title("Distance moyenne par mode")
    plt.tight_layout()
    plt.savefig(out_dir / "distance_by_mode.png", dpi=150)
    plt.close()


def plot_reward_by_mode(df: pd.DataFrame, out_dir: Path) -> None:
    if "mode" not in df.columns or "reward" not in df.columns:
        return

    stats = (
        df.dropna(subset=["mode", "reward"])
        .groupby("mode")["reward"]
        .mean()
        .sort_index()
    )
    if stats.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats.values)
    plt.xlabel("Mode")
    plt.ylabel("Reward moyenne")
    plt.title("Reward moyenne par mode")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_by_mode.png", dpi=150)
    plt.close()


def plot_top_parkings(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    if "parking_id" not in df.columns:
        return

    stats = df["parking_id"].value_counts().head(top_n)
    if stats.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(stats.index.astype(str), stats.values)
    plt.xlabel("Parking")
    plt.ylabel("Nombre de décisions")
    plt.title(f"Top {top_n} des parkings les plus sollicités")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "top_parkings.png", dpi=150)
    plt.close()


def plot_parking_predicted_occupancy(df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    if "parking_id" not in df.columns or "predicted_occupancy_rate" not in df.columns:
        return

    stats = (
        df.dropna(subset=["parking_id", "predicted_occupancy_rate"])
        .groupby("parking_id")["predicted_occupancy_rate"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        * 100.0
    )
    if stats.empty:
        return

    plt.figure(figsize=(11, 5))
    plt.bar(stats.index.astype(str), stats.values)
    plt.xlabel("Parking")
    plt.ylabel("Occupation prédite moyenne (%)")
    plt.title(f"Top {top_n} parkings par occupation prédite")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "parking_predicted_occupancy_top20.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse un fichier decisions_log.csv et génère des statistiques + graphes."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/decisions_log.csv",
        help="Chemin vers le fichier CSV de décisions",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/analysis",
        help="Dossier de sortie",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier CSV introuvable : {csv_path}")

    df = load_and_clean(csv_path)

    print(f"Nombre total de lignes : {len(df)}")
    print(f"Taux de succès global  : {df['assigned_flag'].mean() * 100:.2f}%")
    print(f"Distance moyenne       : {safe_mean(df['distance_m']) if 'distance_m' in df.columns else 0.0:.2f} m")
    print(f"Prix moyen             : {safe_mean(df['price']) if 'price' in df.columns else 0.0:.2f}")
    print(f"Reward moyenne         : {safe_mean(df['reward']) if 'reward' in df.columns else 0.0:.2f}")
    print(f"Occupation moyenne     : {safe_mean(df['occupancy_rate']) * 100.0:.2f}%")
    print(f"Occupation prédite     : {safe_mean(df['predicted_occupancy_rate']) * 100.0:.2f}%")
    print(f"Efficacité moyenne     : {safe_mean(df['efficiency_score']):.5f}")

    save_summary(df, out_dir)

    # Graphes existants
    plot_distance_histogram(df, out_dir)
    plot_success_rate(df, out_dir)
    plot_reward_over_time(df, out_dir)
    plot_occupancy_by_mode(df, out_dir)
    plot_predicted_occupancy_by_mode(df, out_dir)

    # Graphes ajoutés
    plot_decisions_by_agent(df, out_dir)
    plot_distance_by_agent(df, out_dir)
    plot_reward_by_agent(df, out_dir)
    plot_price_by_mode(df, out_dir)
    plot_distance_by_mode(df, out_dir)
    plot_reward_by_mode(df, out_dir)
    plot_top_parkings(df, out_dir)
    plot_parking_predicted_occupancy(df, out_dir)

    print("\nAnalyse terminée.")
    print(f"Résultats enregistrés dans : {out_dir}")


if __name__ == "__main__":
    main()