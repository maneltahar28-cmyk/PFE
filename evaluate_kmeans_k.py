import os
import json
import math
import xml.etree.ElementTree as ET
from itertools import combinations

import numpy as np


def parse_net_lane_midpoints(net_xml_path):
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    lane_midpoints = {}

    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue

        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            shape = lane.get("shape")

            if lane_id is None or shape is None:
                continue

            pts = []
            for pair in shape.strip().split():
                x_str, y_str = pair.split(",")
                pts.append((float(x_str), float(y_str)))

            if not pts:
                continue

            mid = pts[len(pts) // 2]
            lane_midpoints[lane_id] = [mid[0], mid[1]]

    return lane_midpoints


def parse_parkings(parkings_xml_path, lane_midpoints):
    tree = ET.parse(parkings_xml_path)
    root = tree.getroot()

    parkings = []
    for elem in root.findall("parkingArea"):
        pid = elem.get("id")
        lane = elem.get("lane")

        if pid is None or lane is None:
            continue

        if lane not in lane_midpoints:
            continue

        x, y = lane_midpoints[lane]
        parkings.append({
            "parking_id": pid,
            "lane": lane,
            "x": float(x),
            "y": float(y),
        })

    return parkings


def run_kmeans(points, k=3, max_iters=100, seed=42):
    rng = np.random.default_rng(seed)
    n = len(points)

    if n == 0:
        raise ValueError("Aucun point pour K-means.")
    if k <= 0:
        raise ValueError("k doit être > 0.")
    if k > n:
        k = n

    indices = rng.choice(n, size=k, replace=False)
    centroids = points[indices].copy()

    for _ in range(max_iters):
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()
        for j in range(k):
            cluster_points = points[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)

        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, labels


def compute_inertia(points, centroids, labels):
    inertia = 0.0
    for i, p in enumerate(points):
        c = centroids[labels[i]]
        inertia += float(np.sum((p - c) ** 2))
    return inertia


def compute_cluster_sizes(labels, k):
    sizes = []
    for j in range(k):
        sizes.append(int(np.sum(labels == j)))
    return sizes


def compute_balance_score(cluster_sizes):
    sizes = np.array(cluster_sizes, dtype=np.float64)
    mean_size = sizes.mean()
    std_size = sizes.std()

    if mean_size == 0:
        return 0.0

    cv = std_size / mean_size
    score = 1.0 / (1.0 + cv)
    return float(score)


def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist


def compute_silhouette_score(points, labels, k):
    n = len(points)
    if n < 2:
        return 0.0

    dmat = pairwise_distances(points)
    silhouettes = []

    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = [c for c in range(k) if c != labels[i]]

        same_idx = np.where(same_cluster)[0]
        same_idx = same_idx[same_idx != i]

        if len(same_idx) == 0:
            a_i = 0.0
        else:
            a_i = float(np.mean(dmat[i, same_idx]))

        b_i = float("inf")
        for c in other_clusters:
            other_idx = np.where(labels == c)[0]
            if len(other_idx) == 0:
                continue
            dist_mean = float(np.mean(dmat[i, other_idx]))
            if dist_mean < b_i:
                b_i = dist_mean

        if b_i == float("inf"):
            b_i = 0.0

        denom = max(a_i, b_i)
        if denom == 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / denom

        silhouettes.append(s_i)

    return float(np.mean(silhouettes))


def compute_centroid_separation(centroids):
    if len(centroids) < 2:
        return 0.0

    dists = []
    for i, j in combinations(range(len(centroids)), 2):
        d = np.linalg.norm(centroids[i] - centroids[j])
        dists.append(float(d))

    return float(np.mean(dists))


def normalize_series(values, reverse=False):
    arr = np.array(values, dtype=np.float64)
    vmin = arr.min()
    vmax = arr.max()

    if np.isclose(vmax, vmin):
        scores = np.ones_like(arr)
    else:
        scores = (arr - vmin) / (vmax - vmin)

    if reverse:
        scores = 1.0 - scores

    return scores.tolist()


def build_cluster_payload(parkings, centroids, labels):
    agents = {}
    k = len(centroids)

    for j in range(k):
        agent_name = f"agent_{j + 1}"
        centroid = centroids[j].tolist()

        assigned_parkings = [
            parkings[i]["parking_id"]
            for i in range(len(parkings))
            if int(labels[i]) == j
        ]

        agents[agent_name] = {
            "centroid": [float(centroid[0]), float(centroid[1])],
            "seed_parkings": assigned_parkings,
        }

    return {
        "num_agents": k,
        "agents": agents,
    }


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    net_xml = os.path.join(project_root, "scenarios", "luxembourg", "lust.net.xml")
    parkings_xml = os.path.join(project_root, "scenarios", "luxembourg", "parkings.add.xml")
    output_dir = os.path.join(project_root, "outputs", "kmeans_model_selection")

    os.makedirs(output_dir, exist_ok=True)

    lane_midpoints = parse_net_lane_midpoints(net_xml)
    parkings = parse_parkings(parkings_xml, lane_midpoints)

    points = np.array([[p["x"], p["y"]] for p in parkings], dtype=np.float64)

    k_values = [2, 3, 4, 5]
    results = []

    for k in k_values:
        centroids, labels = run_kmeans(points, k=k, max_iters=100, seed=42)

        inertia = compute_inertia(points, centroids, labels)
        silhouette = compute_silhouette_score(points, labels, k)
        sizes = compute_cluster_sizes(labels, k)
        balance = compute_balance_score(sizes)
        centroid_sep = compute_centroid_separation(centroids)

        payload = build_cluster_payload(parkings, centroids, labels)

        json_path = os.path.join(output_dir, f"agents_kmeans_k{k}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        results.append({
            "k": k,
            "inertia": inertia,
            "silhouette": silhouette,
            "balance_score": balance,
            "centroid_separation": centroid_sep,
            "cluster_sizes": sizes,
            "json_path": json_path,
        })

    inertia_scores = normalize_series([r["inertia"] for r in results], reverse=True)
    sil_scores = normalize_series([r["silhouette"] for r in results], reverse=False)
    bal_scores = normalize_series([r["balance_score"] for r in results], reverse=False)
    sep_scores = normalize_series([r["centroid_separation"] for r in results], reverse=False)

    for i, r in enumerate(results):
        composite = (
            0.35 * sil_scores[i] +
            0.25 * bal_scores[i] +
            0.20 * sep_scores[i] +
            0.20 * inertia_scores[i]
        )
        r["composite_score"] = float(composite)

    results = sorted(results, key=lambda x: x["composite_score"], reverse=True)

    summary_path = os.path.join(output_dir, "kmeans_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n=== COMPARAISON SCIENTIFIQUE DES VALEURS DE K ===\n")
    for r in results:
        print(f"k = {r['k']}")
        print(f"  inertia             = {r['inertia']:.2f}")
        print(f"  silhouette          = {r['silhouette']:.4f}")
        print(f"  balance_score       = {r['balance_score']:.4f}")
        print(f"  centroid_separation = {r['centroid_separation']:.2f}")
        print(f"  cluster_sizes       = {r['cluster_sizes']}")
        print(f"  composite_score     = {r['composite_score']:.4f}")
        print(f"  json                = {r['json_path']}")
        print()

    best = results[0]
    print("=== MEILLEURE VALEUR SELON LE SCORE COMPOSITE ===")
    print(f"k optimal proposé = {best['k']}")
    print(f"Fichier cluster    = {best['json_path']}")
    print(f"Résumé complet     = {summary_path}")


if __name__ == "__main__":
    main()