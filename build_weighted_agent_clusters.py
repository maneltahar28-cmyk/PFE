#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import re
import xml.etree.ElementTree as ET
import numpy as np


# =========================================================
# CONFIG
# =========================================================

NUM_AGENTS = 4
RANDOM_SEED = 42

# Poids du coût d'affectation
POSITION_COST_WEIGHT = 1.00
TRAFFIC_BALANCE_WEIGHT = 1.40
CAPACITY_BALANCE_WEIGHT = 0.90
SIZE_BALANCE_WEIGHT = 0.35

# Poids des features pour initialisation des centroïdes
POSITION_FEATURE_WEIGHT = 1.00
CAPACITY_FEATURE_WEIGHT = 0.30
TRAFFIC_FEATURE_WEIGHT = 0.50

OUTPUT_NAME = "agents_weighted_kmeans_balanced.json"


# =========================================================
# XML PARSING
# =========================================================

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

            if not lane_id or not shape:
                continue

            points = []
            for pair in shape.strip().split():
                try:
                    x_str, y_str = pair.split(",")
                    points.append((float(x_str), float(y_str)))
                except Exception:
                    continue

            if not points:
                continue

            mid = points[len(points) // 2]
            lane_midpoints[lane_id] = [mid[0], mid[1]]

    return lane_midpoints


def parse_parkings(parkings_xml_path, lane_midpoints):
    tree = ET.parse(parkings_xml_path)
    root = tree.getroot()

    parkings = []

    for elem in root.findall("parkingArea"):
        pid = elem.get("id")
        lane = elem.get("lane")
        cap = int(float(elem.get("roadsideCapacity", 1)))

        if not pid or not lane:
            continue

        if lane not in lane_midpoints:
            print(f"[WARN] parking {pid} ignoré : lane introuvable {lane}")
            continue

        x, y = lane_midpoints[lane]

        parkings.append({
            "parking_id": pid,
            "lane": lane,
            "edge": lane.rsplit("_", 1)[0] if "_" in lane else lane,
            "x": float(x),
            "y": float(y),
            "capacity": max(cap, 1),
            "traffic_score": 0.0,
        })

    return parkings


def select_preferred_route_files(routes_dir):
    """
    MODIFICATION STRATÉGIQUE V2 : 
    Force la sélection des fichiers continus originaux et ignore les versions shifted.
    """
    all_route_files = sorted(
        f for f in os.listdir(routes_dir)
        if f.endswith(".rou.xml")
    )

    local_route_pattern = re.compile(r"^(local\.\d+)(\.shifted)?\.rou\.xml$")
    local_variants = {}
    selected_files = []

    for file_name in all_route_files:
        match = local_route_pattern.match(file_name)
        if not match:
            continue

        base_name = match.group(1)
        variant_name = "shifted" if match.group(2) else "original"
        local_variants.setdefault(base_name, {})[variant_name] = file_name

    for base_name in sorted(local_variants):
        variants = local_variants[base_name]
        # RECTIFICATION : Priorité absolue à l'original continu de 24h
        preferred_file = variants.get("original") or variants.get("shifted")

        if preferred_file:
            selected_files.append(preferred_file)

        ignored_files = sorted(
            file_name for file_name in variants.values()
            if file_name != preferred_file
        )

        for ignored_file in ignored_files:
            print(
                f"[INFO] doublon ignore : {ignored_file} "
                f"(preference pour {preferred_file})"
            )

    return selected_files


def parse_routes_traffic(routes_dir, parkings, max_files=None):
    parking_edges = {p["edge"]: p["parking_id"] for p in parkings}
    traffic_count = {p["parking_id"]: 0 for p in parkings}

    if not os.path.exists(routes_dir):
        print(f"[WARN] dossier routes introuvable : {routes_dir}")
        return traffic_count

    rou_files = select_preferred_route_files(routes_dir)

    if max_files is not None:
        rou_files = rou_files[:max_files]

    for file_name in rou_files:
        file_path = os.path.join(routes_dir, file_name)
        print(f"[INFO] lecture trafic : {file_name}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for route in root.findall(".//route"):
                edges = route.get("edges", "")
                if not edges: continue
                for edge in edges.split():
                    if edge in parking_edges:
                        traffic_count[parking_edges[edge]] += 1
        except Exception as e:
            print(f"[WARN] impossible de lire {file_name} : {e}")
            continue

    # NOUVEAUTÉ : Intégration et parsing du fichier de flux exogène extra_flow.rou.xml
    project_root = os.path.dirname(routes_dir)
    extra_flow_file = os.path.join(project_root, "extra_flow.rou.xml")
    
    if os.path.exists(extra_flow_file):
        print(f"[INFO] lecture trafic additionnel : {os.path.basename(extra_flow_file)}")
        try:
            tree = ET.parse(extra_flow_file)
            root = tree.getroot()
            for flow in root.findall("flow"):
                from_edge = flow.get("from")
                to_edge = flow.get("to")
                vehs_per_hour = float(flow.get("vehsPerHour", 0))
                begin = float(flow.get("begin", 0))
                end = float(flow.get("end", 0))
                
                total_vehs = int(vehs_per_hour * ((end - begin) / 3600.0))
                
                # Affectation proportionnelle du trafic sur les parkings situés sur les axes du flux
                for edge in [from_edge, to_edge]:
                    if edge in parking_edges:
                        traffic_count[parking_edges[edge]] += total_vehs
        except Exception as e:
            print(f"[WARN] Impossible de parser extra_flow pour le clustering : {e}")

    return traffic_count


# =========================================================
# HELPERS
# =========================================================

def normalize(values):
    values = np.array(values, dtype=np.float64)
    if len(values) == 0: return values
    min_v = values.min()
    max_v = values.max()
    if abs(max_v - min_v) < 1e-12:
        return np.ones_like(values)
    return (values - min_v) / (max_v - min_v)


def safe_mean(points):
    if len(points) == 0: return None
    return np.mean(points, axis=0)


def euclidean(a, b):
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


# =========================================================
# FEATURE MATRIX
# =========================================================

def build_feature_matrix(parkings):
    xs = np.array([p["x"] for p in parkings], dtype=np.float64)
    ys = np.array([p["y"] for p in parkings], dtype=np.float64)
    caps = np.array([p["capacity"] for p in parkings], dtype=np.float64)
    traffic = np.array([p["traffic_score"] for p in parkings], dtype=np.float64)

    x_norm = normalize(xs)
    y_norm = normalize(ys)
    cap_norm = normalize(caps)
    traffic_norm = normalize(traffic)

    features = np.column_stack([
        POSITION_FEATURE_WEIGHT * x_norm,
        POSITION_FEATURE_WEIGHT * y_norm,
        CAPACITY_FEATURE_WEIGHT * cap_norm,
        TRAFFIC_FEATURE_WEIGHT * traffic_norm,
    ])

    return features


# =========================================================
# INITIAL CENTROIDS
# =========================================================

def init_weighted_centroids(features, k, seed=42):
    rng = np.random.default_rng(seed)
    n = len(features)

    if k >= n: return features.copy()
    indices = [rng.integers(0, n)]

    for _ in range(1, k):
        dists = np.array([
            min(np.linalg.norm(features[i] - features[j]) for j in indices)
            for i in range(n)
        ])
        probs = dists ** 2
        probs_sum = probs.sum()

        if probs_sum <= 0:
            idx = rng.integers(0, n)
        else:
            probs = probs / probs_sum
            idx = rng.choice(n, p=probs)

        indices.append(idx)

    return features[indices].copy()


# =========================================================
# BALANCED CLUSTERING
# =========================================================

def balanced_weighted_kmeans_with_loads(parkings, features, k=4, max_iters=50, seed=42):
    rng = np.random.default_rng(seed)
    n = len(parkings)

    if n == 0: raise ValueError("Aucun parking trouvé.")
    if k > n: k = n

    min_cluster_size = max(1, n // k - 1)
    max_cluster_size = math.ceil(n / k) + 1

    capacities = np.array([p["capacity"] for p in parkings], dtype=np.float64)
    traffics = np.array([p["traffic_score"] for p in parkings], dtype=np.float64)

    total_capacity = capacities.sum()
    total_traffic = traffics.sum()

    target_capacity = total_capacity / k if k > 0 else total_capacity
    target_traffic = total_traffic / k if k > 0 else total_traffic

    centroids = init_weighted_centroids(features, k=k, seed=seed)
    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iters):
        distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)

        cluster_sizes = [0 for _ in range(k)]
        cluster_capacity = [0.0 for _ in range(k)]
        cluster_traffic = [0.0 for _ in range(k)]
        new_labels = np.full(n, -1, dtype=int)

        order = np.argsort(-traffics)

        for i in order:
            best_cluster = None
            best_cost = None

            for c in range(k):
                if cluster_sizes[c] >= max_cluster_size: continue

                pos_cost = distances[i, c]
                new_cap = cluster_capacity[c] + capacities[i]
                new_trf = cluster_traffic[c] + traffics[i]
                new_size = cluster_sizes[c] + 1

                cap_penalty = abs(new_cap - target_capacity) / max(target_capacity, 1.0)
                trf_penalty = abs(new_trf - target_traffic) / max(target_traffic, 1.0)
                size_penalty = abs(new_size - (n / k)) / max((n / k), 1.0)

                total_cost = (
                    POSITION_COST_WEIGHT * pos_cost
                    + CAPACITY_BALANCE_WEIGHT * cap_penalty
                    + TRAFFIC_BALANCE_WEIGHT * trf_penalty
                    + SIZE_BALANCE_WEIGHT * size_penalty
                )

                if best_cost is None or total_cost < best_cost:
                    best_cost = total_cost
                    best_cluster = c

            if best_cluster is None:
                best_cluster = int(np.argmin(distances[i]))

            new_labels[i] = best_cluster
            cluster_sizes[best_cluster] += 1
            cluster_capacity[best_cluster] += capacities[i]
            cluster_traffic[best_cluster] += traffics[i]

        # Réparation si clusters trop petits
        for c in range(k):
            while cluster_sizes[c] < min_cluster_size:
                donor_candidates = [
                    i for i in range(n)
                    if new_labels[i] != c and cluster_sizes[new_labels[i]] > min_cluster_size
                ]

                if not donor_candidates: break

                best_i = None
                best_move_cost = None

                for i in donor_candidates:
                    old_c = new_labels[i]
                    current_cost = distances[i, old_c]
                    target_cost = distances[i, c]
                    move_penalty = target_cost - current_cost

                    if best_move_cost is None or move_penalty < best_move_cost:
                        best_move_cost = move_penalty
                        best_i = i

                if best_i is None: break

                old_c = new_labels[best_i]
                new_labels[best_i] = c

                cluster_sizes[c] += 1
                cluster_sizes[old_c] -= 1
                cluster_capacity[c] += capacities[best_i]
                cluster_capacity[old_c] -= capacities[best_i]
                cluster_traffic[c] += traffics[best_i]
                cluster_traffic[old_c] -= traffics[best_i]

        new_centroids = centroids.copy()
        for c in range(k):
            cluster_points = features[new_labels == c]
            if len(cluster_points) > 0:
                new_centroids[c] = cluster_points.mean(axis=0)

        if np.array_equal(labels, new_labels):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    return centroids, labels


# =========================================================
# OUTPUT
# =========================================================

def build_agents_payload(parkings, labels, centroids):
    agents = {}

    for c in range(len(centroids)):
        cluster_parkings = [
            p for i, p in enumerate(parkings)
            if int(labels[i]) == c
        ]

        if not cluster_parkings: continue

        centroid_x = float(np.mean([p["x"] for p in cluster_parkings]))
        centroid_y = float(np.mean([p["y"] for p in cluster_parkings]))

        total_capacity = int(sum(p["capacity"] for p in cluster_parkings))
        total_traffic = int(sum(p["traffic_score"] for p in cluster_parkings))

        agent_name = f"agent_{c + 1}"

        agents[agent_name] = {
            "centroid": [centroid_x, centroid_y],
            "seed_parkings": [p["parking_id"] for p in cluster_parkings],
            "total_capacity": total_capacity,
            "total_traffic_score": total_traffic,
        }

    return {
        "num_agents": len(agents),
        "clustering_method": "balanced_weighted_kmeans_with_loads",
        "features": [
            "position",
            "parking_capacity",
            "local_traffic_score"
        ],
        "agents": agents,
    }


def print_summary(payload):
    print("\n=== Résumé clustering pondéré équilibré ===")
    for agent_name, data in payload["agents"].items():
        print(
            f"{agent_name} | "
            f"parkings={len(data['seed_parkings'])} | "
            f"capacity={data['total_capacity']} | "
            f"traffic={data['total_traffic_score']} | "
            f"{data['seed_parkings']}"
        )


# =========================================================
# MAIN
# =========================================================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    net_xml = os.path.join(project_root, "scenarios", "luxembourg", "lust.net.xml")
    parkings_xml = os.path.join(project_root, "scenarios", "luxembourg", "parkings_min300.add.xml")

    if not os.path.exists(parkings_xml):
        parkings_xml = os.path.join(project_root, "scenarios", "luxembourg", "parkings.add.xml")

    routes_dir = os.path.join(project_root, "scenarios", "luxembourg", "DUARoutes")
    output_json = os.path.join(project_root, "config", OUTPUT_NAME)

    print("[1] Lecture réseau SUMO...")
    lane_midpoints = parse_net_lane_midpoints(net_xml)

    print("[2] Lecture parkings...")
    parkings = parse_parkings(parkings_xml, lane_midpoints)
    print(f"[INFO] parkings détectés : {len(parkings)}")

    print("[3] Estimation trafic local depuis DUARoutes...")
    traffic_count = parse_routes_traffic(routes_dir, parkings, max_files=None)

    for p in parkings:
        p["traffic_score"] = traffic_count.get(p["parking_id"], 0)

    print("[4] Construction features pondérées...")
    features = build_feature_matrix(parkings)

    print("[5] Clustering pondéré équilibré...")
    centroids, labels = balanced_weighted_kmeans_with_loads(
        parkings=parkings,
        features=features,
        k=NUM_AGENTS,
        max_iters=60,
        seed=RANDOM_SEED,
    )

    payload = build_agents_payload(parkings, labels, centroids)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    print_summary(payload)
    print(f"\n✅ Fichier généré : {output_json}")


if __name__ == "__main__":
    main()