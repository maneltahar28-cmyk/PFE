import json
import math
from typing import Dict, List, Tuple


class AgentMapper:
    """
    Mapping intelligent agent <-> véhicule basé sur :
    - distance au centroïde
    - parkings seeds associés à chaque agent
    - possibilité de retourner plusieurs agents candidats

    Version adaptée à 4 agents :
    agent_1, agent_2, agent_3, agent_4
    """

    def __init__(self, agents_json_path: str):
        self.agents_json_path = agents_json_path
        self.agents_data = self._load_agents()

    def _load_agents(self) -> Dict:
        with open(self.agents_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "agents" not in data:
            raise ValueError("Le fichier agents_kmeans.json doit contenir une clé 'agents'.")

        required_agents = {"agent_1", "agent_2", "agent_3", "agent_4"}
        found_agents = set(data["agents"].keys())

        if found_agents != required_agents:
            raise ValueError(
                f"agents_kmeans.json doit contenir exactement {sorted(required_agents)}, "
                f"mais contient {sorted(found_agents)}"
            )

        return data

    def get_agent_names(self) -> List[str]:
        """
        Retourne toujours les agents dans l'ordre agent_1 -> agent_4
        """
        return sorted(
            self.agents_data["agents"].keys(),
            key=lambda name: int(name.split("_")[1])
        )

    def get_agent_centroid(self, agent_name: str) -> Tuple[float, float]:
        centroid = self.agents_data["agents"][agent_name]["centroid"]
        return float(centroid[0]), float(centroid[1])

    def get_agent_seed_parkings(self, agent_name: str) -> List[str]:
        return list(self.agents_data["agents"][agent_name].get("seed_parkings", []))

    def _euclidean(self, p1, p2) -> float:
        return math.sqrt(
            (float(p1[0]) - float(p2[0])) ** 2 +
            (float(p1[1]) - float(p2[1])) ** 2
        )

    def get_nearest_agent(self, vehicle_pos: Tuple[float, float]) -> Tuple[str, float]:
        best_agent = None
        best_dist = float("inf")

        for agent_name in self.get_agent_names():
            centroid = self.get_agent_centroid(agent_name)
            dist = self._euclidean(vehicle_pos, centroid)
            if dist < best_dist:
                best_dist = dist
                best_agent = agent_name

        return best_agent, float(best_dist)

    def get_top_k_agents(
        self,
        vehicle_pos: Tuple[float, float],
        k: int = 2
    ) -> List[Tuple[str, float]]:
        ranked = []

        for agent_name in self.get_agent_names():
            centroid = self.get_agent_centroid(agent_name)
            dist = self._euclidean(vehicle_pos, centroid)
            ranked.append((agent_name, float(dist)))

        ranked.sort(key=lambda x: x[1])
        return ranked[:k]

    def score_agents_with_seed_bonus(
        self,
        vehicle_pos: Tuple[float, float],
        parking_positions: Dict[str, Tuple[float, float]],
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Score agent = distance au centroïde - bonus si ses parkings seeds sont proches.
        Plus petit = meilleur agent.
        """
        scored = []

        for agent_name in self.get_agent_names():
            centroid = self.get_agent_centroid(agent_name)
            centroid_dist = self._euclidean(vehicle_pos, centroid)

            seed_parkings = self.get_agent_seed_parkings(agent_name)
            seed_dists = []

            for pid in seed_parkings:
                if pid in parking_positions:
                    seed_dists.append(self._euclidean(vehicle_pos, parking_positions[pid]))

            if len(seed_dists) > 0:
                min_seed_dist = min(seed_dists)
                seed_bonus = min_seed_dist * 0.25
            else:
                seed_bonus = 0.0

            final_score = centroid_dist - seed_bonus
            scored.append((agent_name, float(final_score)))

        scored.sort(key=lambda x: x[1])
        return scored[:k]