import json
import os
from datetime import datetime

import numpy as np

from env.multi_agent_env import MultiAgentParkingEnv


class ACOStrategyRunner:
    """
    Baseline inspirée Ant Colony Optimization (ACO) adaptée à la nouvelle API.
    Les phéromones sont mises à jour sur chaque parking selon la récompense.
    """

    def __init__(
        self,
        sumo_cfg,
        agents_json,
        parkings_xml,
        max_rounds=1,
        steps_per_round=40,
        alpha=1.0,
        beta=2.0,
        evaporation=0.10,
        pheromone_init=1.0,
        pheromone_deposit=0.25,
        q0=0.85,
    ):
        self.sumo_cfg = sumo_cfg
        self.agents_json = agents_json
        self.parkings_xml = parkings_xml
        self.max_rounds = max_rounds
        self.steps_per_round = steps_per_round
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromone_init = pheromone_init
        self.pheromone_deposit = pheromone_deposit
        self.q0 = q0

        with open(agents_json, "r", encoding="utf-8") as f:
            self.agents_data = json.load(f)

        # Assurer la présence de 'parkings' pour chaque zone
        for zone_name, zone_info in self.agents_data.get("agents", {}).items():
            if "parkings" not in zone_info:
                zone_info["parkings"] = list(zone_info.get("seed_parkings", []))

        # Phéromones initialisées pour chaque parking
        self.pheromones = {}
        for zone_info in self.agents_data["agents"].values():
            for pid in zone_info.get("parkings", []):
                self.pheromones[pid] = float(self.pheromone_init)

    def _evaporate_pheromones(self):
        """Applique l'évaporation globale des phéromones."""
        for pid in self.pheromones:
            self.pheromones[pid] = max(0.05, self.pheromones[pid] * (1.0 - self.evaporation))

    def _deposit_pheromone(self, parking_id, reward_value):
        """Dépose des phéromones sur un parking si la récompense est positive."""
        if reward_value > 0:
            scaled_reward = max(0.0, float(reward_value))
            self.pheromones[parking_id] = self.pheromones.get(parking_id, self.pheromone_init) \
                                          + self.pheromone_deposit * scaled_reward

    # ---------------------------------------------------------
    # Nouvelle version de run pour l'API reset()/step()
    # ---------------------------------------------------------
    def run(self, use_gui: bool = False):
        """
        Exécute ACO en utilisant reset()/step() sans notion de zones multiples.
        À chaque round, l'environnement est réinitialisé via reset() puis
        steps_per_round actions sont exécutées.
        """
        max_steps = int(os.environ.get("MADINA_STEPS", "57600"))
        env = MultiAgentParkingEnv(
            sumo_cfg=self.sumo_cfg,
            parkings_xml=self.parkings_xml,
            agents_json=self.agents_json,
            warmup_steps=500,
            max_steps=max_steps,
            max_assignments_per_step=8,
        )

        round_logs = []
        try:
            env.use_gui = bool(use_gui)
            env.start()

            for rnd in range(1, self.max_rounds + 1):
                self._evaporate_pheromones()

                state = env.reset()
                reward_sum = assigned_sum = dist_sum = occ_sum = congestion_sum = 0.0
                steps = 0

                for _ in range(self.steps_per_round):
                    candidates = getattr(env, "current_candidates", [])
                    if not candidates:
                        action_payload = {"parking_index": 0, "price_level": "standard"}
                        chosen_dist = 0.0
                        next_state, reward, done, info = env.step(action_payload)
                    else:
                        pids = [pid for (pid, _, _) in candidates]
                        dists = [dist for (_, dist, _) in candidates]

                        desirabilities = []
                        for pid, dist in zip(pids, dists):
                            tau = max(self.pheromones.get(pid, self.pheromone_init), 1e-8)
                            cap = max(env.parking_manager.get_capacity(pid), 1)
                            free_ratio = env.parking_manager.get_free_slots(pid) / cap
                            pred_occ_ratio = env._get_corrected_predicted_occupancy_ratio(pid)
                            real_occ_ratio = env.parking_manager.get_real_occupancy(pid) / cap
                            incoming_ratio = env.parking_manager.get_incoming_count(pid) / cap
                            dist_norm = min(float(dist) / 1300.0, 1.0)

                            heuristic = (
                                5.0 * free_ratio
                                + 1.2 * (1.0 - pred_occ_ratio)
                                + 0.4 * (1.0 - real_occ_ratio)
                                + 0.4 * (1.0 - incoming_ratio)
                                + 1.0 * (1.0 - dist_norm)
                            )
                            heuristic = max(heuristic, 1e-8)
                            desirabilities.append((tau ** self.alpha) * (heuristic ** self.beta))

                        if np.random.random() < self.q0:
                            best_idx = int(np.argmax(desirabilities))
                        else:
                            desir_sum = sum(desirabilities)
                            if desir_sum <= 0:
                                best_idx = int(np.random.randint(0, len(candidates)))
                            else:
                                probs = [d / desir_sum for d in desirabilities]
                                best_idx = int(np.random.choice(len(candidates), p=probs))

                        chosen_pid = pids[best_idx]
                        chosen_dist = float(dists[best_idx])
                        parking_index = best_idx
                        action_payload = {"parking_index": parking_index, "price_level": "standard"}
                        next_state, reward, done, info = env.step(action_payload)

                        # mise à jour des phéromones
                        self._deposit_pheromone(chosen_pid, reward)

                    reward_sum += float(reward)
                    assigned_sum += float(info.get("assigned", 0))
                    dist_sum += chosen_dist
                    occ_sum += float(env._get_average_system_occupancy())
                    congestion_sum += float(env.get_traffic_pressure())
                    steps += 1
                    state = next_state

                    if done:
                        break

                avg_reward = reward_sum / steps if steps > 0 else 0.0
                avg_assigned = assigned_sum / steps if steps > 0 else 0.0
                avg_distance = dist_sum / steps if steps > 0 else 0.0
                avg_occupancy = occ_sum / steps if steps > 0 else 0.0
                avg_congestion = congestion_sum / steps if steps > 0 else 0.0

                round_logs.append({
                    "round": rnd,
                    "avg_reward": float(avg_reward),
                    "avg_assigned": float(avg_assigned),
                    "avg_distance": float(avg_distance),
                    "avg_search_time": 0.0,
                    "avg_occupancy": float(avg_occupancy),
                    "avg_congestion": float(avg_congestion),
                    "avg_imbalance": 0.0,
                })

                print(f"[ACO] Round {rnd}/{self.max_rounds} terminé")
                if steps < self.steps_per_round:
                    break

            return self._save_results(round_logs)

        finally:
            env.close()

    def _save_results(self, round_logs):
        os.makedirs("results/comparison", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("results/comparison", f"aco_{ts}.json")

        summary = {
            "method": "ACO",
            "rounds": round_logs,
            "final": round_logs[-1] if round_logs else {},
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"[ACO] Résultats sauvegardés dans {out_path}")
        return summary