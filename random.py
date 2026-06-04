import json
import os
import random
from datetime import datetime

import numpy as np

from env.multi_agent_env import MultiAgentParkingEnv


class RandomBaselineRunner:
    """
    Baseline aléatoire adaptée à la nouvelle API reset()/step(action_payload).
    """

    def __init__(self, sumo_cfg, agents_json, parkings_xml, max_rounds=1, steps_per_round=40):
        self.sumo_cfg = sumo_cfg
        self.agents_json = agents_json
        self.parkings_xml = parkings_xml
        self.max_rounds = max_rounds
        self.steps_per_round = steps_per_round

        with open(agents_json, "r", encoding="utf-8") as f:
            self.agents_data = json.load(f)

    def run(self, use_gui=False):
        """Exécute la baseline aléatoire sur plusieurs rounds avec la nouvelle API."""
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
                state = env.reset()
                reward_sum = assigned_sum = dist_sum = occ_sum = congestion_sum = 0.0
                steps = 0

                while True:
                    candidates = getattr(env, "current_candidates", [])

                    if candidates:
                        idx = random.randint(0, len(candidates) - 1)
                        chosen_dist = candidates[idx][1]
                    else:
                        idx = 0
                        chosen_dist = 0.0

                    action_payload = {"parking_index": idx, "price_level": "standard"}
                    next_state, reward, done, info = env.step(action_payload)

                    reward_sum += float(reward)
                    assigned_sum += float(info.get("assigned", 0))
                    dist_sum += float(chosen_dist)
                    occ_sum += float(env._get_average_system_occupancy())
                    congestion_sum += float(env.get_traffic_pressure())
                    steps += 1

                    state = next_state
                    if done or steps >= self.steps_per_round:
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
                    "avg_search_time": 0.0,   # non disponible
                    "avg_occupancy": float(avg_occupancy),
                    "avg_congestion": float(avg_congestion),
                    "avg_imbalance": 0.0,     # non disponible
                })

                print(f"[RANDOM] Round {rnd}/{self.max_rounds} terminé")
                if steps < self.steps_per_round:
                    break

            return self._save_results(round_logs)

        finally:
            env.close()

    def _save_results(self, round_logs):
        os.makedirs("results/comparison", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("results/comparison", f"random_{ts}.json")

        summary = {
            "method": "Random",
            "rounds": round_logs,
            "final": round_logs[-1] if round_logs else {},
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"[RANDOM] Résultats sauvegardés dans {out_path}")
        return summary