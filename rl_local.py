import json
import os
from datetime import datetime

import numpy as np
import torch

from env.multi_agent_env import MultiAgentParkingEnv
from rl.dqn import DQN
from rl.replay import ReplayBuffer


class RLLocalRunner:
    """
    Entraînement RL local avec DQN, adapté à l'API reset()/step(action_payload).
    L'index d'action est converti en (parking_index, price_level).
    """

    def __init__(
        self,
        sumo_cfg,
        agents_json,
        parkings_xml,
        episodes=1,
        steps_per_episode=40,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        memory_capacity=10000,
    ):
        self.sumo_cfg = sumo_cfg
        self.agents_json = agents_json
        self.parkings_xml = parkings_xml
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        with open(agents_json, "r", encoding="utf-8") as f:
            self.agents_data = json.load(f)

        self.zone_names = list(self.agents_data["agents"].keys())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # On crée un environnement temporaire pour récupérer state_dim et action_dim
        tmp_env = MultiAgentParkingEnv(
            sumo_cfg=self.sumo_cfg,
            agents_json=self.agents_json,
            parkings_xml=self.parkings_xml,
        )
        self.state_dim = int(getattr(tmp_env, "state_dim", 0))
        self.action_dim = int(getattr(tmp_env, "action_dim", 0))
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Impossible de déterminer les dimensions de l’état ou de l’action.")

        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(memory_capacity)

    def _select_action(self, state, epsilon):
        """Politique epsilon-greedy."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.model(s).argmax(dim=1).item())

    def _normalize_transition(self, transition):
        """Rend compatible une transition du ReplayBuffer."""
        if not isinstance(transition, (list, tuple)):
            raise ValueError(f"Transition invalide: {type(transition)}")
        if len(transition) < 5:
            raise ValueError(f"Transition trop courte: {len(transition)} éléments")
        return transition[0], transition[1], transition[2], transition[3], transition[4]

    def _update_model(self):
        """Met à jour le réseau DQN à partir d'un batch d'expériences."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        normalized = [self._normalize_transition(t) for t in batch]
        states, actions, rewards, next_states, dones = zip(*normalized)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.model(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = torch.nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def run(self, use_gui: bool = False):
        """
        Entraîne le modèle DQN en utilisant la nouvelle API reset()/step().
        À chaque épisode on réinitialise l'environnement via reset() puis on boucle
        sur steps_per_episode actions, sauf si l'environnement signale done.
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
        episode_logs = []
        try:
            env.use_gui = bool(use_gui)
            env.start()

            price_levels = ["cheap", "standard", "premium"]

            for ep in range(1, self.episodes + 1):
                state = env.reset()
                epsilon = max(0.1, 0.95 * (0.95 ** (ep - 1)))
                losses = []

                reward_sum = assigned_sum = dist_sum = occ_sum = congestion_sum = 0.0
                steps = 0

                for _ in range(self.steps_per_episode):
                    action_idx = self._select_action(state, epsilon)
                    parking_index = int(action_idx // 3)
                    price_idx = int(action_idx % 3)
                    price_level = price_levels[price_idx]

                    candidates = getattr(env, "current_candidates", [])
                    if candidates and 0 <= parking_index < len(candidates):
                        chosen_dist = float(candidates[parking_index][1])
                    else:
                        chosen_dist = 0.0
                        parking_index = 0

                    action_payload = {
                        "parking_index": parking_index,
                        "price_level": price_level,
                    }

                    next_state, reward, done, info = env.step(action_payload)
                    self.memory.push(state, action_idx, reward, next_state, float(done))
                    loss = self._update_model()
                    if loss is not None:
                        losses.append(loss)

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
                avg_loss = float(np.mean(losses)) if losses else 0.0
                avg_distance = dist_sum / steps if steps > 0 else 0.0
                avg_occupancy = occ_sum / steps if steps > 0 else 0.0
                avg_congestion = congestion_sum / steps if steps > 0 else 0.0

                episode_logs.append({
                    "episode": ep,
                    "epsilon": float(epsilon),
                    "avg_reward": float(avg_reward),
                    "avg_assigned": float(avg_assigned),
                    "avg_loss": float(avg_loss),
                    "avg_distance": float(avg_distance),
                    "avg_search_time": 0.0,   # non disponible
                    "avg_occupancy": float(avg_occupancy),
                    "avg_congestion": float(avg_congestion),
                    "avg_imbalance": 0.0,     # non disponible
                })

                print(f"[RL_LOCAL] Episode {ep}/{self.episodes} terminé")
                if steps < self.steps_per_episode:
                    break

            return self._save_results(episode_logs)

        finally:
            env.close()

    def _save_results(self, episode_logs):
        os.makedirs("results/comparison", exist_ok=True)
        os.makedirs("results/models", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = os.path.join("results/comparison", f"rl_local_{ts}.json")
        out_model = os.path.join("results/models", f"rl_local_model_{ts}.pt")

        torch.save(self.model.state_dict(), out_model)

        summary = {
            "method": "RL Local",
            "episodes": episode_logs,
            "final": episode_logs[-1] if episode_logs else {},
            "model_path": out_model,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        return summary