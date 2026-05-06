import copy
import os
import sys
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.multi_agent_env import MultiAgentParkingEnv
from rl.dqn import DQN
from rl.replay import ReplayBuffer


class ParkingFlowerClient(fl.client.NumPyClient):
    def __init__(self, env_config, client_id="client_0", agent_name="agent_1"):
        self.env_config = copy.deepcopy(env_config)
        self.client_id = str(client_id)
        self.agent_name = str(agent_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gui = bool(self.env_config.get("use_gui", False))

        self.top_k = int(self.env_config.get("top_k", 5))
        self.state_dim = 8 + (self.top_k * 8)
        self.action_dim = self.top_k

        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.memory = ReplayBuffer(20000)

        self.gamma = 0.95
        self.batch_size = 64
        self.target_update_freq = 100
        self.train_step_count = 0

    def _build_env(self):
        return MultiAgentParkingEnv(
            sumo_cfg=self.env_config["sumo_cfg"],
            parkings_xml=self.env_config["parkings_xml"],
            agents_json=self.env_config["agents_json"],
            agent_name=self.agent_name,
            top_k=self.env_config.get("top_k", 5),
            max_steps=self.env_config.get("max_steps", 1000),
            agent_detection_radius=self.env_config.get("agent_detection_radius", 1500.0),
            use_gui=self.use_gui,
            warmup_steps=self.env_config.get("warmup_steps", 100),
            gui_delay=self.env_config.get("gui_delay", 0.03),
            parking_demand_prob=self.env_config.get("parking_demand_prob", 0.8),
        )

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.target_model.load_state_dict(self.model.state_dict())

        self.train_step_count = 0

        if len(self.memory) > 8000:
            purge_main = (len(self.memory.main_buffer) * 2) // 5
            purge_important = len(self.memory.important_buffer) // 5

            for _ in range(purge_main):
                if self.memory.main_buffer:
                    self.memory.main_buffer.popleft()

            for _ in range(purge_important):
                if self.memory.important_buffer:
                    self.memory.important_buffer.popleft()

    def _select_action(self, state, env, epsilon):
        num_candidates = len(env.current_candidates)

        if num_candidates <= 0:
            return 0

        if np.random.random() < epsilon:
            return np.random.randint(0, num_candidates)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0)

            masked_q = q_values.clone()
            if num_candidates < self.action_dim:
                masked_q[num_candidates:] = -1e9

            return int(masked_q.argmax().item())

    def _warmup_until_candidates(self, env, max_warmup_steps=120):
        state = env.reset()

        for _ in range(max_warmup_steps):
            if len(env.current_candidates) > 0:
                return state, True

            next_state, _, done, _ = env.step(0)
            state = next_state

            if done:
                state = env.reset()

        return state, len(env.current_candidates) > 0

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        epsilon = float(config.get("epsilon", 0.1))
        self.gamma = float(config.get("gamma", 0.99))
        self.batch_size = int(config.get("batch_size", 64))
        steps_per_round = int(config.get("steps_per_round", 600))
        warmup_steps = int(config.get("warmup_steps", 300))
        min_useful_steps = int(config.get("min_useful_steps", 20))
        max_total_sim_steps = int(config.get("max_total_sim_steps", steps_per_round * 4))

        if self.agent_name == "agent_2":
            steps_per_round = int(steps_per_round * 1.2)

        env = self._build_env()

        total_reward = 0.0
        total_assigned = 0.0
        local_losses = []

        useful_steps = 0
        total_sim_steps = 0
        zero_candidate_steps = 0

        try:
            env.start()
            state, found_candidates = self._warmup_until_candidates(
                env, max_warmup_steps=warmup_steps
            )

            if not found_candidates:
                print(
                    f"[{self.client_id}/{self.agent_name}] fit | "
                    f"aucun candidat trouvé après warmup={warmup_steps}"
                )

            while useful_steps < steps_per_round and total_sim_steps < max_total_sim_steps:
                if env.step_count >= env.max_steps:
                    break

                num_candidates_before = len(env.current_candidates)

                action = self._select_action(state, env, epsilon)
                next_state, reward, done, info = env.step(action)

                total_sim_steps += 1

                if num_candidates_before > 0:
                    total_reward += float(reward)
                    total_assigned += float(info.get("assigned", 0))
                    useful_steps += 1

                    next_num_candidates = len(env.current_candidates)

                    self.memory.push(
                        state,
                        action,
                        reward,
                        next_state,
                        float(done),
                        float(next_num_candidates),
                    )

                    if len(self.memory) >= self.batch_size:
                        loss = self._update_model()
                        if loss is not None:
                            local_losses.append(loss)
                else:
                    zero_candidate_steps += 1

                state = next_state

                if done:
                    state = env.reset()

                if len(env.current_candidates) <= 0 and useful_steps < min_useful_steps:
                    state, _ = self._warmup_until_candidates(env, max_warmup_steps=30)

            metrics = {
                "reward": float(total_reward),
                "assigned": float(total_assigned),
                "loss": float(np.mean(local_losses)) if local_losses else 0.0,
                "buffer_size": float(len(self.memory)),
                "effective_steps": float(useful_steps),
                "total_sim_steps": float(total_sim_steps),
                "zero_candidate_steps": float(zero_candidate_steps),
            }

            print(
                f"[{self.client_id}/{self.agent_name}] fit | "
                f"reward={metrics['reward']:.3f} | "
                f"assigned={metrics['assigned']:.1f} | "
                f"loss={metrics['loss']:.4f} | "
                f"useful_steps={useful_steps} | "
                f"sim_steps={total_sim_steps} | "
                f"use_gui={self.use_gui}"
            )

            num_examples = max(1, int(useful_steps))
            return self.get_parameters(config={}), num_examples, metrics

        finally:
            env.close()

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        env = self._build_env()
        eval_steps = int(config.get("eval_steps", 80))
        warmup_steps = int(config.get("warmup_steps", 300))
        max_total_sim_steps = int(config.get("max_total_sim_steps", eval_steps * 4))

        total_reward = 0.0
        total_assigned = 0.0
        useful_eval_steps = 0
        total_sim_steps = 0
        zero_candidate_steps = 0

        try:
            env.start()
            state, found_candidates = self._warmup_until_candidates(
                env, max_warmup_steps=warmup_steps
            )

            if not found_candidates:
                print(
                    f"[{self.client_id}/{self.agent_name}] eval | "
                    f"aucun candidat trouvé après warmup={warmup_steps}"
                )

            while useful_eval_steps < eval_steps and total_sim_steps < max_total_sim_steps:
                num_candidates = len(env.current_candidates)

                if num_candidates <= 0:
                    next_state, _, done, _ = env.step(0)
                    zero_candidate_steps += 1
                    total_sim_steps += 1
                    state = next_state

                    if done:
                        state = env.reset()
                    continue

                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_values = self.model(state_t).squeeze(0)

                    masked_q = q_values.clone()
                    if num_candidates < self.action_dim:
                        masked_q[num_candidates:] = -1e9

                    action = int(masked_q.argmax().item())

                next_state, reward, done, info = env.step(action)

                total_reward += float(reward)
                total_assigned += float(info.get("assigned", 0))
                useful_eval_steps += 1
                total_sim_steps += 1
                state = next_state

                if done:
                    state = env.reset()

            loss_proxy = float(-total_reward)
            num_examples = max(1, useful_eval_steps)

            metrics = {
                "reward": float(total_reward),
                "assigned": float(total_assigned),
                "effective_eval_steps": float(useful_eval_steps),
                "total_sim_steps": float(total_sim_steps),
                "zero_candidate_steps": float(zero_candidate_steps),
            }

            print(
                f"[{self.client_id}/{self.agent_name}] eval | "
                f"reward={metrics['reward']:.3f} | "
                f"assigned={metrics['assigned']:.1f} | "
                f"useful_steps={useful_eval_steps} | "
                f"sim_steps={total_sim_steps} | "
                f"use_gui={self.use_gui}"
            )

            return loss_proxy, num_examples, metrics

        finally:
            env.close()

    def _update_model(self):
        try:
            batch = self.memory.sample(self.batch_size)
        except Exception:
            return None

        clean_batch = []
        for item in batch:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                clean_batch.append(item[:6])

        if len(clean_batch) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_num_candidates = zip(*clean_batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        next_num_candidates = torch.tensor(
            np.array(next_num_candidates), dtype=torch.long, device=self.device
        )

        curr_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            online_next_q = self.model(next_states)
            target_next_q = self.target_model(next_states)

            masked_online_next_q = online_next_q.clone()
            masked_target_next_q = target_next_q.clone()

            for i in range(self.batch_size):
                n = int(next_num_candidates[i].item())
                if n <= 0:
                    masked_online_next_q[i, :] = -1e9
                    masked_target_next_q[i, :] = -1e9
                elif n < self.action_dim:
                    masked_online_next_q[i, n:] = -1e9
                    masked_target_next_q[i, n:] = -1e9

            next_actions = masked_online_next_q.argmax(dim=1, keepdim=True)
            next_q = masked_target_next_q.gather(1, next_actions).squeeze(1)
            next_q = torch.where(
                next_num_candidates > 0,
                next_q,
                torch.zeros_like(next_q),
            )

            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())


def build_env_config():
    use_gui_env = os.getenv("SMARTPARKING_USE_GUI", "0").strip()
    use_gui = use_gui_env == "1"

    return {
        "sumo_cfg": os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "dua.static.sumocfg"),
        "parkings_xml": os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "parkings_min300.add.xml"),
        "agents_json": os.path.join(PROJECT_ROOT, "config", "agents_weighted_kmeans_balanced.json"),
        "top_k": 5,
        "max_steps": 2500,
        "agent_detection_radius": 1500.0,
        "use_gui": use_gui,
        "warmup_steps": 300,
        "gui_delay": 0.02,
        "parking_demand_prob": 0.35,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m training.run_fl_client <agent_name>")

    agent_name = sys.argv[1]
    env_config = build_env_config()

    client = ParkingFlowerClient(
        env_config=env_config,
        client_id= agent_name,
        agent_name=agent_name,
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )