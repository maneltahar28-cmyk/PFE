import os
import csv
import random
import time
import numpy as np
import traci

from env.parking_manager import ParkingManager
from env.agent_mapper import AgentMapper
from v2x.v2x_comm import V2XCommunication, V2XFeatureExtractor


class MultiAgentParkingEnv:
    def __init__(
        self,
        sumo_cfg,
        parkings_xml,
        agents_json,
        agent_name,
        top_k=5,
        max_steps=1000,
        agent_detection_radius=1500.0,
        use_gui=False,
        warmup_steps=100,
        gui_delay=0.03,
        parking_demand_prob=0.8,
    ):
        self.sumo_cfg = sumo_cfg
        self.parkings_xml = parkings_xml
        self.agents_json = agents_json
        self.agent_name = agent_name
        self.use_gui = bool(use_gui)

        self.warmup_steps = int(warmup_steps)
        self.gui_delay = float(gui_delay)
        self.top_k = int(top_k)
        self.max_steps = int(max_steps)
        self.parking_demand_prob = float(parking_demand_prob)

        agent_radius_map = {
            "agent_1": 1200.0,
            "agent_2": 1500.0,
            "agent_3": 2200.0,
            "agent_4": 1750.0,
        }
        self.agent_detection_radius = float(
            agent_radius_map.get(agent_name, agent_detection_radius)
        )

        self.conn = None
        self.step_count = 0

        self.parking_manager = ParkingManager(parkings_xml)
        self.agent_mapper = AgentMapper(agents_json)
        self.v2x = V2XCommunication()
        self.v2x_features = V2XFeatureExtractor(max_vehicles=10, max_edges=20)

        self.current_vehicle_id = None
        self.current_mode = None
        self.current_radius = None
        self.current_candidates = []
        self.current_agent_distance = None
        self.current_source_agent = None

        self.agent_assignment_count = 0
        self.global_assignment_count = 0

        self.local_assigned_vehicles = set()
        self.local_failed_vehicles = set()
        self.local_pending_vehicles = set()

        self.parking_usage_count = {
            pid: 0 for pid in self.parking_manager.parking_meta.keys()
        }

        self._cached_parking_positions = {}

        if self.agent_name == "agent_2":
            self.vehicle_search_limit = 190
        elif self.agent_name == "agent_3":
            self.vehicle_search_limit = 220
        else:
            self.vehicle_search_limit = 150

        self.state_dim = 8 + (self.top_k * 8)
        self.action_dim = self.top_k

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(project_root, "outputs")
        self.decisions_csv_path = os.path.join(self.output_dir, "decisions_log.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        self._ensure_decisions_csv_header()

        self.agent_usage = {name: 0 for name in self.agent_mapper.get_agent_names()}

    def get_traffic_pressure(self):
        if self.conn is not None:
            try:
                pressure = self.v2x_features.get_network_pressure(self.conn)
                return float(pressure)
            except Exception:
                pass

        cycle = self.step_count % 1200
        if 200 <= cycle <= 450:
            return 1.35
        if 700 <= cycle <= 950:
            return 1.45
        return 1.0

    def get_dynamic_demand_probability(self):
        base = self.parking_demand_prob
        pressure = self.get_traffic_pressure()
        if pressure >= 1.40:
            return min(0.80, base + 0.25)
        if pressure >= 1.30:
            return min(0.70, base + 0.18)
        return base

    def _publish_zone_state(self):
        if len(self.parking_manager.parking_meta) == 0:
            return

        occ = []
        free = []
        inc = []

        for pid in self.parking_manager.parking_meta:
            cap = max(self.parking_manager.get_capacity(pid), 1)
            occ.append(self.parking_manager.get_real_occupancy(pid) / cap)
            free.append(self.parking_manager.get_free_slots(pid) / cap)
            inc.append(self.parking_manager.get_incoming_count(pid) / cap)

        self.v2x.publish_zone_state(
            zone_name=self.agent_name,
            free_ratio=float(np.mean(free)),
            mean_occupancy=float(np.mean(occ)),
            incoming_ratio=float(np.mean(inc)),
        )

    def _get_corrected_predicted_occupancy(self, pid):
        real = int(self.parking_manager.get_real_occupancy(pid))
        incoming = int(self.parking_manager.get_incoming_count(pid))
        capacity = max(int(self.parking_manager.get_capacity(pid)), 1)

        incoming_capped = min(incoming, int(0.35 * capacity))
        predicted = real + incoming_capped

        return int(min(predicted, capacity))

    def _get_corrected_predicted_occupancy_ratio(self, pid):
        cap = max(int(self.parking_manager.get_capacity(pid)), 1)
        return float(self._get_corrected_predicted_occupancy(pid)) / float(cap)

    def warmup_until_traffic(self, min_vehicles=10, max_warmup_steps=None):
        if max_warmup_steps is None:
            max_warmup_steps = self.warmup_steps

        for i in range(max_warmup_steps):
            self.conn.simulationStep()
            self.parking_manager.refresh()
            self._publish_zone_state()

            try:
                nveh = len(self.conn.vehicle.getIDList())
            except Exception:
                nveh = 0

            if self.use_gui and self.gui_delay > 0:
                time.sleep(self.gui_delay)

            if nveh >= min_vehicles:
                print(f"[WARMUP] {self.agent_name} | steps={i + 1} | vehicles={nveh}")
                return

        print(f"[WARMUP] {self.agent_name} | max steps atteints={max_warmup_steps}")

    def start(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c",
            self.sumo_cfg,
            "--step-length",
            "1",
            "--quit-on-end",
            "true",
            "--duration-log.disable",
            "true",
            "--no-step-log",
            "true",
        ]
        if self.use_gui:
            sumo_cmd += ["--start", "--delay", str(int(self.gui_delay * 1000))]

        traci.start(sumo_cmd)
        self.conn = traci
        self.parking_manager.set_connection(self.conn)
        self.parking_manager.initialize()
        self.v2x.reset()

        self.step_count = 0
        self.current_vehicle_id = None
        self.current_mode = None
        self.current_radius = None
        self.current_candidates = []
        self.current_agent_distance = None
        self.current_source_agent = None

        self.agent_assignment_count = 0
        self.global_assignment_count = 0

        self.local_assigned_vehicles.clear()
        self.local_failed_vehicles.clear()
        self.local_pending_vehicles.clear()

        self.agent_usage = {name: 0 for name in self.agent_mapper.get_agent_names()}
        self.parking_usage_count = {
            pid: 0 for pid in self.parking_manager.parking_meta
        }
        self._cached_parking_positions = {}

        self.warmup_until_traffic(min_vehicles=10, max_warmup_steps=self.warmup_steps)

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close(wait=False)
            except Exception:
                try:
                    self.conn.close()
                except Exception:
                    pass
            self.conn = None

    def _ensure_decisions_csv_header(self):
        if not os.path.exists(self.decisions_csv_path):
            try:
                with open(self.decisions_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step",
                        "agent_name",
                        "source_agent",
                        "vehicle_id",
                        "mode",
                        "parking_id",
                        "distance_m",
                        "price",
                        "free_slots",
                        "capacity",
                        "real_occupancy",
                        "incoming_count",
                        "predicted_occupancy",
                        "reward",
                        "reason",
                    ])
            except PermissionError:
                pass

    def _log_decision_csv(
        self,
        vehicle_id,
        mode,
        parking_id,
        distance_m,
        price,
        free_slots,
        capacity,
        real_occupancy,
        incoming_count,
        predicted_occupancy,
        reward,
        reason,
        source_agent=None,
    ):
        try:
            with open(self.decisions_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.step_count,
                    self.agent_name,
                    source_agent if source_agent is not None else "",
                    vehicle_id,
                    mode,
                    parking_id,
                    f"{float(distance_m):.2f}" if distance_m is not None else "",
                    f"{float(price):.2f}" if price is not None else "",
                    int(free_slots) if free_slots is not None else "",
                    int(capacity) if capacity is not None else "",
                    int(real_occupancy) if real_occupancy is not None else "",
                    int(incoming_count) if incoming_count is not None else "",
                    int(predicted_occupancy) if predicted_occupancy is not None else "",
                    f"{float(reward):.2f}" if reward is not None else "",
                    reason,
                ])
        except PermissionError:
            return

    def get_client_mode(self):
        r = random.random()
        if r < 0.35:
            return "close"
        if r < 0.65:
            return "cheap"
        return "balanced"

    def get_radius(self, mode):
        pressure = self.get_traffic_pressure()

        if mode == "close":
            return 500.0 if pressure < 1.3 else 700.0

        if mode == "cheap":
            return 1800.0 if pressure < 1.3 else 2500.0

        return 1200.0 if pressure < 1.3 else 1700.0

    def compute_candidate_score(self, pid, dist, mode):
        traffic_pressure = self.get_traffic_pressure()

        price = self.parking_manager.compute_dynamic_price(
            pid,
            distance_m=dist,
            mode=mode,
            traffic_pressure=traffic_pressure,
        )

        cap = max(self.parking_manager.get_capacity(pid), 1)
        free = self.parking_manager.get_free_slots(pid)
        incoming = self.parking_manager.get_incoming_count(pid)

        dist_norm = min(float(dist) / 3000.0, 1.0)
        price_norm = min(float(price) / 6.5, 1.0)
        free_ratio = float(free) / float(cap)
        pred_occ_ratio = self._get_corrected_predicted_occupancy_ratio(pid)
        incoming_ratio = min(float(incoming) / float(cap), 1.0)

        usage = self.parking_usage_count.get(pid, 0)
        total_usage = max(sum(self.parking_usage_count.values()), 1)
        usage_ratio = float(usage) / float(total_usage)

        agent_total = max(self.agent_usage.get(self.agent_name, 1), 1)
        local_usage_ratio = float(usage) / float(agent_total)

        dominance_penalty = min(1.20 * np.sqrt(usage_ratio), 0.75)
        local_penalty = min(0.75 * np.sqrt(local_usage_ratio), 0.55)

        saturation_penalty = 0.0
        if pred_occ_ratio > 0.60:
            saturation_penalty += 0.25
        if pred_occ_ratio > 0.80:
            saturation_penalty += 0.70
        if pred_occ_ratio > 0.92:
            saturation_penalty += 1.60

        if mode == "close":
            score = (
                1.65 * dist_norm
                + 0.10 * price_norm
                + 0.18 * pred_occ_ratio
                + 0.05 * incoming_ratio
                - 0.08 * free_ratio
            )
        elif mode == "cheap":
            score = (
                0.45 * dist_norm
                + 1.30 * price_norm
                + 0.18 * pred_occ_ratio
                + 0.05 * incoming_ratio
                - 0.08 * free_ratio
            )
        else:
            score = (
                0.75 * dist_norm
                + 0.75 * price_norm
                + 0.22 * pred_occ_ratio
                + 0.05 * incoming_ratio
                - 0.10 * free_ratio
            )

        score += dominance_penalty
        score += local_penalty
        score += saturation_penalty

        return float(score)

    def _get_parking_positions(self):
        positions = {}
        for pid, meta in self.parking_manager.parking_meta.items():
            try:
                lane = meta["lane"]
                shape = self.conn.lane.getShape(lane)
                if shape:
                    positions[pid] = shape[len(shape) // 2]
            except Exception:
                continue
        return positions

    def _vehicle_is_valid_for_assignment(self, vid):
        if vid in self.local_assigned_vehicles:
            return False
        if vid in self.local_pending_vehicles:
            return False
        if vid in self.local_failed_vehicles:
            return False
        if hasattr(self.parking_manager, "is_vehicle_valid"):
            return self.parking_manager.is_vehicle_valid(vid)
        return False

    def _vehicle_is_eligible_for_agent(self, vid):
        try:
            if not self._vehicle_is_valid_for_assignment(vid):
                return False, None, None

            pos = self.conn.vehicle.getPosition(vid)
            road_id = self.conn.vehicle.getRoadID(vid)
            self.v2x.send_vehicle_status(vid, road_id, current_zone=self.agent_name)

            candidate_agents = self.agent_mapper.score_agents_with_seed_bonus(
                vehicle_pos=pos,
                parking_positions=self._cached_parking_positions,
                k=4,
            )
            candidate_names = [a for a, _ in candidate_agents]

            if self.agent_name in candidate_names:
                idx = candidate_names.index(self.agent_name)
                source_agent, dist_score = candidate_agents[idx]

                if self.agent_name == "agent_3":
                    margin = 1.60
                elif self.agent_name == "agent_4":
                    margin = 1.45
                else:
                    margin = 1.35

                if dist_score <= self.agent_detection_radius * margin:
                    return True, pos, source_agent

            return False, pos, None
        except Exception:
            return False, None, None

    def _get_valid_vehicles(self, limit=20):
        try:
            vehicles = list(self.conn.vehicle.getIDList())
        except Exception:
            return []

        random.shuffle(vehicles)
        valid = []
        demand_prob = self.get_dynamic_demand_probability()

        for vid in vehicles:
            if not self._vehicle_is_valid_for_assignment(vid):
                continue
            if random.random() > demand_prob:
                continue

            ok, pos, source_agent = self._vehicle_is_eligible_for_agent(vid)
            if not ok:
                continue

            nearest_agent, dist_to_agent = self.agent_mapper.get_nearest_agent(pos)
            valid.append((vid, dist_to_agent, source_agent, nearest_agent))

            if len(valid) >= limit:
                break

        return valid

    def _build_candidates_for_vehicle(self, vid):
        mode = self.get_client_mode()
        radius = self.get_radius(mode)
        max_distance = 3000.0

        try:
            raw_candidates = self.parking_manager.get_candidate_parkings(
                self.conn,
                vid,
                radius,
                include_reachability=True,
                allow_global_fallback=True,
                max_global_candidates=28,
                saturation_threshold=0.995,
                max_distance=max_distance,
            )
        except Exception:
            return mode, radius, []

        scored = []

        for pid, dist in raw_candidates:
            try:
                if dist > max_distance:
                    continue

                if self.parking_manager.get_free_slots(pid) <= 0:
                    continue

                pred_occ_ratio = self._get_corrected_predicted_occupancy_ratio(pid)
                if pred_occ_ratio >= 0.995:
                    continue

                score = self.compute_candidate_score(pid, dist, mode)
                scored.append((pid, dist, score))
            except Exception:
                continue

        if len(scored) == 0:
            return mode, radius, []

        scored.sort(key=lambda x: x[2])
        return mode, radius, scored[: self.top_k]

    def _build_state(self, vid, mode, radius, agent_distance, candidates, source_agent):
        state = []

        mode_vec = {
            "close": [1.0, 0.0, 0.0],
            "cheap": [0.0, 1.0, 0.0],
            "balanced": [0.0, 0.0, 1.0],
        }
        state += mode_vec.get(mode, [0.0, 0.0, 1.0])

        same_agent_flag = 1.0 if source_agent == self.agent_name else 0.0
        traffic_pressure = self.get_traffic_pressure()
        pressure_norm = min(max((traffic_pressure - 1.0) / 0.6, 0.0), 1.0)
        cycle_phase = (self.step_count % 1200) / 1200.0

        state += [
            radius / 3000.0,
            min(float(agent_distance) / 3000.0, 1.0),
            same_agent_flag,
            pressure_norm,
            cycle_phase,
        ]

        total_usage = max(sum(self.parking_usage_count.values()), 1)

        seed_parkings = set()
        if hasattr(self.agent_mapper, "get_agent_seed_parkings"):
            seed_parkings = set(self.agent_mapper.get_agent_seed_parkings(self.agent_name))

        for pid, dist, _score in candidates:
            price = self.parking_manager.compute_dynamic_price(
                pid,
                distance_m=dist,
                mode=mode,
                traffic_pressure=traffic_pressure,
            )

            free = self.parking_manager.get_free_slots(pid)
            cap = max(self.parking_manager.get_capacity(pid), 1)
            real_occ = self.parking_manager.get_real_occupancy(pid)
            incoming = self.parking_manager.get_incoming_count(pid)
            pred_occ_ratio = self._get_corrected_predicted_occupancy_ratio(pid)

            dist_norm = min(float(dist) / 3000.0, 1.0)
            price_norm = min(float(price) / 6.5, 1.0)
            free_ratio = float(free) / float(cap)
            real_occ_ratio = float(real_occ) / float(cap)
            incoming_ratio = min(float(incoming) / float(cap), 1.0)

            usage = self.parking_usage_count.get(pid, 0)
            usage_ratio = float(usage) / float(total_usage)
            seed_flag = 1.0 if pid in seed_parkings else 0.0

            state += [
                dist_norm,
                price_norm,
                free_ratio,
                pred_occ_ratio,
                real_occ_ratio,
                incoming_ratio,
                usage_ratio,
                seed_flag,
            ]

        while len(state) < self.state_dim:
            state.append(0.0)

        return np.array(state[: self.state_dim], dtype=np.float32)

    def reset(self):
        self.parking_manager.refresh()
        self._publish_zone_state()
        self._cached_parking_positions = self._get_parking_positions()

        valid_vehicles = self._get_valid_vehicles(limit=self.vehicle_search_limit)

        if self.agent_name == "agent_1" and len(valid_vehicles) > 1:
            valid_vehicles = sorted(valid_vehicles, key=lambda x: x[1])
            head = valid_vehicles[: min(25, len(valid_vehicles))]
            random.shuffle(head)
            valid_vehicles = head + valid_vehicles[min(25, len(valid_vehicles)):]

        if self.agent_name == "agent_3" and len(valid_vehicles) > 1:
            random.shuffle(valid_vehicles)

        for vid, dist_to_agent, source_agent, nearest_agent in valid_vehicles:
            if not self._vehicle_is_valid_for_assignment(vid):
                continue

            result = self._build_candidates_for_vehicle(vid)
            if result is None:
                continue

            mode, radius, candidates = result
            if len(candidates) == 0:
                continue

            self.current_vehicle_id = vid
            self.current_mode = mode
            self.current_radius = radius
            self.current_candidates = candidates
            self.current_agent_distance = dist_to_agent
            self.current_source_agent = source_agent if source_agent is not None else nearest_agent
            self.local_pending_vehicles.add(vid)

            return self._build_state(
                vid,
                mode,
                radius,
                dist_to_agent,
                candidates,
                self.current_source_agent,
            )

        self.current_vehicle_id = None
        self.current_mode = None
        self.current_radius = None
        self.current_candidates = []
        self.current_agent_distance = None
        self.current_source_agent = None
        return np.zeros(self.state_dim, dtype=np.float32)

    def _compute_reward(
        self,
        mode,
        dist,
        price,
        free_slots,
        capacity,
        real_occupancy=0,
        predicted_occupancy=0,
        agent_assignments=0,
        total_assignments=1,
        parking_usage_ratio=0.0,
    ):
        cap = max(int(capacity), 1)
        dist_norm = min(float(dist) / 3000.0, 1.0)
        pred_occ_ratio = float(predicted_occupancy) / float(cap)
        free_ratio = float(free_slots) / float(cap)

        reward = 0.0

        if mode == "close":
            reward += 7.5 * (1.0 - dist_norm)
        elif mode == "cheap":
            reward += 4.5 * (1.0 - dist_norm)
        else:
            reward += 6.5 * (1.0 - dist_norm)

        if free_slots > 0:
            reward += 1.0
        else:
            reward -= 5.0

        if mode == "cheap":
            reward += max(0.0, 3.4 - float(price)) * 1.1
            reward -= 0.10 * float(price)
        elif mode == "close":
            reward -= 0.06 * float(price)
        else:
            reward += max(0.0, 3.1 - float(price)) * 0.55
            reward -= 0.08 * float(price)

        reward -= 1.5 * max(0.0, pred_occ_ratio - 0.60)
        reward -= 3.0 * max(0.0, pred_occ_ratio - 0.80)
        reward -= 5.0 * max(0.0, pred_occ_ratio - 0.92)

        reward -= 3.0 * float(parking_usage_ratio)

        if free_ratio >= 0.25 and pred_occ_ratio <= 0.70:
            reward += 0.7

        if mode == "balanced" and dist <= 1200.0 and price <= 3.2:
            reward += 0.8

        return float(np.clip(reward, -8.0, 8.0))

    def step(self, action):
        self.parking_manager.refresh()

        if self.current_vehicle_id is None or len(self.current_candidates) == 0:
            self.conn.simulationStep()
            self.step_count += 1
            done = self.step_count >= self.max_steps
            next_state = self.reset()
            return next_state, 0.0, done, {
                "assigned": 0,
                "agent_name": self.agent_name,
                "num_candidates": 0,
                "reason": "no_active_request",
            }

        vid = self.current_vehicle_id
        mode = self.current_mode
        candidates = self.current_candidates
        source_agent = self.current_source_agent

        if action < 0 or action >= self.top_k or action >= len(candidates):
            reward = -3.0
            assigned = 0
            reason = "invalid_action"
            self.local_pending_vehicles.discard(vid)
            self.local_failed_vehicles.add(vid)

            self._log_decision_csv(
                vid, mode, "", None, None, None, None, None, None, None,
                reward, reason, source_agent
            )
        else:
            chosen_pid, chosen_dist, _ = candidates[action]
            free_before = self.parking_manager.get_free_slots(chosen_pid)
            pred_ratio_before = self._get_corrected_predicted_occupancy_ratio(chosen_pid)

            if not self.parking_manager.is_vehicle_valid(vid):
                ok = False
                forced_reason = "vehicle_invalid_before_assign"
            elif free_before <= 0:
                ok = False
                forced_reason = "parking_full_before_assign"
            elif pred_ratio_before >= 0.995:
                ok = False
                forced_reason = "parking_saturated_before_assign"
            else:
                ok = self.parking_manager.assign(vid, chosen_pid)
                forced_reason = "assign_failed"

            if ok:
                self.agent_assignment_count += 1
                self.global_assignment_count += 1
                self.agent_usage[self.agent_name] += 1
                self.parking_usage_count[chosen_pid] = (
                    self.parking_usage_count.get(chosen_pid, 0) + 1
                )

                self.local_pending_vehicles.discard(vid)
                self.local_assigned_vehicles.add(vid)

                traffic_pressure = self.get_traffic_pressure()
                price = self.parking_manager.compute_dynamic_price(
                    chosen_pid,
                    distance_m=chosen_dist,
                    mode=mode,
                    traffic_pressure=traffic_pressure,
                )

                free_slots = self.parking_manager.get_free_slots(chosen_pid)
                capacity = self.parking_manager.get_capacity(chosen_pid)
                real_occupancy = self.parking_manager.get_real_occupancy(chosen_pid)
                incoming_count = self.parking_manager.get_incoming_count(chosen_pid)
                predicted_occupancy = self._get_corrected_predicted_occupancy(chosen_pid)

                total_usage = max(sum(self.parking_usage_count.values()), 1)
                parking_usage_ratio = float(
                    self.parking_usage_count.get(chosen_pid, 0)
                ) / float(total_usage)

                reward = self._compute_reward(
                    mode=mode,
                    dist=chosen_dist,
                    price=price,
                    free_slots=free_slots,
                    capacity=capacity,
                    real_occupancy=real_occupancy,
                    predicted_occupancy=predicted_occupancy,
                    agent_assignments=self.agent_assignment_count,
                    total_assignments=self.global_assignment_count,
                    parking_usage_ratio=parking_usage_ratio,
                )

                assigned = 1
                reason = "assigned"

                self.v2x.send_parking_recommendation(
                    vid, self.agent_name, chosen_pid, score=reward
                )

                self._log_decision_csv(
                    vid,
                    mode,
                    chosen_pid,
                    chosen_dist,
                    price,
                    free_slots,
                    capacity,
                    real_occupancy,
                    incoming_count,
                    predicted_occupancy,
                    reward,
                    reason,
                    source_agent,
                )
            else:
                reward = -2.0
                assigned = 0
                reason = forced_reason
                self.local_pending_vehicles.discard(vid)
                self.local_failed_vehicles.add(vid)

                self._log_decision_csv(
                    vid, mode, chosen_pid, chosen_dist, None, None, None,
                    None, None, None, reward, reason, source_agent
                )

        self.conn.simulationStep()
        self.step_count += 1

        if self.step_count % 100 == 0:
            print("[AGENT USAGE]", self.agent_usage)

        done = self.step_count >= self.max_steps
        next_state = self.reset()

        info = {
            "assigned": assigned,
            "agent_name": self.agent_name,
            "vehicle_id": vid,
            "mode": mode,
            "num_candidates": len(candidates),
            "reason": reason,
            "source_agent": source_agent,
        }
        return next_state, float(reward), done, info