import time
import xml.etree.ElementTree as ET

import numpy as np


class ParkingManager:
    MIN_ASSIGN_ROUTE_LENGTH_M = 90.0

    def __init__(self, parkings_xml_path):
        self.parkings_xml_path = parkings_xml_path
        self.conn = None
        self.parking_meta = {}
        self.cap = {}
        self.real_occupancy = {}
        self.incoming_count = {}
        self.vehicle_target_parking = {}
        self.assignment_time = {}
        self._load_parkings_from_xml()

    def set_connection(self, conn):
        self.conn = conn

    def _load_parkings_from_xml(self):
        tree = ET.parse(self.parkings_xml_path)
        root = tree.getroot()

        for elem in root.findall("parkingArea"):
            pid = elem.get("id")
            lane = elem.get("lane")
            if pid is None or lane is None:
                continue

            edge = lane.rsplit("_", 1)[0] if "_" in lane else lane
            cap = int(float(elem.get("roadsideCapacity", 1)))

            self.parking_meta[pid] = {
                "lane": lane,
                "edge": edge,
            }
            self.cap[pid] = max(cap, 1)

    def initialize(self):
        self.real_occupancy = {pid: 0 for pid in self.parking_meta}
        self.incoming_count = {pid: 0 for pid in self.parking_meta}
        self.vehicle_target_parking = {}
        self.assignment_time = {}

    def refresh(self):
        if self.conn is None:
            return

        for pid in self.parking_meta:
            try:
                occ = int(self.conn.parkingarea.getVehicleCount(pid))
            except Exception:
                occ = self.real_occupancy.get(pid, 0)
            self.real_occupancy[pid] = max(0, occ)

        self._rebuild_incoming_from_active_vehicles()

    def _rebuild_incoming_from_active_vehicles(self):
        try:
            active_vehicles = set(self.conn.vehicle.getIDList())
        except Exception:
            active_vehicles = set()

        current_time = time.time()
        new_targets = {}
        new_incoming = {pid: 0 for pid in self.parking_meta}

        for vid, (pid, assign_time, travel_time) in list(self.assignment_time.items()):
            if vid in active_vehicles:
                if current_time - assign_time < travel_time * 0.8:
                    new_targets[vid] = pid
                    new_incoming[pid] += 1

        self.vehicle_target_parking = new_targets
        self.incoming_count = new_incoming
        self.assignment_time = {
            k: v for k, v in self.assignment_time.items() if k in new_targets
        }

    def is_vehicle_valid(self, vehicle_id):
        if self.conn is None:
            return False

        if vehicle_id in self.vehicle_target_parking:
            return False

        try:
            vehicle_ids = set(self.conn.vehicle.getIDList())
            if vehicle_id not in vehicle_ids:
                return False

            current_edge = self.conn.vehicle.getRoadID(vehicle_id)
            if not current_edge or current_edge.startswith(":"):
                return False

            route = self.conn.vehicle.getRoute(vehicle_id)
            if route is None or len(route) == 0:
                return False

            return True
        except Exception:
            return False

    def get_capacity(self, pid):
        return int(self.cap.get(pid, 1))

    def get_real_occupancy(self, pid):
        return int(self.real_occupancy.get(pid, 0))

    def get_incoming_count(self, pid):
        return int(self.incoming_count.get(pid, 0))

    def get_predicted_occupancy(self, pid):
        real = self.real_occupancy.get(pid, 0)
        incoming = self.incoming_count.get(pid, 0)
        cap = max(self.get_capacity(pid), 1)
        occ_ratio = real / cap

        if occ_ratio < 0.50:
            incoming_weight = 0.45
        elif occ_ratio < 0.80:
            incoming_weight = 0.35
        else:
            incoming_weight = 0.20

        predicted = real + incoming_weight * incoming
        predicted = min(predicted, cap)
        return int(predicted)

    def get_free_slots(self, pid):
        cap = self.get_capacity(pid)
        predicted = self.get_predicted_occupancy(pid)
        return max(cap - predicted, 0)

    def get_real_occupancy_ratio(self, pid):
        return self.get_real_occupancy(pid) / max(self.get_capacity(pid), 1)

    def get_predicted_occupancy_ratio(self, pid):
        return self.get_predicted_occupancy(pid) / max(self.get_capacity(pid), 1)

    def compute_dynamic_price(self, pid, distance_m=0.0, mode="balanced", traffic_pressure=1.0):
        base_price = 2.0
        cap = max(self.get_capacity(pid), 1)

        real_occ = self.get_real_occupancy(pid)
        predicted = self.get_predicted_occupancy(pid)

        occ_ratio = real_occ / cap
        pred_ratio = predicted / cap

        price = base_price
        price += 0.80 * occ_ratio
        price += 0.60 * pred_ratio

        if mode == "cheap":
            price -= 0.30
        elif mode == "close":
            price += 0.25

        traffic_pressure = min(max(float(traffic_pressure), 0.9), 1.2)
        price *= traffic_pressure

        return float(max(1.5, min(price, 6.5)))

    def _get_parking_position(self, conn, pid):
        try:
            lane = self.parking_meta[pid]["lane"]
            shape = conn.lane.getShape(lane)
            if not shape:
                return None
            return shape[len(shape) // 2]
        except Exception:
            return None

    def _get_route_length(self, route):
        if route is None:
            return None

        try:
            length = float(getattr(route, "length", 0.0))
            if length > 0:
                return length
        except Exception:
            pass

        total = 0.0
        try:
            for edge_id in getattr(route, "edges", []):
                total += float(self.conn.edge.getLength(edge_id))
        except Exception:
            return None

        return total if total > 0 else None

    def _find_route_to_parking(self, current_edge, pid):
        try:
            if not current_edge or current_edge.startswith(":"):
                return None, None

            target_edge = self.parking_meta[pid]["edge"]
            route = self.conn.simulation.findRoute(current_edge, target_edge)
            if route is None:
                return None, None

            edges = list(getattr(route, "edges", []) or [])
            if len(edges) == 0 or target_edge not in edges:
                return None, None

            route_length = self._get_route_length(route)
            return route, route_length
        except Exception:
            return None, None

    def _is_reachable(self, current_edge, pid):
        route, _ = self._find_route_to_parking(current_edge, pid)
        return route is not None

    def get_candidate_parkings(
        self,
        conn,
        vehicle_id,
        radius,
        include_reachability=False,
        allow_global_fallback=False,
        max_global_candidates=15,
        saturation_threshold=0.985,
        max_distance=3000.0,
    ):
        candidates = []
        global_candidates = []

        try:
            veh_pos = conn.vehicle.getPosition(vehicle_id)
            current_edge = conn.vehicle.getRoadID(vehicle_id)
        except Exception:
            return []

        for pid in self.parking_meta:
            try:
                if self.get_free_slots(pid) <= 0:
                    continue

                pred_ratio = self.get_predicted_occupancy_ratio(pid)
                if pred_ratio >= saturation_threshold:
                    continue

                park_pos = self._get_parking_position(conn, pid)
                if park_pos is None:
                    continue

                air_dist = float(np.linalg.norm(np.array(veh_pos) - np.array(park_pos)))
                if air_dist > max_distance:
                    continue

                route_length = None
                if include_reachability:
                    route, route_length = self._find_route_to_parking(current_edge, pid)
                    if route is None:
                        continue
                    if route_length is not None and route_length < self.MIN_ASSIGN_ROUTE_LENGTH_M:
                        continue

                effective_dist = air_dist
                if route_length is not None:
                    effective_dist = max(air_dist, route_length)

                item = (pid, effective_dist)
                global_candidates.append(item)

                if effective_dist <= radius:
                    candidates.append(item)
            except Exception:
                continue

        if len(candidates) == 0 and allow_global_fallback:
            global_candidates.sort(key=lambda x: x[1])
            return global_candidates[:max_global_candidates]

        candidates.sort(key=lambda x: x[1])
        return candidates

    def assign(self, vehicle_id, pid):
        if self.conn is None:
            return False

        if pid not in self.parking_meta:
            return False

        if self.get_free_slots(pid) <= 0:
            return False

        if vehicle_id in self.vehicle_target_parking:
            return False

        try:
            if not self.is_vehicle_valid(vehicle_id):
                return False

            current_edge = self.conn.vehicle.getRoadID(vehicle_id)
            route, route_length = self._find_route_to_parking(current_edge, pid)
            if route is None:
                return False

            if route_length is not None and route_length < self.MIN_ASSIGN_ROUTE_LENGTH_M:
                return False

            route_edges = list(getattr(route, "edges", []) or [])
            if len(route_edges) == 0:
                return False

            self.conn.vehicle.setRoute(vehicle_id, route_edges)
            self.conn.vehicle.setParkingAreaStop(vehicle_id, pid, duration=600)

            try:
                veh_pos = self.conn.vehicle.getPosition(vehicle_id)
                park_pos = self._get_parking_position(self.conn, pid)
                dist = np.linalg.norm(np.array(veh_pos) - np.array(park_pos))
                travel_time = dist / 10.0
            except Exception:
                travel_time = 9999.0

            self.assignment_time[vehicle_id] = (pid, time.time(), travel_time)
            self.vehicle_target_parking[vehicle_id] = pid
            self.incoming_count[pid] = self.incoming_count.get(pid, 0) + 1
            return True
        except Exception:
            return False