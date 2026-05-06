"""
v2x_comm.py
-----------
Communication V2X / Edge légère pour smart parking.

Version pipeline corrigée :
- aucune dépendance à traci global ;
- messages véhicule, recommandation, état zone ;
- extracteur de features robuste ;
- helpers pour pression trafic et état agrégé.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class VehicleStatusMessage:
    vehicle_id: str
    current_edge: str
    current_zone: str | None
    timestamp: str


@dataclass
class ParkingRecommendationMessage:
    vehicle_id: str
    zone_name: str
    parking_id: str
    score: float
    timestamp: str


@dataclass
class ZoneStateMessage:
    zone_name: str
    free_ratio: float
    mean_occupancy: float
    incoming_ratio: float
    traffic_pressure: float
    timestamp: str


class V2XCommunication:
    def __init__(self, max_log_size=50000):
        self.max_log_size = int(max_log_size)
        self.vehicle_status_log = []
        self.recommendation_log = []
        self.zone_state_log = []

    def _now(self):
        return datetime.now().isoformat(timespec="seconds")

    def _append_limited(self, store, msg):
        store.append(msg)
        if len(store) > self.max_log_size:
            del store[: len(store) - self.max_log_size]

    def send_vehicle_status(self, vehicle_id, current_edge, current_zone=None):
        msg = VehicleStatusMessage(
            vehicle_id=str(vehicle_id),
            current_edge=str(current_edge),
            current_zone=current_zone,
            timestamp=self._now(),
        )
        self._append_limited(self.vehicle_status_log, msg)
        return msg

    def send_parking_recommendation(self, vehicle_id, zone_name, parking_id, score=0.0):
        msg = ParkingRecommendationMessage(
            vehicle_id=str(vehicle_id),
            zone_name=str(zone_name),
            parking_id=str(parking_id),
            score=float(score),
            timestamp=self._now(),
        )
        self._append_limited(self.recommendation_log, msg)
        return msg

    def publish_zone_state(self, zone_name, free_ratio, mean_occupancy, incoming_ratio, traffic_pressure=1.0):
        msg = ZoneStateMessage(
            zone_name=str(zone_name),
            free_ratio=float(free_ratio),
            mean_occupancy=float(mean_occupancy),
            incoming_ratio=float(incoming_ratio),
            traffic_pressure=float(traffic_pressure),
            timestamp=self._now(),
        )
        self._append_limited(self.zone_state_log, msg)
        return msg

    def get_vehicle_status_dicts(self):
        return [asdict(msg) for msg in self.vehicle_status_log]

    def get_recommendation_dicts(self):
        return [asdict(msg) for msg in self.recommendation_log]

    def get_zone_state_dicts(self):
        return [asdict(msg) for msg in self.zone_state_log]

    def reset(self):
        self.vehicle_status_log.clear()
        self.recommendation_log.clear()
        self.zone_state_log.clear()


class V2XFeatureExtractor:
    def __init__(self, max_vehicles=10, max_edges=20):
        self.max_vehicles = int(max_vehicles)
        self.max_edges = int(max_edges)

    def get_vehicle_features(self, conn):
        if conn is None:
            return np.zeros(self.max_vehicles * 4, dtype=np.float32)

        try:
            vehicles = list(conn.vehicle.getIDList())
        except Exception:
            return np.zeros(self.max_vehicles * 4, dtype=np.float32)

        features = []
        for vid in vehicles[: self.max_vehicles]:
            try:
                speed = float(conn.vehicle.getSpeed(vid))
                pos = conn.vehicle.getPosition(vid)
                road_id = conn.vehicle.getRoadID(vid)
                valid_edge = 0.0 if (not road_id or str(road_id).startswith(":")) else 1.0
                features.append([speed / 20.0, pos[0] / 10000.0, pos[1] / 10000.0, valid_edge])
            except Exception:
                features.append([0.0, 0.0, 0.0, 0.0])

        while len(features) < self.max_vehicles:
            features.append([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32).flatten()

    def get_traffic_features(self, conn):
        if conn is None:
            return np.zeros(self.max_edges, dtype=np.float32)

        try:
            edges = list(conn.edge.getIDList())
        except Exception:
            return np.zeros(self.max_edges, dtype=np.float32)

        densities = []
        for edge_id in edges[: self.max_edges]:
            try:
                d = float(conn.edge.getLastStepVehicleNumber(edge_id))
                densities.append(d / 50.0)
            except Exception:
                densities.append(0.0)

        while len(densities) < self.max_edges:
            densities.append(0.0)

        return np.array(densities, dtype=np.float32)

    def get_network_pressure(self, conn):
        features = self.get_traffic_features(conn)
        if len(features) == 0:
            return 1.0
        mean_density = float(np.mean(features))
        return float(max(1.0, min(1.6, 1.0 + mean_density)))
