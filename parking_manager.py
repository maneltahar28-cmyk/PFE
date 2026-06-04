import os
import csv
import random
import numpy as np
import xml.etree.ElementTree as ET


class ParkingManager:
    """
    Gestionnaire centralisé de l'infrastructure des parkings du projet Madina.
    Supervise l'état d'occupation, les réservations entrantes, la tarification dynamique,
    l'attribution d'itinéraires et le suivi logistique (logs IN/OUT) via SUMO/TraCI.
    """
    MIN_ASSIGN_ROUTE_LENGTH_M = 20.0
    
    # PARAMÈTRES DE STATIONNEMENT STOCHASTIQUE DE BASE (Rétention longue active)
    DEFAULT_PARKING_DURATION = 18000  # Espérance de 5 heures (en secondes) pour l'accumulation de charge
    PARKING_STD_DEV = 3600            # Écart-type de 1 heure

    def __init__(self, parkings_xml_path):
        self.parkings_xml_path = parkings_xml_path
        self.conn = None

        # Métadonnées de structure de l'infrastructure urbaine
        self.parking_meta = {}
        self.cap = {}

        # Tableaux de bord d'occupation et d'affectation dynamique
        self.real_occupancy = {}
        self.incoming_count = {}
        self.vehicle_target_parking = {}
        self.assignment_time = {}

        # Structures pour le tracking In/Out et exportation CSV
        self.previously_parked = {}
        self.active_reservations = {}  # Clé: vehicle_id, Valeur: {"t_in": step, "pid": pid}
        self.csv_filename = "parking_events_log.csv"

        # Chargement initial de l'infrastructure et initialisation du système de fichiers
        self._load_parkings_from_xml()
        self._init_events_csv()

    def set_connection(self, conn):
        """Associe l'instance active de connexion TraCI pour interagir avec le simulateur."""
        self.conn = conn

    def _load_parkings_from_xml(self):
        """Analyse le fichier de configuration XML pour extraire la topologie et la capacité des parkings."""
        tree = ET.parse(self.parkings_xml_path)
        root = tree.getroot()

        for elem in root.findall("parkingArea"):
            pid = elem.get("id")
            lane = elem.get("lane")

            if pid is None or lane is None:
                continue

            # Extraction de l'identifiant de la rue (edge) à partir du nom de la voie (lane)
            edge = lane.rsplit("_", 1)[0] if "_" in lane else lane

            # Gestion de la variabilité nominative de l'attribut de capacité dans le fichier XML
            cap_value = (
                elem.get("roadsideCapacity")
                or elem.get("roadSideCapacity")
                or elem.get("capacity")
                or "1"
            )

            try:
                cap = int(float(cap_value))
            except Exception:
                cap = 1

            # Structuration et mémorisation des configurations d'infrastructure du parking Area
            self.parking_meta[pid] = {
                "lane": lane,
                "edge": edge,
            }
            self.cap[pid] = max(cap, 1)
            
        # Initialisation de la mémoire de suivi des présences physiques pour chaque parking détecté
        self.previously_parked = {pid: set() for pid in self.parking_meta.keys()}

    def _init_events_csv(self):
        """Initialise le fichier CSV de tracking s'il n'existe pas."""
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["vehicle_id", "parking_id", "event_type", "time_step", "duration_steps"])

    def log_event(self, vid, pid, event_type, time, duration=""):
        """Écrit un événement physique IN/OUT dans le fichier de log."""
        with open(self.csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([vid, pid, event_type, time, duration])

    def track_parking_events(self, step_count, current_traci_instance):
        """
        Détecte en temps réel les entrées (IN) et sorties (OUT) des zones de parking.
        Compare l'état courant de la simulation avec le pas de temps précédent pour identifier les deltas.
        """
        for pid in self.parking_meta.keys():
            try:
                currently_parked = set(current_traci_instance.parkingarea.getVehicleIDs(pid))
            except Exception:
                continue

            # 1. Détection des entrées : Véhicules présents maintenant mais absents au pas de temps précédent
            for vid in currently_parked - self.previously_parked[pid]:
                self.log_event(vid, pid, event_type="IN", time=step_count)
                self.active_reservations[vid] = {"t_in": step_count, "pid": pid}

            # 2. Détection des sorties : Véhicules présents précédemment mais qui ont quitté la zone
            for vid in self.previously_parked[pid] - currently_parked:
                if vid in self.active_reservations:
                    t_in = self.active_reservations[vid]["t_in"]
                    duration = step_count - t_in
                    self.log_event(vid, pid, event_type="OUT", time=step_count, duration=duration)
                    self.active_reservations.pop(vid, None)
                else:
                    # Cas de secours si le véhicule s'était garé pendant la phase initiale de warmup
                    self.log_event(vid, pid, event_type="OUT", time=step_count, duration="Inconnu")

            # Sauvegarde de l'état actuel pour la comparaison lors du prochain pas de temps
            self.previously_parked[pid] = currently_parked

    def initialize(self):
        """Réinitialise complètement l'état du gestionnaire de parkings entre les rounds Flower."""
        self.real_occupancy = {pid: 0 for pid in self.parking_meta}
        self.incoming_count = {pid: 0 for pid in self.parking_meta}
        self.previously_parked = {pid: set() for pid in self.parking_meta}
        self.active_reservations = {}
        self.vehicle_target_parking = {}
        self.assignment_time = {}

    def _get_sim_time(self):
        """Récupère le temps absolu de la simulation SUMO en secondes."""
        if self.conn is None:
            return 0.0
        try:
            return float(self.conn.simulation.getTime())
        except Exception:
            return 0.0

    def refresh(self):
        """Met à jour les taux d'occupation réels et reconstruit l'état des flux de véhicules entrants."""
        if self.conn is None:
            return

        for pid in self.parking_meta:
            try:
                occ = int(self.conn.parkingarea.getVehicleCount(pid))
            except Exception:
                occ = self.real_occupancy.get(pid, 0)

            self.real_occupancy[pid] = max(0, occ)

        # Nettoyage et reconstruction prédictive des réservations de places
        self._rebuild_incoming_from_active_vehicles()

    def _rebuild_incoming_from_active_vehicles(self):
        """
        Filtre et recalcule le volume de véhicules en approche (incoming) pour chaque parking.
        Élimine les réservations obsolètes ou associées à des véhicules ayant quitté le réseau.
        """
        if self.conn is None:
            return

        try:
            active_vehicles = set(self.conn.vehicle.getIDList())
        except Exception:
            active_vehicles = set()

        current_time = self._get_sim_time()

        new_targets = {}
        new_incoming = {pid: 0 for pid in self.parking_meta}
        new_assignment_time = {}

        for vid, data in list(self.assignment_time.items()):
            try:
                pid, assign_time, travel_time = data
            except Exception:
                continue

            # Exclusion si le véhicule a fini son trajet ou a disparu des radars de SUMO
            if vid not in active_vehicles:
                continue

            if pid not in self.parking_meta:
                continue

            # Seuil de tolérance temporelle (80%) : si le temps de trajet estimé est largement dépassé,
            # on considère que le véhicule a atteint sa destination ou a été dérouté
            if current_time - float(assign_time) < float(travel_time) * 0.8:
                new_targets[vid] = pid
                new_incoming[pid] = new_incoming.get(pid, 0) + 1
                new_assignment_time[vid] = data

        # Consolidation des dictionnaires mis à jour
        self.vehicle_target_parking = new_targets
        self.incoming_count = new_incoming
        self.assignment_time = new_assignment_time

    def is_vehicle_valid(self, vehicle_id):
        """Valide l'éligibilité d'un véhicule à recevoir une attribution (exclut les doubles réservations)."""
        if self.conn is None:
            return False
        if vehicle_id in self.vehicle_target_parking:
            return False
        try:
            vehicle_ids = set(self.conn.vehicle.getIDList())
            if vehicle_id not in vehicle_ids:
                return False

            current_edge = self.conn.vehicle.getRoadID(vehicle_id)
            # Invalidation si le véhicule se trouve au milieu d'une intersection (jonction interne)
            if not current_edge or current_edge.startswith(":"):
                return False

            route = self.conn.vehicle.getRoute(vehicle_id)
            if route is None or len(route) == 0:
                return False

            return True
        except Exception:
            return False

    def get_capacity(self, pid):
        """Renvoie la capacité d'accueil maximale d'un parking."""
        return int(self.cap.get(pid, 1))

    def get_real_occupancy(self, pid):
        """Renvoie le nombre de véhicules physiquement stationnés dans le parking."""
        return int(self.real_occupancy.get(pid, 0))

    def get_incoming_count(self, pid):
        """Renvoie le nombre de véhicules actuellement en route vers ce parking."""
        return int(self.incoming_count.get(pid, 0))

    def get_predicted_occupancy(self, pid):
        """Calcule l'occupation à court terme attendue (Somme des véhicules garés et en approche)."""
        return int(self.get_real_occupancy(pid) + self.get_incoming_count(pid))

    def get_predicted_occupancy_ratio(self, pid):
        """Renvoie le ratio d'occupation prédictif normalisé (entre 0.0 et 1.0)."""
        cap = max(self.get_capacity(pid), 1)
        return float(self.get_predicted_occupancy(pid)) / float(cap)

    def get_free_slots(self, pid):
        """Renvoie le nombre de places réelles théoriquement encore disponibles (inclut la réserve incoming)."""
        cap = self.get_capacity(pid)
        occ = self.get_real_occupancy(pid)
        incoming = self.get_incoming_count(pid)
        return max(0, int(cap - occ - incoming))

    def _get_parking_position(self, conn, pid):
        """Extrait la coordonnée géospatiale 2D centrale (médiane) de la voie de stationnement."""
        try:
            lane = self.parking_meta[pid]["lane"]
            shape = conn.lane.getShape(lane)
            if not shape:
                return None
            return shape[len(shape) // 2]
        except Exception:
            return None

    def _find_route_to_parking(self, current_edge, pid):
        """Calcule le chemin optimal et la distance de routage dans le graphe routier vers un parking cible."""
        if self.conn is None:
            return None, None
        try:
            if not current_edge or current_edge.startswith(":"):
                return None, None
            if pid not in self.parking_meta:
                return None, None

            target_edge = self.parking_meta[pid]["edge"]
            if not target_edge or target_edge.startswith(":"):
                return None, None

            # Interrogation de l'A* / Dijkstra de SUMO pour trouver l'itinéraire optimal
            route = self.conn.simulation.findRoute(current_edge, target_edge)
            route_edges = list(getattr(route, "edges", []) or [])

            if len(route_edges) == 0:
                return None, None

            route_length = getattr(route, "length", None)
            if route_length is not None:
                route_length = float(route_length)

            return route, route_length
        except Exception:
            return None, None

    def get_candidate_parkings(self, conn, vehicle_id, radius, include_reachability=False, allow_global_fallback=False, max_global_candidates=15, saturation_threshold=0.985, max_distance=1500.0):
        """
        Filtre l'ensemble des infrastructures de stationnement pour dégager les parkings candidats éligibles.
        Prend en compte la portée du rayon d'action, le taux de saturation maximal et la connectivité topologique.
        """
        candidates = []
        global_candidates = []
        try:
            veh_pos = conn.vehicle.getPosition(vehicle_id)
            current_edge = conn.vehicle.getRoadID(vehicle_id)
        except Exception:
            return []

        if not current_edge or current_edge.startswith(":"):
            return []

        for pid in self.parking_meta:
            try:
                # Élimination immédiate des structures saturées au sens prédictif du terme
                if self.get_free_slots(pid) <= 0:
                    continue

                pred_ratio = self.get_predicted_occupancy_ratio(pid)
                if pred_ratio >= saturation_threshold:
                    continue

                park_pos = self._get_parking_position(conn, pid)
                if park_pos is None:
                    continue

                # Calcul de la distance euclidienne à vol d'oiseau
                air_dist = float(np.linalg.norm(np.array(veh_pos) - np.array(park_pos)))
                if air_dist > max_distance:
                    continue

                route_length = None
                if include_reachability:
                    route, route_length = self._find_route_to_parking(current_edge, pid)
                    if route is None:
                        continue
                    # Évite d'attribuer un parking situé trop près (sécurité anti-blocage)
                    if route_length is not None and route_length < self.MIN_ASSIGN_ROUTE_LENGTH_M:
                        continue

                # Détermination de la distance effective de déplacement (routière prioritaire sur euclidienne)
                effective_dist = air_dist
                if route_length is not None:
                    effective_dist = max(air_dist, float(route_length))

                item = (pid, effective_dist)
                global_candidates.append(item)

                # Validation par rapport au rayon de tolérance comportemental du conducteur
                if effective_dist <= radius:
                    candidates.append(item)
            except Exception:
                continue

        # Stratégie de secours adaptative globale si aucun parking n'est trouvé dans le rayon d'action restreint
        if len(candidates) == 0 and allow_global_fallback:
            global_candidates.sort(key=lambda x: x[1])
            return global_candidates[:max_global_candidates]

        candidates.sort(key=lambda x: x[1])
        return candidates

    # =========================================================================
    # 🌟 STRATÉGIE A : TARIFICATION DYNAMIQUE (YIELD MANAGEMENT V2X)
    # =========================================================================
    def compute_dynamic_price(self, price_level: str, mode: str, traffic_pressure: float) -> float:
        """
        Calcule le tarif exact en euros (Yield Management) appliqué à une décision.
        Garantit le respect des bornes strictes [3.00, 7.00] imposées par le cahier des charges.
        """
        base_pivot = 4.50
        
        # 1. Application du multiplicateur selon le palier choisi par la politique du DQN
        if price_level == "premium":
            price = base_pivot * 1.50  # Équivaut à 6.75€
        elif price_level == "discount":
            price = base_pivot * 0.70  # Équivaut à 3.15€
        else:
            price = base_pivot         # Équivaut à 4.50€

        # 2. Ajustement dynamique contextuel selon la pression du réseau routier V2X (Majoration haute à +25%)
        if traffic_pressure > 1.20:
            price *= min(1.0 + (traffic_pressure - 1.0) * 0.5, 1.25)
            
        # 3. Équilibre de Nash : Adaptation selon les profils psychographiques et la propension à payer
        if mode == "close":
            price *= 1.10  # Majoration de 10% (Forte disposition à payer pour l'urgence / proximité)
        elif mode == "cheap":
            price *= 0.90  # Réduction de 10% (Incitation d'attraction budgétaire pour profils économes)

        # 4. Clamping de sécurité réglementaire du cahier des charges [3€ - 7€]
        final_price = float(np.clip(price, 3.00, 7.00))
        return final_price

    # =========================================================================
    # 🌟 STRATÉGIE B : LOGIQUE DE LA DURÉE DE STATIONNEMENT DYNAMIQUE ADAPTATIVE
    # =========================================================================
    def _get_dynamic_stochastic_duration(self, pid, traffic_pressure):
        """
        Calcule une durée élastique. Plus le parking se sature ou plus la pression
        du trafic V2X est forte, plus la durée diminue pour accélérer le turnover et libérer l'espace.
        """
        pred_ratio = self.get_predicted_occupancy_ratio(pid)
        
        # Facteur d'élasticité nominal de base
        reduction_factor = 1.0
        
        # Algorithme d'auto-régulation urbaine active par seuils de criticité
        if pred_ratio > 0.85 or traffic_pressure > 1.25:
            reduction_factor = 0.60  # Réduction de 40% du temps sur place (Turnover et libération maximaux)
        elif pred_ratio > 0.65 or traffic_pressure > 1.05:
            reduction_factor = 0.80  # Réduction de 20% du temps sur place
            
        dynamic_mu = self.DEFAULT_PARKING_DURATION * reduction_factor
        
        # Échantillonnage stochastique gaussien (Loi Normale) autour de la moyenne élastique
        duration = random.gauss(dynamic_mu, self.PARKING_STD_DEV)
        
        # Bornes de sécurité physiques de l'API TraCI (Contraint entre 10 min et 4 heures)
        duration = max(600.0, duration)
        duration = min(14400.0, duration)
        
        return int(duration)

    def assign(self, vehicle_id, pid, traffic_pressure=1.0):
        """
        Affectation géospatiale robuste : modifie l'itinéraire de la cible dans SUMO, 
        injecte le dispositif de reroutage coopératif et applique la durée d'arrêt élastique.
        """
        if self.conn is None or pid not in self.parking_meta:
            return False
        if self.get_free_slots(pid) <= 0:
            return False

        try:
            if not self.is_vehicle_valid(vehicle_id):
                return False

            current_edge = self.conn.vehicle.getRoadID(vehicle_id)
            route, route_length = self._find_route_to_parking(current_edge, pid)

            if route is None:
                return False

            route_edges = list(getattr(route, "edges", []) or [])
            if len(route_edges) == 0:
                return False

            # Injection du nouvel itinéraire dans le moteur de simulation SUMO
            self.conn.vehicle.setRoute(vehicle_id, route_edges)

            # =========================================================================
            # 🌟 STRATÉGIE C : ACTIVATION DU DISPOSITIF DE RE-ROUTAGE COOPÉRATIF
            # =========================================================================
            # Permet au véhicule d'adapter dynamiquement sa trajectoire en cours de route via V2X
            self.conn.vehicle.setParameter(vehicle_id, "has.rerouting.device", "true")

            # --- APPLICATION ACTIVE DE LA REGULATION TEMPORELLE ELASTIQUE ---
            # Détermination stochastique adaptative de la durée d'arrêt
            parking_duration = self._get_dynamic_stochastic_duration(pid, traffic_pressure)

            # Commande TraCI ordonnant au véhicule de s'arrêter dans la zone de stationnement spécifiée
            self.conn.vehicle.setParkingAreaStop(
                vehicle_id,
                pid,
                duration=parking_duration,
            )

            # Estimation fine du temps de parcours (travel_time) basée sur la vitesse moyenne théorique (10 m/s)
            try:
                veh_pos = self.conn.vehicle.getPosition(vehicle_id)
                park_pos = self._get_parking_position(self.conn, pid)
                air_dist = float(np.linalg.norm(np.array(veh_pos) - np.array(park_pos))) if park_pos else 1000.0
                dist = max(air_dist, float(route_length)) if route_length is not None else air_dist
                travel_time = max(1.0, dist / 10.0)
            except Exception:
                travel_time = 9999.0

            # Enregistrement de la poignée de main temporelle pour le suivi prédictif de l'état entrant (incoming)
            sim_time = self._get_sim_time()
            self.assignment_time[vehicle_id] = (pid, sim_time, travel_time)
            self.vehicle_target_parking[vehicle_id] = pid
            self.incoming_count[pid] = self.incoming_count.get(pid, 0) + 1

            return True
        except Exception:
            return False