import os
import sys
import csv
import random
import time
import numpy as np
import traci

# --- Force Python à trouver les dossiers 'env' et 'v2x' ---
# Configuration dynamique du chemin d'accès racine pour l'importation inter-modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Importation flexible ---
# Gestion adaptative des chemins selon que le script est lancé depuis la racine ou le sous-dossier env
try:
    from env.parking_manager import ParkingManager
    from env.agent_mapper import AgentMapper
    from v2x.v2x_comm import V2XCommunication, V2XFeatureExtractor
except ImportError:
    from parking_manager import ParkingManager
    from agent_mapper import AgentMapper
    from v2x.v2x_comm import V2XCommunication, V2XFeatureExtractor


class MultiAgentParkingEnv:
    """
    Environnement multi-agents personnalisé modélisent la gestion intelligente des parkings urbains.
    Compatible avec l'API d'apprentissage par renforcement (Méthodes standards : reset, step).
    """
    def __init__(
        self,
        sumo_cfg=None, 
        parkings_xml="",
        agents_json="",
        agent_name="agent_1",
        top_k=5,
        max_steps=3000,
        agent_detection_radius=1500.0,
        use_gui=False,
        warmup_steps=150,
        gui_delay=0.03,
        parking_demand_prob=0.6,
        **kwargs 
    ):
        # Résolution de la commande de configuration de la simulation SUMO
        self.sumo_cfg = sumo_cfg if sumo_cfg is not None else kwargs.get("sumo_cmd")
        
        # Initialisation des paramètres de structure et d'agent local
        self.parkings_xml = parkings_xml
        self.agents_json = agents_json
        self.agent_name = agent_name
        self.use_gui = bool(use_gui)

        self.warmup_steps = int(warmup_steps)
        self.gui_delay = float(gui_delay)
        self.top_k = int(top_k)                       # Nombre maximal de parkings candidats retenus par état
        self.max_steps = int(max_steps)               # Durée maximale d'un épisode/round de simulation

        self.parking_demand_prob = float(parking_demand_prob)

        # Suivi de l'état des véhicules pour le traitement séquentiel des requêtes de stationnement
        self.known_vehicles = set()                   # Véhicules introduits dans le réseau
        self.processed_vehicles = set()               # Véhicules ayant déjà reçu une décision définitive
        self.parking_request_queue = []               # File d'attente active des demandes de stationnement
        self.max_requests_per_step = 40               # Limite de requêtes traitées par pas de simulation (décongestion)
        self.max_queue_size = 500                     # Taille maximale de la file d'attente pour saturer la mémoire
        self.retry_not_eligible = True                # Réinsère un véhicule en file s'il sort temporairement de la zone

        # Cartographie spécifique des rayons d'action (portée géographique) pour chaque agent urbain
        agent_radius_map = {
            "agent_1": 1600.0,
            "agent_2": 1500.0,
            "agent_3": 2200.0,
            "agent_4": 1750.0,
        }
        self.agent_detection_radius = float(
            agent_radius_map.get(agent_name, agent_detection_radius)
        )

        self.conn = None                              # Instance de connexion TraCI vers SUMO
        self.step_count = 0                           # Horloge interne du pas de simulation courant

        # Instanciation des gestionnaires métiers, du routage V2X et de l'extraction de caractéristiques réseau
        self.parking_manager = ParkingManager(parkings_xml)
        self.agent_mapper = AgentMapper(agents_json)
        self.v2x = V2XCommunication()
        self.v2x_features = V2XFeatureExtractor(max_vehicles=10, max_edges=20)

        # Variables d'état transitoires associées au véhicule en cours d'évaluation par l'étape de RL
        self.current_vehicle_id = None
        self.current_mode = None
        self.current_radius = None
        self.current_candidates = []
        self.current_agent_distance = None
        self.current_source_agent = None

        # Compteurs de performance opérationnelle (Attributions réussies)
        self.agent_assignment_count = 0
        self.global_assignment_count = 0

        # Ensembles de catégorisation locale pour le monitoring fin des véhicules par round
        self.local_assigned_vehicles = set()
        self.local_failed_vehicles = set()
        self.local_pending_vehicles = set()

        # Dictionnaire de comptage des assignations physiques par identifiant de parking (PID)
        self.parking_usage_count = {
            pid: 0 for pid in self.parking_manager.parking_meta.keys()
        }

        self._cached_parking_positions = {}           # Cache géospatial des coordonnées des parkings

        # Ajustement des limites de recherche de véhicules pour équilibrer la charge des calculs par agent
        if self.agent_name == "agent_2":
            self.vehicle_search_limit = 80
        elif self.agent_name == "agent_3":
            self.vehicle_search_limit = 90
        else:
            self.vehicle_search_limit = 60

        # --- ÉVALUATION DES DIMENSIONS ---
        # Spécification stricte des dimensions attendues par le réseau de neurones DQN
        self.state_dim = 8 + (self.top_k * 8)
        self.action_dim = self.top_k * 3

        # Configuration des structures d'exportation pour l'enregistrement des historiques
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(project_root, "outputs")
        self.decisions_csv_path = os.path.join(self.output_dir, "decisions_log.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        self._ensure_decisions_csv_header()

        self.agent_usage = {name: 0 for name in self.agent_mapper.get_agent_names()}

        # ====================================================================
        # 💳 CORRECTION 2 : AJOUT DE LA LOGIQUE CRM (CHARGEMENT DES ABONNÉS)
        # ====================================================================
        self.abonnes_list = set()
        abonnes_xml = os.path.join(project_root, "scenarios", "luxembourg", "madina_abonnes.xml")
        
        if os.path.exists(abonnes_xml):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(abonnes_xml)
                root = tree.getroot()
                for sub in root.findall("subscriber"):
                    sub_id = sub.get("id")
                    if sub_id:
                        self.abonnes_list.add(sub_id)
                print(f"💳 [{self.agent_name}] Base CRM chargée : {len(self.abonnes_list)} abonnés actifs configurés.")
            except Exception as e:
                print(f"⚠️ [{self.agent_name}] Erreur lors du parsing du fichier CRM madina_abonnes.xml : {e}")
        else:
            print(f"⚠️ [{self.agent_name}] Alerte CRM : Fichier madina_abonnes.xml introuvable à l'adresse : {abonnes_xml}")

    def get_traffic_pressure(self):
        """Calcule l'indice de pression ou de congestion sur le réseau routier en temps réel."""
        if self.conn is not None:
            try:
                pressure = self.v2x_features.get_network_pressure(self.conn)
                return float(pressure)
            except Exception:
                pass

        # Profil de repli périodique simulant des heures de pointe si TraCI n'est pas encore initialisé
        cycle = self.step_count % 1200
        if 200 <= cycle <= 450:
            return 1.35
        if 700 <= cycle <= 950:
            return 1.45
        return 1.0

    def _get_average_system_occupancy(self):
        """Calcule le taux d'occupation physique moyen sur l'ensemble des parkings du réseau."""
        occupancies = []
        for pid in self.parking_manager.parking_meta.keys():
            cap = max(self.parking_manager.get_capacity(pid), 1)
            real_occ = self.parking_manager.get_real_occupancy(pid)
            occupancies.append(real_occ / cap)
        return np.mean(occupancies) if occupancies else 0.0

    def get_dynamic_demand_probability(self):
        """Génère une probabilité adaptative de demande de parking inversement proportionnelle à la saturation globale."""
        avg_occupancy = self._get_average_system_occupancy()
        
        if avg_occupancy < 0.50:
            return 0.85
        elif avg_occupancy < 0.65:
            return 0.50
        else:
            return 0.10
    
    def _publish_zone_state(self):
        """Diffuse l'état macroscopique de la zone via le module de communication V2X."""
        if len(self.parking_manager.parking_meta) == 0:
            return

        occ = []
        free = []
        inc = []

        # Collecte et normalisation des métriques d'infrastructure par rapport à la capacité nominale
        for pid in self.parking_manager.parking_meta:
            cap = max(self.parking_manager.get_capacity(pid), 1)
            occ.append(self.parking_manager.get_real_occupancy(pid) / cap)
            free.append(self.parking_manager.get_free_slots(pid) / cap)
            inc.append(self.parking_manager.get_incoming_count(pid) / cap)

        try:
            self.v2x.publish_zone_state(
                zone_name=self.agent_name,
                free_ratio=float(np.mean(free)),
                mean_occupancy=float(np.mean(occ)),
                incoming_ratio=float(np.mean(inc)),
                traffic_pressure=self.get_traffic_pressure(),
            )
        except TypeError:
            # Compatibilité descendante si la signature de la méthode v2x n'intègre pas la pression
            self.v2x.publish_zone_state(
                zone_name=self.agent_name,
                free_ratio=float(np.mean(free)),
                mean_occupancy=float(np.mean(occ)),
                incoming_ratio=float(np.mean(inc)),
            )
        except Exception:
            pass

    def _get_corrected_predicted_occupancy(self, pid):
        """Calcule une occupation prédictive court terme corrigée en incluant les flux de véhicules entrants."""
        real = int(self.parking_manager.get_real_occupancy(pid))
        incoming = int(self.parking_manager.get_incoming_count(pid))
        capacity = max(int(self.parking_manager.get_capacity(pid)), 1)

        # Application d'un plafond conservateur sur les véhicules entrants (max 35% de la capacité brute)
        incoming_capped = min(incoming, int(0.35 * capacity))
        predicted = real + incoming_capped

        return int(min(predicted, capacity))

    def _get_corrected_predicted_occupancy_ratio(self, pid):
        """Renvoie le ratio d'occupation prédictive corrigée (valeur normalisée entre 0.0 et 1.0)."""
        cap = max(int(self.parking_manager.get_capacity(pid)), 1)
        return float(self._get_corrected_predicted_occupancy(pid)) / float(cap)

    def warmup_until_traffic(self, min_vehicles=10, max_warmup_steps=None):
        """Fait progresser la simulation SUMO en amont de l'apprentissage pour injecter un flux de trafic initial minimum."""
        if max_warmup_steps is None:
            max_warmup_steps = self.warmup_steps

        for i in range(max_warmup_steps):
            self.conn.simulationStep()
            self.step_count += 1
            self.parking_manager.refresh()
            self._publish_zone_state()
            self.update_parking_requests()

            try:
                nveh = len(self.conn.vehicle.getIDList())
            except Exception:
                nveh = 0

            # Prise en compte du délai graphique si l'affichage SUMO-GUI est actif
            if self.use_gui and self.gui_delay > 0:
                time.sleep(self.gui_delay)

            # Sortie anticipée dès que le seuil de densité automobile requis est atteint
            if nveh >= min_vehicles:
                print(f"[WARMUP] {self.agent_name} | steps={i + 1} | vehicles={nveh}")
                return

        print(f"[WARMUP] {self.agent_name} | max steps atteints={max_warmup_steps}")

    def start(self):
        """
        Initialise la connexion TraCI et lance l'exécutable binaire de simulation SUMO.
        Intègre dynamiquement la fenêtre temporelle dictée par le pipeline multi-scénarios MADINA.
        """
        # Capture directe et robuste de l'horaire depuis les variables système héritées du pipeline
        begin_time = int(os.getenv("MADINA_BEGIN_TIME", 0))
        end_time = begin_time + self.max_steps

        # Détermination du binaire SUMO (avec ou sans GUI)
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        # Construction de la commande standardisée de simulation
        sumo_cmd = [
            sumo_binary,  # ✅ Correction validée : Remplacement de self.sumo_binary défectueux
            "-c", self.sumo_cfg,
            "--begin", str(begin_time),
            "--end", str(end_time),
            "--step-length", "1",
            "--quit-on-end", "true",
            "--duration-log.disable", "true",
            "--no-step-log", "true",
            "--collision.action", "warn",
            "--time-to-teleport", "60",
            "--ignore-route-errors", "true",
        ]

        if self.use_gui:
            sumo_cmd += ["--start", "--delay", str(int(self.gui_delay * 1000))]

        # Lancement de l'instance SUMO dédiée à l'agent courant
        traci.start(sumo_cmd)
        self.conn = traci

        # Liaison et initialisation de l'infrastructure de parking
        self.parking_manager.set_connection(self.conn)
        self.parking_manager.initialize()
        self.v2x.reset()

        # Réinitialisation complète de l'ensemble des structures de données dynamiques au départ d'un round
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

        self.known_vehicles.clear()
        self.processed_vehicles.clear()
        self.parking_request_queue.clear()

        self.agent_usage = {name: 0 for name in self.agent_mapper.get_agent_names()}
        self.parking_usage_count = {
            pid: 0 for pid in self.parking_manager.parking_meta
        }
        self._cached_parking_positions = {}

        # Lancement de la phase transitoire d'injection du trafic
        self.warmup_until_traffic(min_vehicles=10, max_warmup_steps=self.warmup_steps)

    def close(self):
        """Ferme de manière sécurisée et isole l'instance de communication TraCI en cours."""
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
        """Garantit la présence et l'intégrité des en-têtes de colonnes dans le fichier CSV d'historique."""
        if not os.path.exists(self.decisions_csv_path):
            try:
                with open(self.decisions_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step", "agent_name", "source_agent", "vehicle_id", "mode",
                        "parking_id", "distance_m", "price", "free_slots", "capacity",
                        "real_occupancy", "incoming_count", "predicted_occupancy", "reward", "reason",
                    ])
            except PermissionError:
                pass

    def _log_decision_csv(
        self, vehicle_id, mode, parking_id, distance_m, price, free_slots,
        capacity, real_occupancy, incoming_count, predicted_occupancy, reward, reason, source_agent=None,
    ):
        """Enregistre de manière atomique une ligne de décision dans le fichier CSV d'analyse logs."""
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

    def update_parking_requests(self):
        """Met à jour et filtre la file d'attente globale des demandes de stationnement."""
        try:
            active_vehicles = set(self.conn.vehicle.getIDList())
        except Exception:
            return

        # Nettoyage automatique : supprime les véhicules ayant quitté la simulation ou déjà traités
        if self.parking_request_queue:
            self.parking_request_queue = [
                vid for vid in self.parking_request_queue
                if vid in active_vehicles and vid not in self.processed_vehicles
            ]

        # Détection des nouveaux véhicules entrants pour l'évaluation de leur besoin de stationnement
        new_vehicles = active_vehicles - self.known_vehicles
        self.known_vehicles.update(new_vehicles)

        demand_prob = self.get_dynamic_demand_probability()

        for vid in new_vehicles:
            if vid in self.processed_vehicles:
                continue

            try:
                road_id = self.conn.vehicle.getRoadID(vid)
                # Exclusion des jonctions internes de l'infrastructure SUMO (commençant par ':')
                if not road_id or road_id.startswith(":"):
                    continue
            except Exception:
                continue

            # Échantillonnage stochastique selon la probabilité dynamique calculée
            if random.random() <= demand_prob:
                self.parking_request_queue.append(vid)

        # Maintien strict de la file d'attente sous le plafond maximal autorisé
        if len(self.parking_request_queue) > self.max_queue_size:
            self.parking_request_queue = self.parking_request_queue[-self.max_queue_size:]

    def get_client_mode(self):
        """Profilage stochastique du comportement/profil de préférence du conducteur."""
        r = random.random()
        if r < 0.25:
            return "close"       # Recherche en priorité la proximité géographique
        if r < 0.65:
            return "cheap"       # Recherche en priorité l'optimisation des coûts (tarifs bas)
        return "balanced"        # Compromis multicritère équilibré

    def get_radius(self, mode):
        """Détermine le rayon de recherche géographique acceptable en fonction du profil conducteur et de la congestion."""
        pressure = self.get_traffic_pressure()

        if mode == "close":
            return 500.0 if pressure < 1.3 else 700.0

        if mode == "cheap":
            return 1500.0 if pressure < 1.3 else 2000.0

        return 1000.0 if pressure < 1.3 else 1400.0

    def compute_candidate_score(self, pid, dist, mode):
        """
        Calcule un score d'évaluation heuristique a priori pour le tri initial des parkings candidats.
        Modifié : Intègre une estimation physique temporelle linéaire instantanée pour éviter le surcoût TraCI.
        """
        base_price = 4.50
        price = base_price

        cap = max(self.parking_manager.get_capacity(pid), 1)
        free = self.parking_manager.get_free_slots(pid)
        incoming = self.parking_manager.get_incoming_count(pid)

        # --- ÉVALUATION AVANCÉE : Estimation de l'ETA linéaire en minutes (Vitesse moy : 30 km/h ~ 8.3 m/s) ---
        time_estimation_min = (float(dist) / 8.3) / 60.0

        # Normalisation linéaire des variables d'environnement entre 0.0 et 1.0
        dist_norm = min(float(dist) / 1300.0, 1.0)
        price_norm = min(float(price) / 6.5, 1.0)
        free_ratio = float(free) / float(cap)
        pred_occ_ratio = self._get_corrected_predicted_occupancy_ratio(pid)
        incoming_ratio = min(float(incoming) / float(cap), 1.0)

        # Calcul des ratios d'utilisation historiques globaux et locaux pour l'équilibrage de charge
        usage = self.parking_usage_count.get(pid, 0)
        total_usage = max(sum(self.parking_usage_count.values()), 1)
        usage_ratio = float(usage) / float(total_usage)

        agent_total = max(self.agent_usage.get(self.agent_name, 1), 1)
        local_usage_ratio = float(usage) / float(agent_total)

        # Calcul des pénalités de sur-utilisation et de dominance infrastructurelle
        dominance_penalty = min(1.20 * np.sqrt(usage_ratio), 0.75)
        local_penalty = min(0.75 * np.sqrt(local_usage_ratio), 0.55)

        # Application de pénalités par paliers en situation de saturation prédictive critique
        saturation_penalty = 0.0
        if pred_occ_ratio > 0.60:
            saturation_penalty += 0.50
        if pred_occ_ratio > 0.80:
            saturation_penalty += 1.50
        if pred_occ_ratio > 0.90:
            saturation_penalty += 3.50

        # Pondérations objectives adaptatives selon les préférences du profil conducteur actif
        if mode == "close":
            score = (
                4.50 * time_estimation_min  # Focus lourd sur le temps de parcours (ETA)
                + 0.10 * price_norm
                + 0.18 * pred_occ_ratio
                + 0.05 * incoming_ratio
                - 0.08 * free_ratio
            )
        elif mode == "cheap":
            score = (
                1.50 * time_estimation_min
                + 1.30 * price_norm
                + 3.50 * pred_occ_ratio     # Évite fortement les zones saturées
                + 0.05 * incoming_ratio
                - 0.08 * free_ratio
            )
        else:
            score = (
                2.50 * time_estimation_min
                + 0.75 * price_norm
                + 2.00 * pred_occ_ratio
                + 0.05 * incoming_ratio
                - 0.10 * free_ratio
            )

        # Rapprochement structurel : Pénalité budgétaire anticipée pour le profil cheap sur l'hypercentre
        if mode == "cheap" and pid in ["P25", "P16"]:
            score += 2.0

        # Consolidation finale du score heuristique
        score += dominance_penalty
        score += local_penalty
        score += saturation_penalty

        return float(score)

    def _get_parking_positions(self):
        """Génère l'index spatial bidimensionnel des coordonnées géométriques de chaque parking."""
        positions = {}
        for pid, meta in self.parking_manager.parking_meta.items():
            try:
                lane = meta["lane"]
                shape = self.conn.lane.getShape(lane)
                if shape:
                    positions[pid] = shape[len(shape) // 2]  # Extraction du point médian du segment de voie
            except Exception:
                continue
        return positions

    def _vehicle_is_valid_for_assignment(self, vid):
        """Vérifie si le véhicule est structurellement disponible pour une nouvelle affectation."""
        if vid in self.local_assigned_vehicles:
            return False
        if vid in self.local_pending_vehicles:
            return False
        if vid in self.local_failed_vehicles:
            return False
        if vid in self.processed_vehicles:
            return False
        if hasattr(self.parking_manager, "is_vehicle_valid"):
            return self.parking_manager.is_vehicle_valid(vid)
        return False

    def _vehicle_is_eligible_for_agent(self, vid):
        """Valide si le véhicule se trouve à portée géographique de l'agent urbain local."""
        try:
            if not self._vehicle_is_valid_for_assignment(vid):
                return False, None, None

            pos = self.conn.vehicle.getPosition(vid)
            road_id = self.conn.vehicle.getRoadID(vid)
            self.v2x.send_vehicle_status(vid, road_id, current_zone=self.agent_name)

            # Évaluation et tri des agents géographiquement les plus pertinents via l'AgentMapper
            candidate_agents = self.agent_mapper.score_agents_with_seed_bonus(
                vehicle_pos=pos,
                parking_positions=self._cached_parking_positions,
                k=4,
            )
            candidate_names = [a for a, _ in candidate_agents]

            if self.agent_name in candidate_names:
                idx = candidate_names.index(self.agent_name)
                source_agent, dist_score = candidate_agents[idx]

                # Application d'une marge de tolérance sur le rayon d'action selon la sensibilité de l'agent
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
        """Parcourt la file d'attente et retourne la liste des véhicules éligibles prêts à être traités."""
        self.update_parking_requests()

        valid = []
        retry_queue = []
        checked = 0

        while self.parking_request_queue and checked < self.max_requests_per_step:
            vid = self.parking_request_queue.pop(0)
            checked += 1

            if vid in self.processed_vehicles:
                continue

            if not self._vehicle_is_valid_for_assignment(vid):
                self.processed_vehicles.add(vid)
                continue

            ok, pos, source_agent = self._vehicle_is_eligible_for_agent(vid)

            if not ok:
                if self.retry_not_eligible:
                    retry_queue.append(vid)  # Conservation dans la pile pour réévaluation ultérieure
                else:
                    self.processed_vehicles.add(vid)
                continue

            nearest_agent, dist_to_agent = self.agent_mapper.get_nearest_agent(pos)
            valid.append((vid, dist_to_agent, source_agent, nearest_agent))

            if len(valid) >= limit:
                break

        self.parking_request_queue.extend(retry_queue)

        if len(self.parking_request_queue) > self.max_queue_size:
            self.parking_request_queue = self.parking_request_queue[-self.max_queue_size:]

        return valid

    def _build_candidates_for_vehicle(self, vid):
        """Interroge l'infrastructure sous-jacente pour extraire et ordonner les 'top_k' parkings candidats valides."""
        mode = self.get_client_mode()
        radius = self.get_radius(mode)
        
        # --- ALIGNEMENT PARFAIT : Rayon maximum fixé de manière homogène sur 1300m ---
        max_distance = 1300.0

        try:
            raw_candidates = self.parking_manager.get_candidate_parkings(
                self.conn,
                vid,
                radius,
                include_reachability=True,
                allow_global_fallback=False,
                max_global_candidates=15,
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

        # Tri ascendant basé sur la valeur heuristique (Recherche de la minimisation de la fonction de coût/score)
        scored.sort(key=lambda x: x[2])
        return mode, radius, scored[: self.top_k]

    def _build_state(self, vid, mode, radius, agent_distance, candidates, source_agent):
        """Formate et vectorise les variables contextuelles sous forme de vecteur d'état NumPy structuré."""
        state = []

        # Encodage One-Hot discret du profil d'exigence du conducteur
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

        # Intégration des caractéristiques scalaires globales normalisées
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

        # Injection séquentielle des caractéristiques propres à chaque parking candidat (8 variables par candidat)
        for pid, dist, _score in candidates:
            price = 4.50
            free = self.parking_manager.get_free_slots(pid)
            cap = max(self.parking_manager.get_capacity(pid), 1)
            real_occ = self.parking_manager.get_real_occupancy(pid)
            incoming = self.parking_manager.get_incoming_count(pid)
            pred_occ_ratio = self._get_corrected_predicted_occupancy_ratio(pid)

            dist_norm = min(float(dist) / 1300.0, 1.0)
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

        # Zero-padding de sécurité pour garantir la stricte fixité de la dimension du vecteur d'état entrée du réseau
        while len(state) < self.state_dim:
            state.append(0.0)

        return np.array(state[: self.state_dim], dtype=np.float32)

    def reset(self):
        """Réinitialise le point d'entrée d'une action de RL et extrait le premier état valide de la simulation."""
        self.parking_manager.refresh()
        self._publish_zone_state()
        self._cached_parking_positions = self._get_parking_positions()

        # Récupération de la liste des véhicules valides présents à ce pas de temps
        valid_vehicles = self._get_valid_vehicles(limit=self.vehicle_search_limit)

        # Stratégies d'ordonnancement spécifiques aux politiques de tri de chaque agent
        if self.agent_name == "agent_1" and len(valid_vehicles) > 1:
            valid_vehicles = sorted(valid_vehicles, key=lambda x: x[1])
            head = valid_vehicles[: min(25, len(valid_vehicles))]
            random.shuffle(head)
            valid_vehicles = head + valid_vehicles[min(25, len(valid_vehicles)):]

        if self.agent_name == "agent_3" and len(valid_vehicles) > 1:
            random.shuffle(valid_vehicles)

        # Recherche séquentielle jusqu'à la constitution d'un état possédant des candidats
        for vid, dist_to_agent, source_agent, nearest_agent in valid_vehicles:
            if not self._vehicle_is_valid_for_assignment(vid):
                continue

            result = self._build_candidates_for_vehicle(vid)
            if result is None or len(result) == 0:
                continue

            mode, radius, candidates = result
            if len(candidates) == 0:
                self.processed_vehicles.add(vid)
                continue

            # Verrouillage des variables d'état courantes pour l'appel de la fonction step() subséquente
            self.current_vehicle_id = vid
            self.current_mode = mode
            self.current_radius = radius
            self.current_candidates = candidates
            self.current_agent_distance = dist_to_agent
            self.current_source_agent = source_agent if source_agent is not None else nearest_agent
            self.local_pending_vehicles.add(vid)

            return self._build_state(
                vid, mode, radius, dist_to_agent, candidates, self.current_source_agent,
            )

        # Si aucun véhicule n'exprime de besoin à ce pas de temps, renvoi d'un vecteur nul par défaut
        self.current_vehicle_id = None
        self.current_mode = None
        self.current_radius = None
        self.current_candidates = []
        self.current_agent_distance = None
        self.current_source_agent = None
        return np.zeros(self.state_dim, dtype=np.float32)

    def _compute_gagnant_gagnant_reward(self, assigned, mode, dist, price, price_level, predicted_occupancy=0, capacity=1):
        """
        Calcule la récompense multi-objectif équilibrée (Équilibre de Nash - Gagnant/Gagnant).
        Modifié : Intègre un bonus coopératif territorial de Load Balancing multi-agents.
        """
        if not assigned:
            return -15.0  # Pénalité d'échec robuste pour stabiliser le succès dans la plage cible 80%-90%

        # 1. CAPITAL DE BASE POUR REWARD GLOBALE EXCELLENTE
        base_success_reward = 8.0
        
        # 2. RENTABILITÉ DU SYSTÈME (Yield Management)
        if price_level == "premium":
            revenue_reward = 4.0
        elif price_level == "standard":
            revenue_reward = 2.5
        else:
            revenue_reward = 1.0

        # 3. PÉNALITÉ ET NORMALISATION DE LA DISTANCE (Alignée sur le nouveau max de 1300m)
        dist_norm = min(float(dist) / 1300.0, 1.0)
        distance_penalty = - 4.0 * dist_norm  
        
        # BONUS CRITIQUE DE PROXIMITÉ (Pour attirer l'IA sous les 800m et faire chuter la moyenne)
        if dist <= 400.0:
            distance_penalty += 5.0
        elif dist <= 800.0:
            distance_penalty += 2.5

        # 4. INCITATION À L'OCCUPATION (Cible 60% d'équilibrage de charge urbaine)
        cap = max(int(capacity), 1)
        pred_occ_ratio = float(predicted_occupancy) / float(cap)
        
        occupancy_bonus = 0.0
        if pred_occ_ratio < 0.40:
            occupancy_bonus += 3.0 * (0.40 - pred_occ_ratio)
        elif 0.40 <= pred_occ_ratio <= 0.75:
            occupancy_bonus += 2.0
        else:
            # Pénalité de saturation exponentielle à l'approche de 1.0 (100%)
            occupancy_bonus -= 5.0 * (pred_occ_ratio - 0.75) - np.exp(10 * (pred_occ_ratio - 0.90))    
    
        # 5. SATISFACTION CONDUCTEUR (Équilibre de Nash)
        user_penalty = 0.0
        if mode == "cheap" and price_level == "premium":
            user_penalty -= 6.0
        if mode == "close" and dist > 800.0:
            user_penalty -= 5.0

        # --- NOUVEAUTÉ : Bonus Coopératif Multi-Agent de Redistribution de Charge (Load Balancing) ---
        bonus_load_balancing = 0.0
        try:
            p25_occ_ratio = float(self.parking_manager.get_real_occupancy("P25")) / 50.0
            # Si le pole central central P25 entre dans la zone rouge de saturation (> 75%)
            if p25_occ_ratio > 0.75:
                # Si l'agent réussit à placer la voiture sur un parking secondaire libre (< 40%)
                if pred_occ_ratio < 0.40:
                    bonus_load_balancing = 4.0 * (p25_occ_ratio - pred_occ_ratio)
        except Exception:
            pass

        return float(base_success_reward + revenue_reward + distance_penalty + occupancy_bonus + user_penalty + bonus_load_balancing)

    def step(self, action_payload):
        """
        Exécute un pas d'action combiné avec filtrage CRM transparent des abonnés.
        Version corrigée pour Dounia & Manel (Alignement des variables d'état).
        """
        self.parking_manager.refresh()
        self.update_parking_requests()

        # Si aucune requête active n'est en attente, progression de la simulation
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
                "queue_size": len(self.parking_request_queue),
                "processed_vehicles": len(self.processed_vehicles),
            }

        # Déballage des variables contextuelles mémorisées (Variables validées)
        vid = self.current_vehicle_id
        mode = self.current_mode             # ✅ ALIGNÉ : Alignement parfait
        candidates = self.current_candidates
        source_agent = self.current_source_agent

        # ====================================================================
        # 💳 FILTRAGE CRM TRANSPARENT : Traitement du trafic de transit non-abonné
        # ====================================================================
        if vid not in self.abonnes_list:
            # Évacuer proprement la requête locale de l'agent
            self.local_pending_vehicles.discard(vid)
            self.processed_vehicles.add(vid)
            
            # Faire avancer la simulation SUMO (Crucial pour éviter de figer le trafic !)
            self.conn.simulationStep()
            self.step_count += 1
            
            # Évaluation rigoureuse de la coupure de fin de round
            done = self.step_count >= self.max_steps
            next_state = self.reset()
            
            info = {"assigned": 0, "reason": "non_subscriber_ignored"}
            return next_state, 0.0, done, info

        idx = action_payload["parking_index"]
        price_level = action_payload["price_level"]

        # Traitement d'une action hors-bornes ou invalide (Sécurité structurelle)
        if idx < 0 or idx >= self.top_k or idx >= len(candidates):
            reward = -3.0
            assigned = 0
            reason = "invalid_action"
            self.local_pending_vehicles.discard(vid)
            self.local_failed_vehicles.add(vid)
            self.processed_vehicles.add(vid)

            self._log_decision_csv(
                vid, mode, "", None, None, None, None, None, None, None,
                reward, reason, source_agent
            )
        else:
            chosen_pid, chosen_dist, _ = candidates[idx]
            free_before = self.parking_manager.get_free_slots(chosen_pid)
            pred_ratio_before = self._get_corrected_predicted_occupancy_ratio(chosen_pid)

            # ====================================================================
            # --- STRATÉGIE 1 : TARIFICATION DYNAMIQUE (YIELD MANAGEMENT V2X) ---
            # ====================================================================
            traffic_pressure = self.get_traffic_pressure()
            
            final_price = self.parking_manager.compute_dynamic_price(
                price_level=price_level,
                mode=mode,
                traffic_pressure=traffic_pressure
            )

            # Évaluation déterministe des règles de décision du conducteur
            if not self.parking_manager.is_vehicle_valid(vid):
                ok = False
                forced_reason = "vehicle_invalid_before_assign"
            elif free_before <= 0:
                ok = False
                forced_reason = "parking_full_before_assign"
            elif pred_ratio_before >= 0.995:
                ok = False
                forced_reason = "parking_saturated_before_assign"
            elif mode == "cheap" and final_price > 5.50:
                ok = False  
                forced_reason = "rejected_by_driver_expensive"
            elif mode == "close" and chosen_dist > 1200.0:
                ok = False  
                forced_reason = "rejected_by_driver_too_far"
            else:
                ok = self.parking_manager.assign(vid, chosen_pid, traffic_pressure=traffic_pressure)
                forced_reason = "assign_failed_or_unreachable"

            if ok:
                self.agent_assignment_count += 1
                self.global_assignment_count += 1
                self.agent_usage[self.agent_name] += 1
                self.parking_usage_count[chosen_pid] = (
                    self.parking_usage_count.get(chosen_pid, 0) + 1
                )

                self.local_pending_vehicles.discard(vid)
                self.local_assigned_vehicles.add(vid)
                self.processed_vehicles.add(vid)

                free_slots = self.parking_manager.get_free_slots(chosen_pid)
                capacity = self.parking_manager.get_capacity(chosen_pid)
                real_occupancy = self.parking_manager.get_real_occupancy(chosen_pid)
                incoming_count = self.parking_manager.get_incoming_count(chosen_pid)
                predicted_occupancy = self._get_corrected_predicted_occupancy(chosen_pid)

                reward = self._compute_gagnant_gagnant_reward(
                    assigned=True,
                    mode=mode,
                    dist=chosen_dist,
                    price=final_price,
                    price_level=price_level,
                    predicted_occupancy=predicted_occupancy,  
                    capacity=capacity                        
                )

                assigned = 1
                reason = f"assigned_{price_level}"

                self.v2x.send_parking_recommendation(
                    vid, self.agent_name, chosen_pid, score=reward
                )

                self._log_decision_csv(
                    vid, mode, chosen_pid, chosen_dist, final_price, free_slots,
                    capacity, real_occupancy, incoming_count, predicted_occupancy, reward, reason, source_agent,
                )
            else:
                reward = -12.0  
                assigned = 0
                reason = forced_reason
                self.local_pending_vehicles.discard(vid)
                self.local_failed_vehicles.add(vid)
                self.processed_vehicles.add(vid)

                self._log_decision_csv(
                    vid, mode, chosen_pid, chosen_dist, final_price, None, None,
                    None, None, None, reward, reason, source_agent
                )

        # =========================================================================
        # --- STRATÉGIE C : ACHEMINEMENT PROACTIF SUR SEUIL CRITIQUE DE PRESSION V2X ---
        # =========================================================================
        if traffic_pressure > 1.30 and self.current_vehicle_id is not None:
            try:
                self.conn.vehicle.rerouteTraveltime(self.current_vehicle_id, currentTravelTimes=True)
            except Exception:
                pass

        # Exécution effective du pas physique de simulation dans SUMO
        self.conn.simulationStep()
        self.step_count += 1

        if self.step_count % 100 == 0:
            print(
                f"[{self.agent_name}] step={self.step_count} "
                f"queue={len(self.parking_request_queue)} "
                f"processed={len(self.processed_vehicles)} "
                f"usage={self.agent_usage}"
            )

        # Évaluation de la condition terminale d'un épisode de simulation
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
            "queue_size": len(self.parking_request_queue),
            "processed_vehicles": len(self.processed_vehicles),
        }
        return next_state, float(reward), done, info

    # ====================================================================
    # --- STRATÉGIE D : SUIVI DES ROUNDS (HISTORIQUE PAR PARKING) ---
    # ====================================================================

    def log_round_occupancy_history(self, round_num):
        """
        Prend une photo logistique des taux de charge physiques à la fin d'un round
        et écrit immédiatement les données sur le disque dur (Frugalité RAM).
        """
        # Définition sécurisée du chemin d'exportation
        history_csv_path = os.path.join(self.output_dir, "parking_round_occupancy_history.csv")
        file_exists = os.path.exists(history_csv_path)
        
        try:
            with open(history_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Écriture de l'en-tête structurelle initiale s'il s'agit du Round 1
                if not file_exists:
                    writer.writerow([
                        "round", 
                        "parking_id", 
                        "capacity", 
                        "avg_real_occupancy_percent", 
                        "avg_predicted_occupancy_percent"
                    ])
                
                # Capture des données de charge urbaine brique par brique
                for pid in self.parking_manager.parking_meta.keys():
                    cap = max(self.parking_manager.get_capacity(pid), 1)
                    real_occ = self.parking_manager.get_real_occupancy(pid)
                    pred_occ_ratio = self._get_corrected_predicted_occupancy_ratio(pid)
                    
                    real_occ_percent = (float(real_occ) / float(cap)) * 100.0
                    pred_occ_percent = float(pred_occ_ratio) * 100.0
                    
                    writer.writerow([
                        int(round_num), 
                        pid, 
                        cap, 
                        round(real_occ_percent, 2), 
                        round(pred_occ_percent, 2)
                    ])
                    
            print(f"📊 [MADINA LOG] Historique d'occupation du Round {round_num} sauvegardé avec succès.")
        except Exception as e:
            print(f"⚠️ [MADINA WARNING] Erreur lors de l'écriture de l'historique : {e}")