import os
import copy
import sys
import gc  # --- MODIFICATION : Indispensable pour la libération forcée de la RAM ---
from collections import OrderedDict

# Importation des bibliothèques principales : Flower pour le Fédéré, PyTorch pour le Deep Learning
import flwr as fl
import numpy as np
import torch
import torch.nn.functional as F

# Configuration des chemins systèmes pour permettre l'importation des modules locaux (env, rl)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Importations outils de simulation SUMO (TraCI pour l'interaction en temps réel avec l'interface de trafic)
import traci

# Importation de l'environnement de parking multi-agents et des composants du Dueling DDQN
from env.multi_agent_env import MultiAgentParkingEnv
from rl.dqn import DQN
from rl.replay import ReplayBuffer


class ParkingFlowerClient(fl.client.NumPyClient):
    """
    Client Flower local encapsulant un agent de Reinforcement Learning (DQN) 
    qui interagit avec son propre environnement de simulation SUMO.
    """
    def __init__(self, env_config, client_id="client_0", agent_name="agent_1"):
        # Initialisation et copie profonde de la configuration pour éviter les effets de bord
        self.env_config = copy.deepcopy(env_config)
        self.client_id = str(client_id)
        self.agent_name = str(agent_name)

        # Sélection automatique du GPU si disponible pour accélérer l'inférence et l'entraînement du DQN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gui = bool(self.env_config.get("use_gui", False))

        # Définition des dimensions de l'état (Observation Space) basée sur les 'top_k' meilleurs parkings
        self.top_k = int(self.env_config.get("top_k", 5))
        self.state_dim = 8 + (self.top_k * 8)
        
        # --- MODIFICATION : Espace d'actions étendu (5 parkings * 3 niveaux de prix = 15 actions) ---
        # L'espace est aplati : les indices 0-4 = discount, 5-9 = standard, 10-14 = premium
        self.action_dim = self.top_k * 3

        # Initialisation du réseau de neurones en ligne (online) et du réseau cible (target) pour le Double DQN
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Le modèle cible reste en mode évaluation (pas de dropout/batchnorm actif)

        # Optimiseur Adam avec un taux d'apprentissage standard de 3e-4 et initialisation du Replay Buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.memory = ReplayBuffer(50000)

        # Hyperparamètres du RL
        self.gamma = 0.95              # Facteur d'atténuation (discount factor) pour les récompenses futures
        self.batch_size = 64           # Taille des lots pour l'entraînement du DQN
        self.target_update_freq = 200  # Fréquence de mise à jour des poids du réseau cible (Target)
        self.train_step_count = 0      # Compteur interne de pas d'entraînement effectués

        # Variables de suivi pour la normalisation Welford / Mobile des récompenses
        self.reward_mean = 0.0
        self.reward_std = 1.0

    def _resolve_agents_json(self):
        """Récupère le chemin du fichier de configuration JSON des agents."""
        agents_json = self.env_config.get("agents_json")
        if agents_json:
            return agents_json
        raise KeyError("env_config doit contenir la clé 'agents_json'.")

    def _resolve_parkings_xml(self):
        """Récupère le chemin du fichier d'infrastructure XML des parkings SUMO."""
        parkings_xml = self.env_config.get("parkings_xml")
        if parkings_xml:
            return parkings_xml
        raise KeyError("env_config doit contenir la clé 'parkings_xml'.")

    def _build_env(self):
        """Instancie l'environnement de simulation SUMO avec les paramètres du client."""
        return MultiAgentParkingEnv(
            sumo_cfg=self.env_config["sumo_cfg"],
            parkings_xml=self._resolve_parkings_xml(),
            agents_json=self._resolve_agents_json(),
            agent_name=self.agent_name,
            top_k=self.top_k,
            max_steps=self.env_config.get("max_steps", 3000),
            agent_detection_radius=self.env_config.get("agent_detection_radius", 1500.0),
            use_gui=self.use_gui,
            warmup_steps=self.env_config.get("warmup_steps", 150),
            gui_delay=self.env_config.get("gui_delay", 0.03),
            parking_demand_prob=self.env_config.get("parking_demand_prob", 0.6),
        )

    def get_parameters(self, config):
        """Flower API : Extrait les poids globaux du modèle sous forme de listes NumPy pour les envoyer au serveur."""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Flower API : Reçoit les poids agrégés du serveur et met à jour le modèle local ainsi que le modèle cible."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.target_model.load_state_dict(self.model.state_dict())
        self.train_step_count = 0  # Réinitialisation du compteur de synchronisation cible au début du round

        # Purge adaptative de la mémoire entre les rounds pour éviter la saturation de la RAM
        if len(self.memory) > 15000:
            purge_main = len(self.memory.main_buffer) // 10
            purge_important = len(self.memory.important_buffer) // 10

            # Supprime les 10% de transitions les plus anciennes du buffer principal
            for _ in range(purge_main):
                if self.memory.main_buffer:
                    self.memory.main_buffer.popleft()

            # Supprime les 10% de transitions les plus anciennes du buffer important
            for _ in range(purge_important):
                if self.memory.important_buffer:
                    self.memory.important_buffer.popleft()

    def _select_action(self, state, env, epsilon):
        """Sélection de l'action via la stratégie Epsilon-Greedy avec masquage dynamique."""
        num_candidates = len(env.current_candidates)
        if num_candidates <= 0:
            return 0  # Sécurité : action par défaut si aucun candidat véhicule n'est disponible

        # --- MODIFICATION INTERNE : Exploration intelligente sans action invalide ---
        if np.random.random() < epsilon:
            # 1. Choisir au hasard un index de parking parmi ceux RÉELLEMENT disponibles (0 à num_candidates - 1)
            random_parking_idx = np.random.randint(0, num_candidates)
            
            # 2. Choisir au hasard l'une des 3 stratégies tarifaires (0: discount, 1: standard, 2: premium)
            random_tier = np.random.randint(0, 3)
            
            # 3. Reconstituer l'action combinée stricte (0-14) alignée sur self.top_k (5)
            # Formule mathématique inverse du décodage (tier * top_k + parking_idx)
            return int((random_tier * self.top_k) + random_parking_idx)

        # --- EXPLOITATION : Sélection via la politique apprise par le Dueling DQN ---
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_t).squeeze(0)
            masked_q = q_values.clone()

            # Masquage rigoureux sur les 3 tranches de l'espace d'action étendu
            # Si num_candidates < 5, on invalide les indices hors-bornes de chaque palier tarifaire
            for tier in range(3):
                start_invalid = (tier * self.top_k) + num_candidates
                end_invalid = (tier + 1) * self.top_k
                masked_q[start_invalid:end_invalid] = -1e9  # Attribution d'une valeur très basse pour interdire l'action

            return int(masked_q.argmax().item())

    def _warmup_until_candidates(self, env, max_warmup_steps=150):
        """Fait tourner la simulation à vide (warmup) jusqu'à ce que des véhicules candidats apparaissent."""
        state = env.reset()
        for _ in range(max_warmup_steps):
            if len(env.current_candidates) > 0:
                return state, True

            # Utilise un payload neutre (index 0, prix standard) si aucun candidat n'est encore détecté
            neutral_payload = {"parking_index": 0, "price_level": "standard"}
            next_state, _, done, _ = env.step(neutral_payload)
            state = next_state

            if done:
                state = env.reset()

        return state, len(env.current_candidates) > 0

    def _normalize_reward(self, reward):
        """Normalise en ligne les récompenses reçues pour stabiliser le gradient du DQN (clip entre -3 et 3)."""
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * (reward - self.reward_mean) ** 2
        norm = (reward - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-6)
        return np.clip(norm, -3.0, 3.0)

    def fit(self, parameters, config):
        """Flower API : Entraînement local de l'agent sur la simulation SUMO pour un round donné."""
        self.set_parameters(parameters)

        # Récupération de la configuration transmise par le serveur Flower
        epsilon = float(config.get("epsilon", 0.1))
        self.gamma = float(config.get("gamma", 0.99))
        self.batch_size = int(config.get("batch_size", 64))
        steps_per_round = int(config.get("steps_per_round", 1500))
        warmup_steps = int(config.get("warmup_steps", 150))
        min_useful_steps = int(config.get("min_useful_steps", 20))
        max_total_sim_steps = int(config.get("max_total_sim_steps", steps_per_round * 4))

        # Équilibrage de la charge de simulation si l'agent gère une zone plus dense (ex: agent_2)
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
            env.start()  # Lancement du processus SUMO / TraCI
            state, found_candidates = self._warmup_until_candidates(env, max_warmup_steps=warmup_steps)

            if not found_candidates:
                print(f"[{self.client_id}/{self.agent_name}] fit | aucun candidat trouvé après warmup")

            # Boucle d'entraînement locale protégée contre les déconnexions TraCI/SUMO
            while useful_steps < steps_per_round and total_sim_steps < max_total_sim_steps:
                if env.step_count >= env.max_steps:
                    break

                # --- MODIFICATION : Sécurité RAM (gc.collect) appliqué de manière cyclique ---
                # Évite les fuites de mémoire provoquées par l'accumulation d'états SUMO et PyTorch sur le long terme
                if total_sim_steps % 1000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # --- MODIFICATION : Appel en temps réel du suivi d'infrastructure (Entrées/Sorties) ---
                if hasattr(env, 'parking_manager') and env.parking_manager is not None:
                    env.parking_manager.track_parking_events(total_sim_steps, traci)

                num_candidates_before = len(env.current_candidates)
                raw_action = self._select_action(state, env, epsilon)
                
                # --- MODIFICATION : Décodage de la multi-action (Index Parking & Niveau Tarifaire) ---
                # Rétro-conversion de l'action scalaire (0-14) en composants métier pour l'environnement SUMO
                parking_index = raw_action % self.top_k
                price_tier = raw_action // self.top_k
                price_level = ["discount", "standard", "premium"][price_tier]

                action_payload = {
                    "parking_index": parking_index,
                    "price_level": price_level
                }
                
                try:
                    next_state, reward, done, info = env.step(action_payload)
                except (traci.exceptions.FatalTraCIError, traci.exceptions.TraCIException) as e:
                    print(f"⚠️ [{self.client_id}/{self.agent_name}] SUMO arrêté prématurément (TraCI clos) : {e}")
                    break
                    
                total_sim_steps += 1

                # ====================================================================
                # 💳 INTERCEPTION CRM : Sauter l'apprentissage pour le trafic libre
                # ====================================================================
                if info.get("reason") == "non_subscriber_ignored":
                    state = next_state
                    continue  # On passe instantanément au véhicule suivant dans SUMO sans toucher au DQN

                # On n'enregistre la transition que si un choix d'attribution était réellement possible
                if num_candidates_before > 0:
                    total_reward += float(reward)
                    total_assigned += float(info.get("assigned", 0))
                    useful_steps += 1

                    next_num_candidates = len(env.current_candidates)
                    reward_norm = self._normalize_reward(float(reward))

                    # Sauvegarde de la transition dans la mémoire DQN (avec l'action brute 0-14)
                    self.memory.push(
                        state,
                        raw_action,
                        reward_norm,
                        next_state,
                        float(done),
                        float(next_num_candidates),
                    )

                    # Si le buffer contient assez d'expériences, on effectue un pas d'optimisation (Gradient Descent)
                    if len(self.memory) >= self.batch_size:
                        loss = self._update_model()
                        if loss is not None:
                            local_losses.append(loss)
                else:
                    zero_candidate_steps += 1

                state = next_state

                if done:
                    state = env.reset()

                # Stratégie de secours : si la simulation se vide prématurément, relancer un court warmup
                if len(env.current_candidates) <= 0 and useful_steps < min_useful_steps:
                    state, _ = self._warmup_until_candidates(env, max_warmup_steps=30)

            # Agrégation des métriques locales à renvoyer au serveur Flower
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
                f"reward={metrics['reward']:.3f} | assigned={metrics['assigned']:.1f} | "
                f"loss={metrics['loss']:.4f} | useful_steps={useful_steps} | "
                f"sim_steps={total_sim_steps} | use_gui={self.use_gui}"
            )

            num_examples = max(1, int(useful_steps))
            return self.get_parameters(config={}), num_examples, metrics

        finally:
            # Garantie de fermeture propre du processus SUMO même en cas de crash dans le bloc try
            try:
                env.close()
            except Exception:
                pass
            gc.collect()  # Déchargement final de la mémoire en fin de round

    def evaluate(self, parameters, config):
        """Flower API : Évaluation des performances du modèle global actuel sur un environnement local témoin."""
        self.set_parameters(parameters)

        env = self._build_env()
        eval_steps = int(config.get("eval_steps", 200))
        warmup_steps = int(config.get("warmup_steps", 150))
        max_total_sim_steps = int(config.get("max_total_sim_steps", eval_steps * 4))

        total_reward = 0.0
        total_assigned = 0.0
        useful_eval_steps = 0
        total_sim_steps = 0
        zero_candidate_steps = 0

        try:
            env.start()
            state, found_candidates = self._warmup_until_candidates(env, max_warmup_steps=warmup_steps)

            if not found_candidates:
                print(f"[{self.client_id}/{self.agent_name}] eval | aucun candidat trouvé après warmup")

            while useful_eval_steps < eval_steps and total_sim_steps < max_total_sim_steps:
                num_candidates = len(env.current_candidates)

                # Gestion des pas vides pendant l'évaluation (Pas d'apprentissage ici)
                if num_candidates <= 0:
                    try:
                        neutral_payload = {"parking_index": 0, "price_level": "standard"}
                        next_state, _, done, _ = env.step(neutral_payload)
                    except Exception:
                        break
                    zero_candidate_steps += 1
                    total_sim_steps += 1
                    state = next_state

                    if done:
                        state = env.reset()
                    continue

                # Exploitation pure (Epsilon = 0) : Choix de la meilleure Q-value masquée
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_values = self.model(state_t).squeeze(0)
                    masked_q = q_values.clone()

                    # Masquage strict en mode évaluation (Inférence)
                    for tier in range(3):
                        start_invalid = (tier * self.top_k) + num_candidates
                        end_invalid = (tier + 1) * self.top_k
                        masked_q[start_invalid:end_invalid] = -1e9

                    raw_action = int(masked_q.argmax().item())

                # Décodage de l'action pour l'évaluation
                parking_index = raw_action % self.top_k
                price_tier = raw_action // self.top_k
                price_level = ["discount", "standard", "premium"][price_tier]

                action_payload = {
                    "parking_index": parking_index,
                    "price_level": price_level
                }

                try:
                    next_state, reward, done, info = env.step(action_payload)
                except Exception:
                    break
                    
                total_reward += float(reward)
                total_assigned += float(info.get("assigned", 0))
                useful_eval_steps += 1
                total_sim_steps += 1
                state = next_state

                if done:
                    state = env.reset()

            loss_proxy = float(-total_reward)  # Flower attend une métrique de perte (plus elle est basse, mieux c'est)
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
                f"reward={metrics['reward']:.3f} | assigned={metrics['assigned']:.1f} | "
                f"useful_steps={useful_eval_steps} | sim_steps={total_sim_steps} | use_gui={self.use_gui}"
            )

            return loss_proxy, num_examples, metrics

        finally:
            try:
                env.close()
            except Exception:
                pass
            gc.collect()

    def _update_model(self):
        """Méthode cœur du Double DQN : Échantillonne le buffer et met à jour les poids par rétropropagation."""
        try:
            batch = self.memory.sample(self.batch_size)
        except Exception:
            return None

        # Nettoyage et vérification du format des données extraites du Replay Buffer
        clean_batch = []
        for item in batch:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                clean_batch.append(item[:6])

        if len(clean_batch) < self.batch_size:
            return None

        # Unzipping du lot de transitions (désassemblage en tuples distincts)
        states, actions, rewards, next_states, dones, next_num_candidates = zip(*clean_batch)

        # Conversion globale des listes de données vers des Tenseurs PyTorch sur l'appareil (CPU/GPU) cible
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        next_num_candidates = torch.tensor(np.array(next_num_candidates), dtype=torch.long, device=self.device)

        # Récupération des Q-values de l'action effectivement prise avec le modèle courant
        curr_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Évaluation des états futurs sur les deux modèles (Principe du Double DQN pour éviter la surestimation)
            online_next_q = self.model(next_states)
            target_next_q = self.target_model(next_states)

            masked_online_next_q = online_next_q.clone()
            masked_target_next_q = target_next_q.clone()

            # Ajustement du calcul des Q-values cibles pour le DDQN multi-action étendu
            # Application individuelle du masque d'invalidation sur chaque élément du batch
            for i in range(self.batch_size):
                n = int(next_num_candidates[i].item())
                if n <= 0:
                    masked_online_next_q[i, :] = -1e9
                    masked_target_next_q[i, :] = -1e9
                elif n < self.top_k:
                    for tier in range(3):
                        start_invalid = (tier * self.top_k) + n
                        end_invalid = (tier + 1) * self.top_k
                        masked_online_next_q[i, start_invalid:end_invalid] = -1e9
                        masked_target_next_q[i, start_invalid:end_invalid] = -1e9

            # Sélection de la meilleure action future via le réseau ONLINE
            next_actions = masked_online_next_q.argmax(dim=1, keepdim=True)
            # Évaluation de la valeur de cette action future via le réseau TARGET
            next_q = masked_target_next_q.gather(1, next_actions).squeeze(1)
            # Sécurité : si aucun candidat n'est présent au pas suivant, la Q-value future attendue vaut 0
            next_q = torch.where(next_num_candidates > 0, next_q, torch.zeros_like(next_q))

            # Calcul de la valeur cible de Bellman (Target Q)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Calcul de la perte via Smooth L1 (Huber Loss) résistant aux valeurs aberrantes
        loss = F.smooth_l1_loss(curr_q, target_q)

        # Rétropropagation du gradient et mise à jour des poids du modèle en ligne
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0) # Évite l'explosion du gradient
        self.optimizer.step()

        # Gestion de la mise à jour périodique par blocs (Hard Update) du réseau cible (Target)
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())