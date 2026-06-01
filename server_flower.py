#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Projet MADINA - Smart Parking Management System
Script Serveur Flower (Apprentissage Fédéré) - Version Finale Corrigée Pipeline
"""

import os
import flwr as fl
import torch
from typing import Dict, List, Optional, Tuple

# Configuration du nombre de rounds et définition du chemin de persistance du modèle global
NUM_ROUNDS = 1 
CHECKPOINT_PATH = "checkpoints/madina_global_model.pth"

# =========================================================================
# 🌟 1. STRATÉGIE D'AGRÉGATION ET DE SAUVEGARDE AUTOMATIQUE SUR LE DISQUE
# =========================================================================
class MadinaSaveStrategy(fl.server.strategy.FedAvg):
    """
    Extension personnalisée de la stratégie FedAvg (Federated Averaging).
    Intercepte l'étape d'agrégation des poids pour sauvegarder le modèle global sur le disque.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        # Appel de l'algorithme FedAvg classique (calcul de la moyenne pondérée par le nombre d'exemples locaux)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Enregistrement systématique des poids du modèle si le round est validé (pas d'échec critique d'agrégation)
        if aggregated_parameters is not None:
            print(f"\n💾 [ROUND {server_round}] Agrégation réussie. Enregistrement du point de contrôle...")
            # Dé-sérialisation des paramètres Flower en tableaux NumPy (ndarrays)
            weights_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Création sécurisée du dossier de stockage des points de contrôle
            os.makedirs("checkpoints", exist_ok=True)
            
            # Sauvegarde continue du dernier modèle à jour (écrase le précédent)
            torch.save(weights_ndarrays, CHECKPOINT_PATH)
            # Sauvegarde d'historique propre au round pour le suivi de la convergence
            torch.save(weights_ndarrays, f"checkpoints/madina_model_round_{server_round}.pth")
            print(f"✅ Modèle global synchronisé et sauvegardé sous : {CHECKPOINT_PATH}\n")
            
        return aggregated_parameters, aggregated_metrics


def fit_config(server_round: int):
    """
    Génère le dictionnaire de configuration envoyé aux clients Flower avant l'entraînement local.
    Intègre notamment le calendrier de décroissance synchrone de l'exploration epsilon.
    """
    # Formule géométrique décroissante pour Epsilon (exploration forte au début, exploitation à la fin)
    epsilon = max(0.02, 0.20 * (0.80 ** (server_round - 1)))
    
    config = {
        "server_round": int(server_round),
        "epsilon": float(epsilon),              # Taux d'exploration Epsilon-Greedy transmis à l'agent DQN local
        "batch_size": 64,                       # Taille des lots pour l'optimisation par descente de gradient
        "gamma": 0.95,                          # Facteur d'atténuation des récompenses futures (Discount Factor)
        "steps_per_round": 1500,      
        "eval_steps": 200,                      # Nombre de pas de temps alloués à la validation locale
        "warmup_steps": 150,                    # Phase transitoire d'injection du trafic SUMO avant apprentissage
        "max_total_sim_steps": 2500,            # Garde-fou anti-blocage (sécurité de fin d'épisode dans SUMO)
    }
    print(f"--- Round {server_round} : epsilon={epsilon:.3f} | steps={config['steps_per_round']} ---")
    return config


def evaluate_config(server_round: int):
    """Génère le dictionnaire de configuration envoyé aux clients Flower avant l'évaluation locale."""
    return {
        "server_round": int(server_round),
        "eval_steps": 200,
        "warmup_steps": 150,
    }


def aggregate_fit_metrics(metrics):
    """
    Fonction de rappel (callback) pour agréger les métriques d'entraînement remontées par les clients.
    """
    if not metrics:
        return {}
    total_clients = len(metrics)
    
    # Extraction et moyennage des indicateurs de performance de chaque client actif
    avg_reward = sum(m.get("reward", 0.0) for _, m in metrics) / total_clients
    avg_assigned = sum(m.get("assigned", 0.0) for _, m in metrics) / total_clients
    avg_loss = sum(m.get("loss", 0.0) for _, m in metrics) / total_clients
    avg_buffer = sum(m.get("buffer_size", 0.0) for _, m in metrics) / total_clients
    avg_steps = sum(m.get("effective_steps", 0.0) for _, m in metrics) / total_clients
    
    print(
        f"FIT AGGREGATED | reward={avg_reward:.3f} | assigned={avg_assigned:.3f} | "
        f"loss={avg_loss:.4f} | buffer={avg_buffer:.1f} | steps={avg_steps:.1f}"
    )
    return {
        "reward": float(avg_reward),
        "assigned": float(avg_assigned),
        "loss": float(avg_loss),
        "buffer_size": float(avg_buffer),
        "effective_steps": float(avg_steps),
    }


def aggregate_evaluate_metrics(metrics):
    """
    Fonction de rappel (callback) pour agréger les métriques d'évaluation (validation) des clients.
    """
    if not metrics:
        return {}
    total_clients = len(metrics)
    
    # Calcul des moyennes pour la phase d'inférence pure (sans exploration)
    avg_reward = sum(m.get("reward", 0.0) for _, m in metrics) / total_clients
    avg_assigned = sum(m.get("assigned", 0.0) for _, m in metrics) / total_clients
    
    print(f"EVAL AGGREGATED | reward={avg_reward:.3f} | assigned={avg_assigned:.3f}")
    return {
        "reward": float(avg_reward),
        "assigned": float(avg_assigned),
    }


if __name__ == "__main__":
    print("     DÉMARRAGE DU SERVEUR FLOWER - PROJET MADINA (D&M SMART PARK)   ")
    print(f"  NUM_ROUNDS     = {NUM_ROUNDS}")
    print(f"  steps_per_round = 1500")
    
    # Tentative de récupération et de chargement des acquis d'exécutions antérieures (Warm-start)
    initial_parameters = None
    if os.path.exists(CHECKPOINT_PATH):
        print(f"✨ Alignement historique détecté : {CHECKPOINT_PATH}")
        try:
            # 🔥 SÉCURISATION DU PIPELINE : Ajout de weights_only=False pour autoriser les structures DQN/NumPy
            loaded_weights = torch.load(CHECKPOINT_PATH, weights_only=False)
            # Conversion des structures ndarrays en paramètres Flower sérialisés
            initial_parameters = fl.common.ndarrays_to_parameters(loaded_weights)
            print("✅ Modèle global réinitialisé avec succès pour le nouveau scénario.")
        except Exception as e:
            print(f"❌ Impossible de charger les poids existants ({e}). Démarrage de zéro.")
    else:
        print("🚀 Aucun point de contrôle antérieur détecté. Initialisation de zéro.")

    # Instanciation de la stratégie personnalisée avec configuration des seuils de synchronisation
    strategy = MadinaSaveStrategy(
        fraction_fit=1.0,               # Sollicite 100% des clients disponibles pour l'entraînement
        fraction_evaluate=1.0,          # Sollicite 100% des clients disponibles pour la validation
        min_fit_clients=4,              # Nombre minimal de clients requis pour valider une étape d'entraînement
        min_evaluate_clients=4,         # Nombre minimal de clients requis pour valider une étape d'évaluation
        min_available_clients=4,        # Attend que 4 clients soient connectés avant de lancer le Round 1
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        initial_parameters=initial_parameters,  # Chargement persistant du modèle rechargé
    )
    
    # Lancement effectif du serveur RPC Flower à l'adresse locale spécifiée
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )