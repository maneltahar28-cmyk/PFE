import os
import flwr as fl
import torch
from typing import Dict, List, Optional, Tuple

# Configuration du nombre de rounds et définition du chemin de persistance du modèle global
NUM_ROUNDS = 1
CHECKPOINT_PATH = "checkpoints/madina_global_model.pth"

# 🌟 1. STRATÉGIE D'AGRÉGATION ET DE SAUVEGARDE AUTOMATIQUE SUR LE DISQUE
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
            print(f"💾 [ROUND {server_round}] Agrégation réussie. Enregistrement du point de contrôle...")
            
            # Conversion des paramètres Flower (octets) en tenseurs PyTorch
            tensors = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Création du dossier des points de contrôle s'il n'existe pas
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            
            # Sauvegarde physique du dictionnaire d'état des réseaux de neurones MADINA
            torch.save(tensors, CHECKPOINT_PATH)
            print(f"✅ Modèle global synchronisé et sauvegardé sous : {CHECKPOINT_PATH}\n")
            
        return aggregated_parameters, aggregated_metrics

# ⚙️ 2. CONFIGURATION DYNAMIQUE DES PARAMÈTRES D'ENTRAÎNEMENT LOCAUX
def fit_config(server_round: int):
    """
    Génère le dictionnaire de configuration envoyé aux clients Flower avant l'entraînement local.
    Intègre notamment le calendrier de décroissance synchrone de l'exploration epsilon.
    """
    # Formule géométrique décroissante pour Epsilon (exploration forte au début, exploitation à la fin)
    # Reste bloqué à un plancher minimum de sécurité de 2% (0.02) pour maintenir une exploration résiduelle
    epsilon = max(0.02, 0.20 * (0.80 ** (server_round - 1)))
    
    # --- 📊 INTERCEPTION DU SCRIPT AUTOMATIQUE MULTI-SCÉNARIOS (S1 à S5) ---
    # Si le protocole d'évaluation est actif, on écrase epsilon à 0 (Exploitation Pure)
    # et on aligne dynamiquement le garde-fou de pas SUMO sur la durée du scénario courant
    if os.environ.get("MADINA_EVAL_MODE") == "True":
        epsilon = 0.0
        max_sim_steps = int(os.environ.get("MADINA_STEPS", 2500))
    else:
        max_sim_steps = 2500 # Garde-fou nominal de sécurité en phase d'entraînement classique
    
    config = {
        "server_round": int(server_round),
        "epsilon": float(epsilon),              # Taux d'exploration Epsilon-Greedy transmis à l'agent DQN local
        "batch_size": 64,                       # Taille des lots pour l'optimisation par descente de gradient
        "gamma": 0.95,                           # Facteur d'atténuation des récompenses futures (Discount Factor)
        "steps_per_round": 1500,      
        "eval_steps": 200,                      # Nombre de pas de temps alloués à la validation locale
        "warmup_steps": 150,                    # Phase transitoire d'injection du trafic SUMO avant apprentissage
        "max_total_sim_steps": max_sim_steps,   # Ajustement dynamique ou nominal du temps d'épisode
    }
    print(f"--- Round {server_round} : epsilon={epsilon:.3f} | steps={config['steps_per_round']} | max_sim_steps={config['max_total_sim_steps']} ---")
    return config


def evaluate_config(server_round: int):
    """Génère la configuration spécifique pour la phase de validation locale des clients."""
    return {
        "server_round": int(server_round),
        "batch_size": 64,
        "gamma": 0.95,
    }

# 📊 3. FONCTIONS DE CENTRALISATION ET D'AGRÉGATION DES MÉTRIQUES URBAINES
def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    """Calcule la moyenne pondérée des indicateurs de performance d'entraînement collectés."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # Initialisation des accumulateurs de KPIs macroéconomiques
    agg_reward = 0.0
    agg_loss = 0.0
    agg_assigned = 0.0
    
    for num_examples, m in metrics:
        agg_reward += m.get("reward", 0.0) * num_examples
        agg_loss += m.get("loss", 0.0) * num_examples
        agg_assigned += m.get("assigned", 0.0) * num_examples
        
    return {
        "reward": agg_reward / total_examples,
        "loss": agg_loss / total_examples,
        "assigned": agg_assigned / total_examples,
    }


def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    """Calcule la moyenne pondérée des indicateurs de performance de validation collectés."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    agg_reward = 0.0
    agg_assigned = 0.0
    
    for num_examples, m in metrics:
        agg_reward += m.get("reward", 0.0) * num_examples
        agg_assigned += m.get("assigned", 0.0) * num_examples
        
    print(f"\nEVAL AGGREGATED | reward={agg_reward / total_examples:.3f} | assigned={agg_assigned / total_examples:.3f}")
    return {
        "reward": agg_reward / total_examples,
        "assigned": agg_assigned / total_examples,
    }

# 🚀 4. POINT D'ENTRÉE DU SERVEUR DE CONVERGENCE FÉDÉRÉE
def main():
    print("\n" + "="*70)
    print("🧠 [MADINA CENTRAL SERVER] Initialisation du Serveur Fédéré Flower")
    print("="*70)
    
    initial_parameters = None
    
    # Stratégie de rechargement à chaud : Reprise automatique des poids existants s'ils sont sur le disque
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📦 Point de contrôle détecté sous {CHECKPOINT_PATH}. Chargement en mémoire vive...")
        try:
            tensors = torch.load(CHECKPOINT_PATH)
            initial_parameters = fl.common.ndarrays_to_parameters(tensors)
            print("✅ Poids du modèle global Duel DDQN unifiés chargés avec succès. Reprise de l'apprentissage.")
        except Exception as e:
            print(f"❌ Impossible de charger les poids existants ({e}). Démarrage de zéro.")
    else:
        print("🚀 Aucun point de contrôle antérieur détecté. Initialisation à zéro.")

    # Instanciation de la stratégie personnalisée avec configuration des seuils de synchronisation
    strategy = MadinaSaveStrategy(
        fraction_fit=1.0,               # Sollicite 100% des clients disponibles pour l'entraînement (les 4 agents)
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

if __name__ == "__main__":
    main()