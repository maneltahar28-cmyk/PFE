import flwr as fl

NUM_ROUNDS = 4   

def fit_config(server_round: int):
    # Schedule epsilon plus agressif
    epsilon = max(0.05, 0.9 * (0.80 ** (server_round - 1)))
    config = {
        "server_round": int(server_round),
        "epsilon": float(epsilon),
        "batch_size": 64,
        "gamma": 0.9,
        "steps_per_round": 800,      
        "eval_steps": 120,
        "warmup_steps": 100,
        "max_total_sim_steps": 1000,
    }
    print(f"--- Round {server_round} : epsilon={epsilon:.3f} | steps={config['steps_per_round']} ---")
    return config

def evaluate_config(server_round: int):
    return {
        "server_round": int(server_round),
        "eval_steps": 120,
        "warmup_steps": 300,
    }

def aggregate_fit_metrics(metrics):
    if not metrics:
        return {}
    total_clients = len(metrics)
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
    if not metrics:
        return {}
    total_clients = len(metrics)
    avg_reward = sum(m.get("reward", 0.0) for _, m in metrics) / total_clients
    avg_assigned = sum(m.get("assigned", 0.0) for _, m in metrics) / total_clients
    print(f"EVAL AGGREGATED | reward={avg_reward:.3f} | assigned={avg_assigned:.3f}")
    return {
        "reward": float(avg_reward),
        "assigned": float(avg_assigned),
    }

if __name__ == "__main__":
    print("Démarrage du Serveur Flower (Smart Parking Multi-Agent)...")
    print(f"  NUM_ROUNDS     = {NUM_ROUNDS}")
    print(f"  steps_per_round = 800")
    print(f"  epsilon r1/r5/r10 ≈ 0.85 / 0.20 / 0.07")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )