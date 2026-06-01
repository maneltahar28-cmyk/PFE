import os
import sys

import flwr as fl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.client_flower import ParkingFlowerClient


def build_env_config():
    use_gui = os.getenv("SMARTPARKING_USE_GUI", "0").strip() == "1"

    # Capture dynamique des variables du pipeline ou valeurs par défaut pour tests unitaires
    sumo_begin_time = int(os.getenv("MADINA_BEGIN_TIME", 36000))
    max_steps = int(os.getenv("MADINA_MAX_STEPS", 3000))
    parking_demand_prob = float(os.getenv("MADINA_DEMAND_PROB", 0.4))

    parkings_xml = os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "parkings_min300.add.xml")
    if not os.path.exists(parkings_xml):
        parkings_xml = os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "parkings.add.xml")

    return {
        "sumo_cfg": os.path.join(PROJECT_ROOT, "scenarios", "luxembourg", "dua.static.sumocfg"),
        "parkings_xml": parkings_xml,
        "agents_json": os.path.join(PROJECT_ROOT, "config", "agents_weighted_kmeans_balanced.json"),
        "top_k": 5,
        "max_steps": max_steps,
        "sumo_begin_time": sumo_begin_time,
        "agent_detection_radius": 1500.0,
        "use_gui": use_gui,
        "warmup_steps": 150,
        "gui_delay": 0.03,
        "parking_demand_prob": parking_demand_prob,
        "max_requests_per_step": 150,
        "max_queue_size": 2000,
    }

def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m training.run_fl_client <agent_name>")

    agent_name = sys.argv[1]
    env_config = build_env_config()

    client = ParkingFlowerClient(
        env_config=env_config,
        client_id=agent_name,
        agent_name=agent_name,
    )

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()