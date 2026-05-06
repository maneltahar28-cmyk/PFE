import json
import os
import signal
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run_blocking(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Commande échouée : {' '.join(cmd)}")


def run_background(cmd, log_path=None, extra_env=None):
    print(f"[START] {' '.join(cmd)}")

    stdout = None
    stderr = None
    log_file = None

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        stdout = log_file
        stderr = subprocess.STDOUT

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
        env=env,
    )
    return proc, log_file


def load_agents(agents_json):
    if not os.path.exists(agents_json):
        raise FileNotFoundError(f"Fichier introuvable : {agents_json}")

    with open(agents_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "agents" not in data:
        raise ValueError("Le fichier de configuration des agents ne contient pas la clé 'agents'.")

    agent_names = sorted(
        list(data["agents"].keys()),
        key=lambda name: int(name.split("_")[1])
    )

    expected_agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
    if agent_names != expected_agents:
        raise ValueError(
            f"Configuration agents invalide. "
            f"Attendu: {expected_agents} | Trouvé: {agent_names}"
        )

    return agent_names


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            time.sleep(2)
        else:
            proc.terminate()
            time.sleep(2)
    except Exception:
        pass

    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def clean_old_outputs():
    targets = [
        os.path.join(PROJECT_ROOT, "outputs", "decisions_log.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "analysis", "summary_by_mode.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "analysis", "summary_by_agent.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "analysis", "summary_by_parking.csv"),
        os.path.join(PROJECT_ROOT, "outputs", "analysis", "summary_overall.csv"),
    ]

    for path in targets:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"[CLEAN] supprimé : {path}")
            except Exception as e:
                print(f"[WARN] impossible de supprimer {path} : {e}")


def print_new_server_rounds(server_log, last_pos=0):
    if not os.path.exists(server_log):
        return last_pos

    try:
        with open(server_log, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(last_pos)
            new_data = f.read()
            last_pos = f.tell()

        for line in new_data.splitlines():
            if "[ROUND" in line:
                print(line)
    except Exception:
        pass

    return last_pos


def main():
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs", "logs")
    os.makedirs(outputs_dir, exist_ok=True)

    agents_json = os.path.join(PROJECT_ROOT, "config", "agents_weighted_kmeans_balanced.json")

    all_processes = []
    log_files = []
    client_processes = []
    server_proc = None

    try:
        print("\n🚀 === LANCEMENT DU SYSTÈME COMPLET FINAL ===")

        print("\n[0] Nettoyage des anciens outputs...")
        clean_old_outputs()

        print("\n[1] Génération / vérification des agents...")
        run_blocking([PYTHON, "-m", "tools.build_weighted_agent_clusters"])

        agent_names = load_agents(agents_json)
        print(f"\n✅ Agents détectés : {agent_names}")

        print("\n[2] Vérification configuration finale...")
        print(" - nombre d'agents attendu : 4")
        print(" - noms attendus          : agent_1, agent_2, agent_3, agent_4")
        print(" - top_k recommandé       : 5")
        print(" - clustering chargé      : agents_weighted_kmeans_balanced.json")
        print(" - GUI                    : désactivée pour tous les agents")
        print(" - mode SUMO              : sans interface graphique")

        print("\n[3] Démarrage du serveur Flower...")
        server_log = os.path.join(outputs_dir, "server_flower.log")
        server_proc, server_log_file = run_background(
            [PYTHON, "-m", "training.server_flower"],
            log_path=server_log,
        )
        all_processes.append(server_proc)
        log_files.append(server_log_file)

        time.sleep(5)

        print("\n[4] Démarrage des 4 clients Flower...")
        for agent_name in agent_names:
            client_log = os.path.join(outputs_dir, f"{agent_name}.log")

            extra_env = {
                "SMARTPARKING_USE_GUI": "0",
                "SMARTPARKING_GUI_AGENT": "",
            }

            proc, log_file = run_background(
                [PYTHON, "-m", "training.run_fl_client", agent_name],
                log_path=client_log,
                extra_env=extra_env,
            )
            all_processes.append(proc)
            client_processes.append((proc, agent_name))
            log_files.append(log_file)
            time.sleep(2)

        print("\n✅ Système complet lancé.")
        print("\nLogs disponibles :")
        print(f" - serveur : {server_log}")
        for agent_name in agent_names:
            print(f" - client {agent_name} : outputs/logs/{agent_name}.log")

        print("\nAucune fenêtre SUMO-GUI ne doit s'ouvrir.")
        print("Tous les agents tournent en mode sumo sans interface.")
        print("\nLe script s'arrêtera automatiquement à la fin des rounds.\n")

        already_reported_clients = set()
        server_log_pos = 0

        while True:
            time.sleep(3)

            server_log_pos = print_new_server_rounds(server_log, server_log_pos)

            server_code = server_proc.poll()
            if server_code is not None:
                print(f"\n✅ Serveur Flower terminé avec code {server_code}")
                print("Fin normale de l'expérience.")
                break

            for proc, agent_name in client_processes:
                code = proc.poll()
                if code is not None and agent_name not in already_reported_clients:
                    if code == 0:
                        print(f"✅ Client {agent_name} terminé normalement avec code 0")
                    else:
                        print(f"⚠️ Client {agent_name} arrêté avec erreur code {code}")
                    already_reported_clients.add(agent_name)

    except KeyboardInterrupt:
        print("\n⛔ Arrêt demandé par l'utilisateur")

    except Exception as e:
        print(f"\n❌ Erreur dans run_full_system : {e}")

    finally:
        print("\n🧹 Arrêt des processus...")

        for proc in reversed(all_processes):
            stop_process(proc)

        for f in log_files:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass

        print("✅ Tous les processus ont été arrêtés.")


if __name__ == "__main__":
    main()