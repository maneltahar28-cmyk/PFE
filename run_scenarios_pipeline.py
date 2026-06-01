#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Projet MADINA - Smart Parking Management System
Pipeline d'évaluation automatisé épuré (Affichage exclusif des rounds).
Filtre les terminaux en arrière-plan pour éviter l'inondation visuelle.
"""

import os
import sys
import time
import subprocess
import signal
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

# =========================================================
# DEFINITION DES 5 SCÉNARIOS OPÉRATIONNELS (TESTS COURTS)
# =========================================================
SCENARIOS = {
    "S1": {
        "name": "Trafic normal (Baseline)",
        "begin_time": 36000,       # 10h00 du matin
        "duration": 3000,          # Test court de 3000 pas
        "demand_prob": 0.35,       
        "extra_flow": False
    },
    "S2": {
        "name": "Heure de pointe matinale (Saturation)",
        "begin_time": 25200,       # 07h00 du matin
        "duration": 3000,          
        "demand_prob": 0.60,       
        "extra_flow": False
    },
    "S3": {
        "name": "Événement exceptionnel (Résilience)",
        "begin_time": 50400,       # 14h00
        "duration": 3000,          
        "demand_prob": 0.65,       
        "extra_flow": True         
    },
    "S4": {
        "name": "Nuit creuse (Sous-utilisation)",
        "begin_time": 82800,       # 23h00
        "duration": 3000,          
        "demand_prob": 0.20,       
        "extra_flow": False
    },
    "S5": {
        "name": "Journée complète (Robustesse)",
        "begin_time": 21600,       # 06h00 du matin
        "duration": 3000,          
        "demand_prob": 0.40,       
        "extra_flow": True
    }
}

def clean_previous_outputs():
    """Supprime l'ancien journal pour éviter la pollution des données inter-scénarios."""
    decisions_path = os.path.join(PROJECT_ROOT, "outputs", "decisions_log.csv")
    if os.path.exists(decisions_path):
        try:
            os.remove(decisions_path)
        except Exception:
            pass

def stop_process(proc):
    """Arrête proprement un sous-processus d'arrière-plan."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        time.sleep(1)
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass

def print_new_server_rounds(server_log, last_pos=0):
    """Parcourt le fichier log du serveur Flower et n'affiche QUE les lignes de round."""
    if not os.path.exists(server_log):
        return last_pos
    try:
        with open(server_log, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(last_pos)
            new_data = f.read()
            last_pos = f.tell()
        for line in new_data.splitlines():
            # Filtre strict : laisse passer uniquement les indicateurs de rounds Flower
            if "[ROUND" in line:
                print(f" 🟢 {line.strip()}")
    except Exception:
        pass
    return last_pos

def execute_environment_run(scen_id, config):
    """Démarre le serveur et les clients en masquant leur affichage brut."""
    logs_dir = os.path.join(PROJECT_ROOT, "outputs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Injection des paramètres temporels du scénario courant
    os.environ["MADINA_BEGIN_TIME"] = str(config["begin_time"])
    os.environ["MADINA_MAX_STEPS"] = str(config["duration"])
    os.environ["MADINA_DEMAND_PROB"] = str(config["demand_prob"])
    os.environ["MADINA_USE_EXTRA_FLOW"] = "1" if config["extra_flow"] else "0"

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

    # 1. Démarrage silencieux du serveur Flower
    server_log = os.path.join(logs_dir, f"server_flower_{scen_id}.log")
    server_f = open(server_log, "w", encoding="utf-8")
    server_script = os.path.join(PROJECT_ROOT, "training", "server_flower.py")
    
    server_proc = subprocess.Popen(
        [PYTHON, server_script], cwd=PROJECT_ROOT,
        stdout=server_f, stderr=subprocess.STDOUT, creationflags=creationflags
    )
    time.sleep(4)  # Attente d'ouverture sécurisée du port gRPC 8080

    # 2. Démarrage silencieux des 4 clients territoriaux
    client_script = os.path.join(PROJECT_ROOT, "training", "run_fl_client.py")
    client_processes = []
    client_files = []
    
    for i in range(1, 5):
        client_log = os.path.join(logs_dir, f"agent_{i}_{scen_id}.log")
        cf = open(client_log, "w", encoding="utf-8")
        client_files.append(cf)
        
        p = subprocess.Popen(
            [PYTHON, client_script, f"agent_{i}"], cwd=PROJECT_ROOT,
            stdout=cf, stderr=subprocess.STDOUT, creationflags=creationflags
        )
        client_processes.append(p)

    # 3. Boucle d'écoute temps réel filtrée
    server_log_pos = 0
    while True:
        time.sleep(2)
        server_log_pos = print_new_server_rounds(server_log, server_log_pos)
        
        # Rupture si le serveur Flower a terminé ses 10 rounds
        if server_proc.poll() is not None:
            break
            
        # Rupture de sécurité si toute la flotte de clients a crashé
        if all(cp.poll() is not None for cp in client_processes):
            print("⚠️ [PIPELINE] Tous les clients se sont arrêtés prématurément.")
            break

    # Nettoyage et fermeture propre des descripteurs de fichiers logs
    for p in client_processes:
        stop_process(p)
    stop_process(server_proc)
    server_f.close()
    for cf in client_files:
        cf.close()

def collect_metrics_from_analysis():
    """Extrait proprement les indicateurs de performance post-simulation."""
    summary_path = os.path.join(PROJECT_ROOT, "outputs", "analysis", "summary_overall.csv")
    if not os.path.exists(summary_path):
        return {
            "Total Décisions": 0, "Taux de Succès": "nan%", "Distance Moyenne": "0.0 m",
            "Prix Moyen": "0.00 €", "Reward Moyenne": "0.00", "Occupation Urbaine": "0.00%"
        }
    df_metrics = pd.read_csv(summary_path)
    metrics_dict = dict(zip(df_metrics["metric"], df_metrics["value"]))
    return {
        "Total Décisions": int(metrics_dict.get("total_decisions", 0)),
        "Taux de Succès": f"{metrics_dict.get('overall_success_rate_percent', 0):.2f}%",
        "Distance Moyenne": f"{metrics_dict.get('avg_distance_m', 0):.1f} m",
        "Prix Moyen": f"{metrics_dict.get('avg_price', 0):.2f} €",
        "Reward Moyenne": f"{metrics_dict.get('avg_reward', 0):.2f}",
        "Occupation Urbaine": f"{metrics_dict.get('avg_occupancy_rate_percent', 0):.2f}%"
    }

def main():
    print("==========================================================================")
    print("👑 [PIPELINE ÉPURÉ MADINA] Évaluation Multi-Scénarios Spatiaux")
    print("==========================================================================")
    results_pipeline = {}

    for scen_id, config in SCENARIOS.items():
        print(f"\n🚀 >>> EXÉCUTION DU SCÉNARIO {scen_id} : {config['name']} <<<")
        clean_previous_outputs()
        
        # Lancement de la simulation filtrée
        execute_environment_run(scen_id, config)
        
        # Traitement analytique en mode silencieux
        analysis_script = os.path.join(PROJECT_ROOT, "tools", "analyze_decisions.py")
        subprocess.run([PYTHON, analysis_script], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Collecte des données consolidées
        results_pipeline[scen_id] = collect_metrics_from_analysis()

    # Compilation finale du rapport technique global
    df_report = pd.DataFrame.from_dict(results_pipeline, orient="index")
    df_report.insert(0, "Description", [SCENARIOS[sid]["name"] for sid in df_report.index])
    
    report_csv = os.path.join(PROJECT_ROOT, "outputs", "analysis", "madina_scenarios_report.csv")
    df_report.to_csv(report_csv, encoding="utf-8")

    print("\n\n==========================================================================")
    print("📊 COMPILATION DU RAPPORT TECHNIQUE FINAL (MADINA EVALUATION)")
    print("==========================================================================\n")
    print(df_report.to_string())
    print("\n==========================================================================")
    print(f"✅ Fichier de synthèse consolidé enregistré : {report_csv}")
    print("==========================================================================")

if __name__ == "__main__":
    main()