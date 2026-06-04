"""
Projet MADINA - Smart Parking Management System
Script d'Évaluation Automatique Multi-Scénarios (S1 à S5)
Méthodologie : Transfer Learning / Inférence pure (Epsilon = 0.0)
"""
import os
import sys
import time
import pandas as pd
import subprocess

# Définition des 5 scénarios opérationnels critiques validés théoriquement
SCENARIOS = {
    "S1": {
        "name": "Trafic normal (10h-12h)",
        "begin": "36000",
        "steps": "7200",
        "scale": "1.0",
        "desc": "Établir une situation de référence"
    },
    "S2": {
        "name": "Heure de pointe matinale (7h-9h)",
        "begin": "25200",
        "steps": "7200",
        "scale": "1.0",
        "desc": "Tester le système en situation de saturation"
    },
    "S3": {
        "name": "Événement exceptionnel (+40%)",
        "begin": "50400",
        "steps": "10800",
        "scale": "1.4",
        "desc": "Évaluer la résilience du système"
    },
    "S4": {
        "name": "Période creuse de nuit (21h-23h)",
        "begin": "75600",
        "steps": "7200",
        "scale": "1.0",
        "desc": "Étudier le comportement en sous-utilisation"
    },
    "S5": {
        "name": "Journée complète (6h-22h)",
        "begin": "21600",
        "steps": "57600",
        "scale": "1.0",
        "desc": "Tester la robustesse globale du système"
    }
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "outputs", "analysis")
FINAL_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "outputs", "scenarios_comparative_matrix.md")

def clean_previous_outputs():
    """Nettoie les logs du run précédent pour éviter les contaminations de données."""
    target_files = ["summary_overall.csv", "summary_by_mode.csv", "summary_by_agent.csv"]
    for f in target_files:
        path = os.path.join(OUTPUT_ANALYSIS_DIR, f)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def run_single_scenario(scen_id, config):
    """Exécute un scénario unique en injectant les hyperparamètres dans run_full_system."""
    print(f"\n==============================================================")
    print(f" [MADINA EVAL] Lancement du Scénario {scen_id} : {config['name']}")
    print(f" Configuration : Début={config['begin']}s | Pas={config['steps']}s | Échelle={config['scale']}")
    print(f"==============================================================")
    
    # Sécurité RAM : Libération des instances SUMO antérieures
    if sys.platform == "win32":
        subprocess.run("taskkill /F /IM sumo.exe 2>NUL", shell=True)
    
    clean_previous_outputs()
    
    # Préparation des variables d'environnement lues par multi_agent_env.py
    env_vars = os.environ.copy()
    env_vars["MADINA_EVAL_MODE"] = "True"
    env_vars["MADINA_BEGIN"] = config["begin"]
    env_vars["MADINA_STEPS"] = config["steps"]
    env_vars["MADINA_SCALE"] = config["scale"]
    
    # ✅ FIX DE CHEMIN : Exécution sous forme de module pour éviter l'erreur d'ouverture de fichier
    cmd = [sys.executable, "-m", "training.run_full_system"]
    
    start_time = time.time()
    process = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env_vars)
    execution_duration = time.time() - start_time
    
    print(f" [Scénario {scen_id}] Exécuté en {execution_duration:.1f} secondes avec code de retour {process.returncode}")
    
    # Extraction des indicateurs générés par ton analyseur de résultats
    summary_path = os.path.join(OUTPUT_ANALYSIS_DIR, "summary_overall.csv")
    if os.path.exists(summary_path):
        try:
            df = pd.read_csv(summary_path)
            kpis = dict(zip(df["metric"], df["value"]))
            kpis["exec_status"] = "✅ Succès"
            return kpis
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture des KPIs du scénario {scen_id} : {e}")
            
    return {"exec_status": "❌ Échec"}

def generate_comparative_report(results):
    """Génère la matrice comparative finale demandée dans le terminal et le fichier Markdown."""
    
    # 1. Construction du fichier Markdown (avec formatage pour ton manuscrit de thèse)
    md_content = (
        "# 📊 Projet MADINA - Matrice Comparative Multi-Scénarios\n\n"
        "Ce rapport compile les performances du modèle centralisé unifié soumis aux différents stress-tests.\n\n"
        "| Indicateurs de Performance Urbaine | S1 : Trafic Normal <br>*(10h - 12h)* | S2 : Heure de Pointe <br>*(7h - 9h)* | S3 : Événement (+40%) <br>*(14h - 17h)* | S4 : Période Creuse <br>*(21h - 23h)* | S5 : Journée Complète <br>*(6h - 22h)* |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )
    
    # Alignement strict sur la liste exacte des colonnes et lignes demandées
    metrics_mapping = [
        ("Nombre total de décisions", "total_decisions", "{:.0f}"),
        ("Taux de succès global", "overall_success_rate_percent", "{:.2f} %"),
        ("Distance moyenne de marche", "avg_distance_m", "{:.2f} m"),
        ("Prix moyen proposé par l'IA", "avg_price", "{:.2f} €"),
        ("Taux d'occupation réel moyen", "avg_occupancy_rate_percent", "{:.2f} %"),
        ("Taux d'occupation prédit moyen", "avg_predicted_occupancy_rate_percent", "{:.2f} %")
    ]
    
    for label, key, fmt in metrics_mapping:
        row = f"| **{label}** "
        for scen_id in ["S1", "S2", "S3", "S4", "S5"]:
            val = results.get(scen_id, {}).get(key, "N/A")
            if isinstance(val, (int, float)):
                row += f"| {fmt.format(val)} "
            else:
                row += f"| {val} "
        row += "|\n"
        md_content += row

    # 2. Affichage direct dans le terminal de commande
    print("\n" + "="*110)
    print("📈 MATRICE COMPARATIVE MULTI-SCÉNARIOS DE ROBUSTESSE OPERATIONNELLE (MADINA)")
    print("="*110)
    # Nettoyage visuel des sauts de ligne HTML pour la console shell
    console_display = md_content.replace("<br>*(", " (").replace(")*", ")")
    print(console_display)
    print("="*110)
    
    # 3. Sauvegarde physique du livrable de thèse
    os.makedirs(os.path.dirname(FINAL_SUMMARY_FILE), exist_ok=True)
    with open(FINAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"📂 Matrice de thèse sauvegardée avec succès dans : {FINAL_SUMMARY_FILE}\n")

def main():
    # Liste des graines pour prouver la reproductibilité (Standard littérature)
    SEEDS = [42, 101, 2026]
    all_results = {"S5": {}}
    
    # Agrégateur pour stocker les résultats de chaque seed
    seed_results = []
    
    print(f"\n🚀 Lancement de l'évaluation Multi-Seeds pour S5...")
    
    for seed in SEEDS:
        print(f"\n🎲 Simulation avec Seed: {seed}")
        # Injection de la seed dans l'environnement système
        os.environ["MADINA_SEED"] = str(seed)
        
        # Exécution du scénario S5
        res = run_single_scenario("S5", SCENARIOS["S5"])
        seed_results.append(res)
    
    # Calcul de la moyenne des KPIs pour les seeds
    final_kpis = {}
    metrics = ["total_decisions", "overall_success_rate_percent", "avg_distance_m", 
               "avg_price", "avg_occupancy_rate_percent", "avg_predicted_occupancy_rate_percent"]
    
    for m in metrics:
        values = [r.get(m, 0) for r in seed_results if isinstance(r.get(m), (int, float))]
        if values:
            final_kpis[m] = sum(values) / len(values)
            
    all_results["S5"] = final_kpis
        
    generate_comparative_report(all_results)
        
if __name__ == "__main__":
    main()