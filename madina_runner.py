import subprocess
import sys
import pandas as pd
import os

class MadinaSystemRunner:
    def __init__(self, project_root):
        self.project_root = project_root

    def run(self):
        # 1. Préparation de l'environnement avec les variables injectées (S5 par défaut)
        env = os.environ.copy()
        env["MADINA_EVAL_MODE"] = "True"
        env["MADINA_BEGIN"] = os.environ.get("MADINA_BEGIN", "21600")
        env["MADINA_STEPS"] = os.environ.get("MADINA_STEPS", "57600")
        
        # 2. Lancement de votre système complet avec les variables d'environnement
        cmd = [sys.executable, "-m", "training.run_full_system"]
        
        print(f"[MADINA SYSTEM] Lancement avec BEGIN={env['MADINA_BEGIN']} et STEPS={env['MADINA_STEPS']}")
        subprocess.run(cmd, cwd=self.project_root, check=True, env=env)

        # 3. Après exécution, lecture des résultats
        summary_path = os.path.join(self.project_root, "outputs", "analysis", "summary_overall.csv")
        
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Le fichier de résultats {summary_path} n'a pas été généré.")
            
        df = pd.read_csv(summary_path)
        
        # 4. Conversion en dictionnaire
        return dict(zip(df["metric"], df["value"]))