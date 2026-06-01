#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Projet MADINA - Smart Parking Management System
Script de pré-traitement Mixte : Fusion (Routes originales + Extra Flows)
Version optimisée pour i7 / 8 Go de RAM - Chemins mis à jour pour l'utilisateur 'HP'
"""

import xml.etree.ElementTree as ET
import os
import random
import gc

def generate_mixed_madina_subscribers(pourcentage_abonnes=0.30):
    # --- CORRECTION DES CHEMINS AVEC VOTRE COMPTE USER REAL 'HP' ---
    route_files = [
        r"C:\Users\user\Desktop\projet luxemburg\scenarios\luxembourg\DUARoutes\local.0.rou.xml",
        r"C:\Users\user\Desktop\projet luxemburg\scenarios\luxembourg\DUARoutes\local.1.rou.xml",
        r"C:\Users\user\Desktop\projet luxemburg\scenarios\luxembourg\DUARoutes\local.2.rou.xml"
    ]
    # Fichier de trafic additionnel (extra flow pour les parkings secondaires)
    extra_flow_file = r"C:\Users\user\Desktop\projet luxemburg\scenarios\luxembourg\extra_flow.rou.xml"
    
    output_xml_path = r"C:\Users\user\Desktop\projet luxemburg\scenarios\luxembourg\madina_abonnes.xml"
    
    vehicules_reels = set()
    vehicules_extra = set()

    print("==========================================================")
    print("🔍 [PRE-PROCESSING MIXTE MADINA] Fusion des flux de trafic...")
    print("==========================================================")

    # 1. Extraction frugale depuis les fichiers de routes réelles (local.*.rou.xml)
    for file_path in route_files:
        if os.path.exists(file_path):
            print(f"📖 Lecture de : {os.path.basename(file_path)} ...")
            try:
                # Utilisation d'iterparse pour traiter le fichier ligne par ligne sans saturer les 8 Go de RAM
                context = ET.iterparse(file_path, events=("start", "end"))
                for event, elem in context:
                    if event == "end":
                        if elem.tag == 'vehicle' or elem.tag == 'trip':
                            v_id = elem.get('id')
                            if v_id: 
                                vehicules_reels.add(v_id)
                        elem.clear()  # Libération instantanée de la mémoire RAM
            except Exception as e:
                print(f"⚠️ Erreur lors du parsing de {os.path.basename(file_path)} : {e}")
        else:
            print(f"❌ Fichier introuvable : {file_path}")

    print(f"🚗 Véhicules réels uniques extraits : {len(vehicules_reels)}")

    # 2. Extraction et extrapolation depuis extra_flow.rou.xml
    if os.path.exists(extra_flow_file):
        print(f"📖 Lecture de extra_flow : {os.path.basename(extra_flow_file)} ...")
        try:
            tree = ET.parse(extra_flow_file)
            root = tree.getroot()
            for flow in root.findall('flow'):
                flow_id = flow.get('id')
                vehs_per_hour = float(flow.get('vehsPerHour', 0))
                begin = float(flow.get('begin', 0))
                end = float(flow.get('end', 0))
                
                # Calcul du nombre de véhicules générés par ce flow dans SUMO
                duration_hours = (end - begin) / 3600.0
                total_vehs_generated = int(vehs_per_hour * duration_hours)
                
                # Recréation des ID générés par SUMO (format: flow_id.index)
                for idx in range(total_vehs_generated):
                    generated_id = f"{flow_id}.{idx}"
                    vehicules_extra.add(generated_id)
        except Exception as e:
            print(f"⚠️ Erreur lors du parsing de extra_flow : {e}")
    else:
        print(f"❌ Fichier extra_flow introuvable : {extra_flow_file}")

    print(f"⚡ Véhicules extra_flow extrapolés : {len(vehicules_extra)}")

    # 3. Fusion des deux ensembles dans la grande flotte de la ville
    flotte_globale_fusionnee = list(vehicules_reels.union(vehicules_extra))
    total_flotte = len(flotte_globale_fusionnee)

    if total_flotte == 0:
        print("❌ Erreur critique : Aucun véhicule détecté dans les fichiers sources.")
        return

    # 4. Échantillonnage des 30% d'abonnés (Mélange parfait garanti par random.sample)
    random.seed(42)  # Seed fixe pour la cohérence multi-agents Flower
    num_abonnes = int(total_flotte * pourcentage_abonnes)
    abonnes_mixtes = random.sample(flotte_globale_fusionnee, num_abonnes)

    # 5. Comptage des proportions dans notre échantillon final pour votre analyse
    nb_reels_choisis = sum(1 for v in abonnes_mixtes if not v.startswith("flow_rl"))
    nb_extra_choisis = len(abonnes_mixtes) - nb_reels_choisis

    # 6. Écriture du fichier XML madina_abonnes.xml
    root_node = ET.Element("subscribers", company="MADINA", total_city_vehicles=str(total_flotte))
    for v_id in abonnes_mixtes:
        ET.SubElement(root_node, "subscriber", id=v_id, type="monthly_pass")

    tree_output = ET.ElementTree(root_node)
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    
    try:
        if hasattr(ET, "indent"):
            ET.indent(tree_output, space="    ", level=0)
            
        tree_output.write(output_xml_path, encoding="utf-8", xml_declaration=True)
        
        print("\n==========================================================")
        print("✅ BASE D'ABONNÉS MADINA MIXTE GÉNÉRÉE AVEC SUCCÈS")
        print(f"📂 Chemin : {output_xml_path}")
        print(f"📊 Volume total de la flotte fusionnée : {total_flotte} véhicules")
        print(f"💳 Nombre total d'abonnés actifs (30%) : {len(abonnes_mixtes)}")
        print(f"    🔹 Dont abonnés issus du trafic réel : {nb_reels_choisis} ({round(nb_reels_choisis/len(abonnes_mixtes)*100, 1)}%)")
        print(f"    🔹 Dont abonnés issus de extra_flow  : {nb_extra_choisis} ({round(nb_extra_choisis/len(abonnes_mixtes)*100, 1)}%)")
        print("==========================================================")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'écriture du fichier XML : {e}")

    # Nettoyage final de la RAM
    del vehicules_reels
    del vehicules_extra
    del flotte_globale_fusionnee
    gc.collect()

if __name__ == "__main__":
    generate_mixed_madina_subscribers()