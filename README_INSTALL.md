# Projet Credit Risk Analysis - Groupe NEOMA

Ce dossier contient une solution Hybride (R + Python) pour l'analyse du risque de crédit.

## 📂 Contenu du dossier

*   `analysis.R` : **NOUVEAU** - Script R principal d'analyse de données.
*   `dashboard.py` : Application interactive (Streamlit) pour visualiser les résultats et simuler des demandes de prêt.
*   `REPORT.html` : Le rapport complet du projet.
*   `data/` : Dossier contenant le jeu de données `credit_risk_dataset(in).csv`.
*   `install_packages.R` : Script pour installer les librairies R nécessaires.
*   `requirements.txt` : Liste des bibliothèques Python nécessaires pour le dashboard.
*   `run_analysis.bat` : Script pour lancer l'analyse R (Génération des graphiques).
*   `run_dashboard.bat` : Script pour lancer le dashboard interactif.
*   `output/` : Dossier contenant les graphiques générés par `analysis.R`.

## 🚀 Installation & Lancement

### 1. Pré-requis
*   **R** doit être installé.
*   **Python** doit être installé.

### 2. Installation des dépendances

**Pour R (Analyse) :**
Double-cliquez sur `install_packages.R` ou lancez dans un terminal R :
```r
source("install_packages.R")
```

**Pour Python (Dashboard) :**
Ouvrez un terminal et installez les dépendances :
```bash
pip install -r requirements.txt
```

### 3. Exécution du Worklow

**Étape A : Lancer l'Analyse (R)**
Double-cliquez sur **`run_analysis.bat`**.
Cela va :
1.  Charger les données.
2.  Générer tous les graphiques d'analyse (Distributions, Corrélations, ROC Curves).
3.  Sauvegarder les résultats dans le dossier `output/plots`.

**Étape B : Lancer le Dashboard (Python)**
Double-cliquez sur **`run_dashboard.bat`**.
Cela ouvrira votre navigateur avec l'interface interactive pour explorer les données et simuler des prédictions.

## 📊 Fonctionnalités
*   **Analyse R** : Traitement statistique robuste, modélisation (XGBoost, Neural Net, GLM), et génération de graphiques de publication.
*   **Dashboard Python** : Exploration interactive et simulateur de risque temps réel.

---

