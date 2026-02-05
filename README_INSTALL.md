# Projet Credit Risk Analysis - Groupe NEOMA

Ce dossier contient l'ensemble du code source, des modèles et du dashboard pour l'analyse du risque de crédit.

## 📂 Contenu du dossier

*   `analysis.py` : Script principal d'analyse de données, entraînement des modèles (XGBoost, Logistic Regression, MLP) et génération des graphiques.
*   `dashboard.py` : Application interactive (Streamlit) pour visualiser les résultats et simuler des demandes de prêt.
*   `REPORT.html` : Le rapport complet du projet (format Web, avec graphiques).
*   `data/` : Dossier contenant le jeu de données `credit_risk_dataset(in).csv`.
*   `requirements.txt` : Liste des bibliothèques Python nécessaires.
*   `run_dashboard.bat` : Script pour lancer le dashboard en un clic (Windows).
*   `output/` : Dossier contenant les graphiques générés par l'analyse.

## 🚀 Installation & Lancement

### 1. Pré-requis
Assurez-vous d'avoir **Python** installé sur votre machine.

### 2. Installation des dépendances
Ouvrez un terminal dans ce dossier et exécutez la commande suivante pour installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

### 3. Lancer le Dashboard (Recommandé)
Double-cliquez simplement sur le fichier **`run_dashboard.bat`**.
Cela ouvrira automatiquement votre navigateur avec l'interface interactive.

### 4. Lancer l'Analyse complète 
Si vous souhaitez régénérer tous les modèles et les graphiques statiques, lancez le script d'analyse :

```bash
python analysis.py
```

## 📊 Fonctionnalités du Dashboard
*   **Overview** : Statistiques globales sur le portefeuille.
*   **EDA** : Exploration interactives des variables (Distributions, Corrélations).
*   **Model Performance** : Comparaison des modèles (ROC Curves, Matrices de confusion).
*   **Risk Simulator** : Outil de simulation temps réel pour estimer la probabilité de défaut d'un nouveau client.

---

