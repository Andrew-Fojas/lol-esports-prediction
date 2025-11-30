"""Configuration settings for the LoL Esports Prediction project."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = RAW_DATA_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
TEAM_DATA_FILE = PROCESSED_DATA_DIR / "team_data_2024.csv"
COMPLETE_TEAM_DATA_FILE = PROCESSED_DATA_DIR / "complete_team_data_2024.csv"
TEAM_METRICS_FILE = PROCESSED_DATA_DIR / "team_metrics_data_2024.csv"
PCA_TRANSFORMED_FILE = PROCESSED_DATA_DIR / "pca_transformed_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

FEATURE_COLUMNS = [
    # Metadata
    'gameid', 'league', 'year', 'split', 'playoffs', 'date', 'patch', 'teamname',

    # Basic game info
    'side', 'result', 'gamelength',

    # Teamfight metrics
    'teamkills', 'teamdeaths', 'team kpm', 'ckpm',

    # Objectives - Dragons & Elder
    'firstdragon', 'dragons', 'opp_dragons', 'elders', 'opp_elders',

    # Objectives - Heralds
    'firstherald', 'heralds', 'opp_heralds',

    # Objectives - Void Grubs
    'void_grubs', 'opp_void_grubs',

    # Objectives - Baron
    'firstbaron', 'barons', 'opp_barons',

    # Objectives - Towers
    'turretplates', 'opp_turretplates', 'firsttower', 'firstmidtower',
    'firsttothreetowers', 'towers', 'opp_towers',

    # Objectives - Inhibitors
    'inhibitors', 'opp_inhibitors',

    # Damage metrics
    'damagetochampions', 'dpm', 'damagetakenperminute', 'damagemitigatedperminute',

    # Economy
    'totalgold', 'goldspent', 'gspd', 'gpr',
    'goldat10', 'goldat15', 'goldat20', 'goldat25',
    'golddiffat10', 'golddiffat15', 'golddiffat20', 'golddiffat25',

    # Vision control
    'wardsplaced', 'wardskilled', 'visionscore', 'controlwardsbought', 'wpm', 'wcpm', 'vspm',

    # Time-based stats
    'killsat10', 'killsat15', 'killsat20', 'killsat25',
    'deathsat10', 'deathsat15', 'deathsat20', 'deathsat25'
]

EXCLUDE_COLUMNS = [
    'gameid', 'league', 'year', 'split', 'playoffs',
    'date', 'patch', 'teamname', 'side'
]

N_PCA_COMPONENTS = 9
PCA_COMPONENT_NAMES = {
    'PC1': 'Gold_Advantage_and_Towers',
    'PC2': 'Vision_Control',
    'PC3': 'Combat_Metrics',
    'PC4': 'Grubs_and_Herald',
    'PC5': 'Herald_Control',
    'PC6': 'Herald_and_Grubs',
    'PC7': 'Laning_Phase_and_Early_Baron_Control',
    'PC8': 'Early_Macro_and_Objective_Priority',
    'PC9': 'Elder_Drake_Control'
}

MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "lol_esports_prediction"
