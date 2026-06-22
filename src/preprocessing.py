"""
src/preprocessing.py

Fonctions de nettoyage et d'encodage du dataset Diabetes 130-US hospitals.
Toute fonction ici doit pouvoir s'appliquer identiquement à de nouvelles
données (un futur patient à l'API) — pas de logique propre au split train/test,
qui reste dans src/train.py.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Regroupement des codes ICD-9 (diag_1, diag_2, diag_3)
# Basé sur Strack et al. (2014), 9 catégories cliniques.
# ============================================================
def group_icd9(code):
    if pd.isna(code):
        return 'Unknown'
    code = str(code)
    if code.startswith('250'):
        return 'Diabetes'
    try:
        c = float(code)
        if 390 <= c <= 459 or c == 785: return 'Circulatory'
        if 460 <= c <= 519 or c == 786: return 'Respiratory'
        if 520 <= c <= 579 or c == 787: return 'Digestive'
        if 800 <= c <= 999:             return 'Injury'
        if 710 <= c <= 739:             return 'Musculoskeletal'
        if 580 <= c <= 629 or c == 788: return 'Genitourinary'
        if 140 <= c <= 239:             return 'Neoplasms'
    except ValueError:
        pass
    return 'Other'


# ============================================================
# Couche 1 — Nettoyage déterministe de la cohorte
# Règles fixes, indépendantes de toute statistique apprise sur les données.
# Applicable identiquement à une seule nouvelle observation.
# ============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Valeurs manquantes encodées "?" plutôt que NaN
    df.replace('?', np.nan, inplace=True)

    # Colonnes trop incomplètes (>80% manquants) : suppression
    df.drop(columns=['weight', 'max_glu_serum', 'A1Cresult'], inplace=True)

    # medical_specialty / payer_code : catégorie "Unknown" plutôt que suppression
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    df['payer_code'] = df['payer_code'].fillna('Unknown')

    # race et diag_1/2/3 : faible taux de manquants, suppression des lignes
    df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3'], inplace=True)

    # Exclusion des séjours décès/hospice (IDs_mapping.csv UCI) :
    # la réadmission n'a pas de sens clinique pour ces cas
    death_hospice_codes = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(death_hospice_codes)]

    # Ces trois variables sont des codes catégoriels, pas des quantités numériques
    id_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    for col in id_cols:
        df[col] = df[col].astype(str)

    # Regroupement des codes ICD-9 en 9 catégories cliniques
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[f'{col}_cat'] = df[col].apply(group_icd9)
    df.drop(columns=['diag_1', 'diag_2', 'diag_3'], inplace=True)

    # Binarisation de la cible : <30 jours = 1, NO/>30 jours = 0
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    df.drop(columns=['readmitted'], inplace=True)

    return df


# ============================================================
# Identification des colonnes catégorielles / numériques
# ============================================================
def get_cat_num_columns(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()
    return cat_cols, num_cols


# ============================================================
# Suppression des colonnes à variance nulle
# Détection sur le train uniquement, suppression identique sur train et test.
# ============================================================
def drop_zero_variance(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list):
    zero_variance_cols = [col for col in cat_cols if X_train[col].nunique() == 1]

    X_train = X_train.drop(columns=zero_variance_cols)
    X_test = X_test.drop(columns=zero_variance_cols)
    cat_cols = [c for c in cat_cols if c not in zero_variance_cols]

    return X_train, X_test, cat_cols, zero_variance_cols


# ============================================================
# Couche 2 — Regroupement des catégories rares (<1% du train)
# Seuil calculé sur le train uniquement, appliqué tel quel au test
# (pas de fuite d'information).
# ============================================================
def group_rare_categories(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           high_card_cols: list, threshold: float = 0.01):
    X_train = X_train.copy()
    X_test = X_test.copy()
    threshold_count = threshold * len(X_train)

    for col in high_card_cols:
        freq = X_train[col].value_counts()
        rare_categories = freq[freq < threshold_count].index
        X_train[col] = X_train[col].replace(rare_categories, 'Other')
        X_test[col] = X_test[col].replace(rare_categories, 'Other')

    return X_train, X_test


# ============================================================
# Couche 2 — One-Hot Encoding
# fit sur le train uniquement, transform sur train et test.
# drop='first' : évite le dummy variable trap (colinéarité par construction).
# handle_unknown='ignore' : robustesse si une catégorie du test est inconnue du train.
# L'encodeur est sauvegardé pour être rechargé à l'identique à l'inférence (J11, API).
# ============================================================
def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                     cat_cols: list, num_cols: list, save_path: str = None):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    encoder.fit(X_train[cat_cols])

    X_train_cat = pd.DataFrame(
        encoder.transform(X_train[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )
    X_test_cat = pd.DataFrame(
        encoder.transform(X_test[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index
    )

    X_train_encoded = pd.concat([X_train[num_cols], X_train_cat], axis=1)
    X_test_encoded = pd.concat([X_test[num_cols], X_test_cat], axis=1)

    if save_path:
        joblib.dump(encoder, save_path)

    return X_train_encoded, X_test_encoded, encoder