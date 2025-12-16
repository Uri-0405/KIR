import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
CUTOFF = 88
print(f">>> CERCA D'ESTABILITAT (Regularització) - Cutoff {CUTOFF}")

# 1. PREPARAR DADES (Versió simplificada in-script per rapidesa)
# -----------------------------------------------------------------------------
# Detectar fitxer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(SCRIPT_DIR, "../Results_Processed/Taula_imputs_ml.xlsx")
if not os.path.exists(RAW_DATA): RAW_DATA = "Taula_imputs_ml.xlsx" # Fallback

df = pd.read_excel(RAW_DATA)

# Filtre Qualitat (88%)
cols_gens = [c for c in df.columns if 'ML_TARGET' not in c and 'DIES' not in c and 'MOSTRA' not in c and 'EDAT' not in c]
n_pacients = len(df)
min_present = int(n_pacients * 0.88)
valid_cols = []
for c in cols_gens:
    if df[c].notna().sum() >= min_present:
        valid_cols.append(c)

# Expandir al·lels (només els freqüents > 5% per reduir soroll extra)
dfs = []
cols_fixes = ['MOSTRA', 'ML_TARGET_EXIT (0/1)', 'EDAT_DX', 'SEXE', 'BCR_ABL_INICIAL', 'TIPUS_TRANSCRIT']
dfs.append(df[[c for c in cols_fixes if c in df.columns]])

print(f" -> Loci originals vàlids: {len(valid_cols)}")

for c in valid_cols:
    s = df[c].dropna().astype(str)
    alleles = set()
    for row in s:
        for a in row.split('/'):
            if a.strip() and a.strip().lower() != 'nan': alleles.add(a.strip())
    
    for al in alleles:
        # Freqüència més estricta (5%) per evitar overfitting
        mask = df[c].astype(str).apply(lambda x: 1 if al in str(x) else 0)
        if mask.sum() >= (n_pacients * 0.05):
            dfs.append(mask.rename(f"{c}_{al}"))

df_clean = pd.concat(dfs, axis=1)
df_clean = df_clean.dropna(subset=['ML_TARGET_EXIT (0/1)'])

# X, y
y = df_clean['ML_TARGET_EXIT (0/1)'].astype(int)
X = df_clean.drop(columns=['MOSTRA', 'ML_TARGET_EXIT (0/1)'], errors='ignore')
for c in X.select_dtypes(include=['object']).columns:
    X[c] = X[c].astype('category').cat.codes

print(f" -> Matriu Final: {X.shape[0]} pacients x {X.shape[1]} variables")

# =============================================================================
# 2. BUCLE DE REGULARITZACIÓ
# =============================================================================
# Provem diferents nivells de "duresa" per evitar memorització
configs = [
    {'name': 'Base', 'depth': 3, 'reg': 0},
    {'name': 'Soft Reg', 'depth': 2, 'reg': 1},     # L1 Regularization petita
    {'name': 'Medium Reg', 'depth': 2, 'reg': 5},   # L1 Regularization mitjana
    {'name': 'Hard Reg', 'depth': 2, 'reg': 10},    # L1 Regularization forta
    {'name': 'Ultra Reg', 'depth': 1, 'reg': 5},    # Stump (arbres d'1 decisió)
]

print("\nResultats (Mitjana 5-Fold CV):")
print(f"{'CONFIG':<12} | {'TRAIN':<6} | {'TEST':<6} | {'GAP':<6} | {'VEREDICTE'}")
print("-" * 60)

for conf in configs:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_scores, te_scores = [], []
    
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        # XGBoost amb paràmetres de control
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=conf['depth'],
            learning_rate=0.05,
            reg_alpha=conf['reg'],      # L1 Regularization (clau per feature selection)
            reg_lambda=conf['reg'],     # L2 Regularization
            subsample=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_tr, y_tr)
        
        tr_scores.append(roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1]))
        te_scores.append(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        
    avg_tr = np.mean(tr_scores)
    avg_te = np.mean(te_scores)
    gap = avg_tr - avg_te
    
    veredicte = "✅ ESTABLE" if gap < 0.10 else "⚠️ OVERFIT"
    if avg_te < 0.55: veredicte = "❌ POBRE"
    
    print(f"{conf['name']:<12} | {avg_tr:.3f}  | {avg_te:.3f}  | {gap:.3f}  | {veredicte}")

print("-" * 60)