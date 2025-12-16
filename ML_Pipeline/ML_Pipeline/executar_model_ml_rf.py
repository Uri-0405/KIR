import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACIÓ FINAL
# =============================================================================
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = 'Taula_imputs_ml.xlsx'

OUTPUT_DIR = 'Results_Finals_RF'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DIES_TALL_CENSURA = 730
N_FOLDS = 5
RANDOM_SEED = 42
MIN_FREQ_ALELS = 0.05

print("-" * 60)
print(">>> EXECUCIÓ FINAL: OPTIMITZACIÓ RANDOM FOREST (GRID SEARCH)")
print("-" * 60)

# =============================================================================
# 1. PREPARACIÓ DADES (BIO + NETEJA + IMPUTACIÓ)
# =============================================================================
def preparar_dades_completes(df, cols_gens, min_freq=0.05):
    # Variables Bio
    activadors = []
    inhibidors = []
    n_act = np.zeros(len(df))
    n_inh = np.zeros(len(df))
    for col in cols_gens:
        present = df[col].notna().astype(int)
        if 'S' in col: n_act += present
        elif 'L' in col: n_inh += present
    df['BIO_Total_Activadors'] = n_act
    df['BIO_Total_Inhibidors'] = n_inh
    df['BIO_Ratio'] = n_act / (n_inh + 0.1)

    # Separar Al·lels
    n_pacients = len(df)
    llindar = int(n_pacients * min_freq)
    df_base = df.drop(columns=cols_gens)
    dfs = [df_base]
    
    for col in cols_gens:
        all_alleles = set()
        s_clean = df[col].dropna().astype(str)
        for item in s_clean:
            for p in item.split('/'):
                if p.strip() and p.strip().lower() != 'nan': all_alleles.add(p.strip())
        
        gen_dict = {}
        for allele in all_alleles:
            mask = df[col].astype(str).apply(lambda x: 1 if allele in x else 0)
            if mask.sum() >= llindar:
                gen_dict[f"{col}_{allele}"] = mask
        if gen_dict: dfs.append(pd.DataFrame(gen_dict))
    
    return pd.concat(dfs, axis=1)

# Càrrega i Neteja
try:
    if INPUT_FILE.endswith('.csv'): df = pd.read_csv(INPUT_FILE)
    else: df = pd.read_excel(INPUT_FILE, engine='openpyxl')
except: sys.exit(f"Error llegint {INPUT_FILE}")

cols_no = ['MOSTRA', 'MATCH_ID', 'ESTAT_DETALLAT', 'ML_TARGET_EXIT (0/1)', 
           'DIES_EVENT', 'EDAT_DX', 'SEXE', 'BCR_ABL_INICIAL', 'TIPUS_TRANSCRIT']
cols_gens = [c for c in df.columns if c not in cols_no]

df_ml = preparar_dades_completes(df, cols_gens, MIN_FREQ_ALELS)

df_ml = df_ml.dropna(subset=['ML_TARGET_EXIT (0/1)'])
df_ml['DIES_EVENT'] = pd.to_numeric(df_ml['DIES_EVENT'], errors='coerce')
mask = ((df_ml['ML_TARGET_EXIT (0/1)']==0) & 
        (df_ml['ESTAT_DETALLAT'].astype(str).str.contains('ACTIU', case=False, na=False)) & 
        (df_ml['DIES_EVENT'] < DIES_TALL_CENSURA))
df_final = df_ml[~mask].copy()

# X, y
y = df_final['ML_TARGET_EXIT (0/1)'].astype(int)
drops = ['MOSTRA', 'MATCH_ID', 'ESTAT_DETALLAT', 'ML_TARGET_EXIT (0/1)', 'DIES_EVENT']
X = df_final.drop(columns=[c for c in drops if c in df_final.columns])
X = pd.get_dummies(X, drop_first=True)

# Imputació + Escalat
imputer = SimpleImputer(strategy='median') 
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

print(f" -> Dataset final: {X_scaled.shape[0]} pacients, {X_scaled.shape[1]} variables.")

# =============================================================================
# 2. GRID SEARCH
# =============================================================================
print("\n>>> CERCANT ELS MILLORS HIPERPARÀMETRES...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [2, 4, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=RANDOM_SEED)

grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)

grid_search.fit(X_scaled, y)

best_model = grid_search.best_estimator_
print(f"\n✅ MILLOR CONFIGURACIÓ TROBADA:")
print(f"   {grid_search.best_params_}")
print(f"   Best Cross-Val AUC Score: {grid_search.best_score_:.3f}")

# Guardar params a txt
with open(os.path.join(OUTPUT_DIR, 'best_params.txt'), 'w') as f:
    f.write(str(grid_search.best_params_))

# =============================================================================
# 3. AVALUACIÓ FINAL + SHAP (CORREGIT)
# =============================================================================
print("\n>>> GENERANT RESULTATS FINALS AMB EL MODEL OPTIMITZAT...")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
acc_scores = []
auc_scores = []
shap_values_list = []
X_test_list = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train_f, X_test_f = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]
    
    # Re-entrenem el millor model en cada fold per tenir mètriques honestes
    model_fold =  grid_search.best_estimator_
    model_fold.fit(X_train_f, y_train_f)
    
    y_pred = model_fold.predict(X_test_f)
    y_prob = model_fold.predict_proba(X_test_f)[:, 1]
    
    acc_scores.append(accuracy_score(y_test_f, y_pred))
    auc_scores.append(roc_auc_score(y_test_f, y_prob))

    # --- CORRECCIÓ SHAP ROBUSTA ---
    explainer = shap.TreeExplainer(model_fold)
    shap_vals = explainer.shap_values(X_test_f, check_additivity=False)
    
    # Gestió de diferents formats de sortida de SHAP (Llista vs 3D Array)
    if isinstance(shap_vals, list):
        # Format antic o RF sklearn: [Array_Class0, Array_Class1]
        shap_for_plot = shap_vals[1]
    elif len(np.array(shap_vals).shape) == 3:
        # Format nou o XGBoost natiu: (n_samples, n_features, n_classes)
        shap_for_plot = shap_vals[:, :, 1]
    else:
        # Binary simple: (n_samples, n_features)
        shap_for_plot = shap_vals
        
    shap_values_list.append(shap_for_plot)
    X_test_list.append(X_test_f)


print("-" * 60)
print(f"📊 RESULTAT FINAL (RF OPTIMITZAT):")
print(f"Accuracy Mitjana: {np.mean(acc_scores):.3f}")
print(f"AUC-ROC Mitjana: {np.mean(auc_scores):.3f}")
print(f"Desviació Accuracy: {np.std(acc_scores):.3f}")
print(f"Desviació AUC-ROC: {np.std(auc_scores):.3f}")
print("-" * 60)

# PLOT SHAP
all_shap = np.concatenate(shap_values_list, axis=0)
all_X = pd.concat(X_test_list, axis=0)

plt.figure(figsize=(12, 10))
shap.summary_plot(all_shap, all_X, show=False, max_display=20)
plt.title("Drivers Genètics de la Curació (Model Final RF)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'FINAL_SHAP_RF.png'), dpi=300)

# EXCEL IMPORTÀNCIA (CORREGIT)
# Assegurem que vals és 1D
vals = np.abs(all_shap).mean(0)
if len(vals.shape) > 1: vals = vals.flatten() # Seguretat extra

imps = pd.DataFrame(list(zip(all_X.columns, vals)), columns=['Feature','Importance'])
imps.sort_values(by='Importance', ascending=False, inplace=True)
imps.to_excel(os.path.join(OUTPUT_DIR, 'Importancia_Variables_RF.xlsx'), index=False)

print(f"✅ Tot guardat a la carpeta '{OUTPUT_DIR}'.")