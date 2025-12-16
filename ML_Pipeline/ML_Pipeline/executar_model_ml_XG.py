import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
import os

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
# Gestió d'arguments
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = 'Taula_imputs_ml.xlsx'

OUTPUT_DIR = 'Results_Processed'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DIES_TALL_CENSURA = 730  # 2 anys
N_FOLDS = 5
RANDOM_SEED = 42
MIN_FREQ_ALELS = 0.05    # LLINDAR CLAU: 5% (Elimina al·lels molt rars)

print("-" * 60)
print(">>> INICIANT ANÀLISI GENÈTICA (XGBOOST + SHAP + FILTRE SOROLL)")
print("-" * 60)

# =============================================================================
# 1. FUNCIÓ OPTIMITZADA AMB FILTRE DE FREQÜÈNCIA
# =============================================================================
def processar_genetica_correctament(df, cols_gens, min_freq=0.05):
    """
    1. Separa al·lels (ex: *001/*002 -> Columna A i Columna B).
    2. ELIMINA variables que apareixen en menys del 'min_freq' dels pacients.
    3. Optimitzat per evitar PerformanceWarning.
    """
    print(f" -> Processant {len(cols_gens)} gens complexos...")
    
    n_pacients_total = len(df)
    llindar_pacients = int(n_pacients_total * min_freq)
    print(f" -> Criteri de Neteja: S'eliminaran al·lels presents en menys de {llindar_pacients} pacients (<{min_freq*100}%).")
    
    # Separem la base (clínica) de la genètica
    df_base = df.drop(columns=cols_gens)
    new_dataframes = [df_base]
    
    total_creats = 0
    total_eliminats = 0
    
    for col in cols_gens:
        # 1. Identificar tots els al·lels únics en aquesta columna
        all_alleles = set()
        series_clean = df[col].dropna().astype(str)
        
        for item in series_clean:
            parts = item.split('/')
            for p in parts:
                p_clean = p.strip()
                if p_clean and p_clean.lower() != 'nan': 
                    all_alleles.add(p_clean)
        
        if not all_alleles: continue
            
        # 2. Avaluar quins al·lels superen el tall
        gen_dict = {}
        for allele in all_alleles:
            # Creem la màscara (Qui té aquest al·lel?)
            # Vectoritzat: més ràpid que fer bucles
            mask = df[col].astype(str).apply(lambda x: 1 if allele in x else 0)
            
            # 3. FILTRE DE FREQÜÈNCIA
            if mask.sum() >= llindar_pacients:
                new_col_name = f"{col}_{allele}"
                gen_dict[new_col_name] = mask
                total_creats += 1
            else:
                total_eliminats += 1
        
        # Afegim només les columnes bones
        if gen_dict:
            new_dataframes.append(pd.DataFrame(gen_dict))
            
    # Unim tot de cop (molt ràpid)
    df_processed = pd.concat(new_dataframes, axis=1)
    
    print(f" -> Resultat Neteja: {total_creats} variables útils conservades. {total_eliminats} variables rares eliminades (soroll).")
    return df_processed

# =============================================================================
# 2. CÀRREGA DE DADES
# =============================================================================
try:
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
except Exception as e:
    sys.exit(f"❌ Error llegint el fitxer {INPUT_FILE}: {e}")

# Definim columnes NO genètiques
cols_no_gens = [
    'MOSTRA', 'MATCH_ID', 'ESTAT_DETALLAT', 'ML_TARGET_EXIT (0/1)', 
    'DIES_EVENT', 'EDAT_DX', 'SEXE', 'BCR_ABL_INICIAL', 'TIPUS_TRANSCRIT'
]

# Identifiquem gens automàticament
cols_gens = [c for c in df.columns if c not in cols_no_gens]

# APLIQUEM LA CORRECCIÓ + FILTRE
df_ml = processar_genetica_correctament(df, cols_gens, min_freq=MIN_FREQ_ALELS)

# -----------------------------------------------------------------------------
# 3. NETEJA DE TARGET I CENSURA
# -----------------------------------------------------------------------------
# 1. Eliminem mostres sense dades clíniques (els orfes)
df_ml = df_ml.dropna(subset=['ML_TARGET_EXIT (0/1)'])

# 2. Filtrem pacients censurats prematurs (<2 anys sense èxit)
df_ml['DIES_EVENT'] = pd.to_numeric(df_ml['DIES_EVENT'], errors='coerce')

mask_censurat_prematur = (
    (df_ml['ML_TARGET_EXIT (0/1)'] == 0) & 
    (df_ml['ESTAT_DETALLAT'].astype(str).str.contains('ACTIU', case=False, na=False)) & 
    (df_ml['DIES_EVENT'] < DIES_TALL_CENSURA)
)

n_censurats = mask_censurat_prematur.sum()
df_final = df_ml[~mask_censurat_prematur].copy()

print(f" -> Pacients censurats (<2 anys) eliminats: {n_censurats}")
print(f" -> Mida final del Dataset per entrenar: {len(df_final)} pacients.")

if len(df_final) < 20:
    sys.exit("❌ Error Crític: Massa pocs pacients per fer estadística fiable.")

# =============================================================================
# 4. PREPARACIÓ DEL MODEL
# =============================================================================
y = df_final['ML_TARGET_EXIT (0/1)'].astype(int)

cols_to_drop = ['MOSTRA', 'MATCH_ID', 'ESTAT_DETALLAT', 'ML_TARGET_EXIT (0/1)', 'DIES_EVENT']
cols_to_drop = [c for c in cols_to_drop if c in df_final.columns]

X = df_final.drop(columns=cols_to_drop)
X = pd.get_dummies(X, drop_first=True) # Per si queda sexe o transcrit com a text

# Pesos per classes
ratio = float(np.sum(y == 0)) / np.sum(y == 1)
if ratio == 0: ratio = 1

print(f" -> Features totals al model: {X.shape[1]}")
print(f" -> Ratio Desbalanceig: {ratio:.2f}")

# =============================================================================
# 5. CROSS-VALIDATION + SHAP
# =============================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

auc_scores = []
acc_scores = []
shap_values_list = []
X_test_list = []

print(f"\n>>> ENTRENANT MODEL (XGBoost) EN {N_FOLDS} FOLDS...")

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Model amb paràmetres conservadors per evitar overfitting
    model = xgb.XGBClassifier(
        n_estimators=150,        # Menys arbres per dades petites
        learning_rate=0.05,
        max_depth=3,             # Arbres menys profunds (més simples)
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        eval_metric='logloss',
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.5
        
    acc_scores.append(acc)
    auc_scores.append(auc)
    
    print(f"   [Fold {fold+1}] Accuracy: {acc:.3f} | AUC: {auc:.3f}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values_list.append(shap_values)
    X_test_list.append(X_test)

# =============================================================================
# 6. RESULTATS FINALS
# =============================================================================
print("-" * 60)
print("📊 RESULTATS FINALS (AMB FILTRE):")
print(f" -> Accuracy Mitjana: {np.mean(acc_scores):.3f} (+/- {np.std(acc_scores):.3f})")
print(f" -> AUC-ROC Mitjana:  {np.mean(auc_scores):.3f} (+/- {np.std(auc_scores):.3f})")
print("-" * 60)

if shap_values_list:
    all_shap_values = np.concatenate(shap_values_list, axis=0)
    all_X_test = pd.concat(X_test_list, axis=0)

    print(" -> Generant gràfic SHAP...")
    plt.figure(figsize=(12, 10))
    # Mostrem les top 20 variables més importants
    shap.summary_plot(all_shap_values, all_X_test, show=False, max_display=20)
    plt.title("Impacte Genètic (Variables Freqüents)", fontsize=16)
    plt.tight_layout()

    output_plot = os.path.join(OUTPUT_DIR, 'SHAP_SUMMARY_PLOT.png')
    plt.savefig(output_plot, dpi=300)
    print(f"✅ Gràfic guardat a: {output_plot}")
    
    # Importància Excel
    vals = np.abs(all_shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(all_X_test.columns, vals)), columns=['Feature','Importance_SHAP'])
    feature_importance.sort_values(by='Importance_SHAP', ascending=False, inplace=True)
    feature_importance.to_excel(os.path.join(OUTPUT_DIR, 'Importancia_Variables.xlsx'), index=False)
    print(f"✅ Taula d'importància guardada.")

print(">>> PROCÉS COMPLETAT.")