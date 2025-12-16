import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
import os

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = 'Taula_imputs_ml.xlsx'

if len(sys.argv) > 2:
    RANDOM_SEED = int(sys.argv[2])
else:
    RANDOM_SEED = 42

DIES_TALL_CENSURA = 730  # 2 anys (365 * 2)
N_FOLDS = 5  # Nombre de folds per Cross-Validation

print(">>> INICIANT ANÀLISI XGBOOST + SHAP AMB CROSS-VALIDATION RIGORÓS")

# 1. CARREGAR DADES
if not os.path.exists(INPUT_FILE):
    sys.exit(f"❌ Error: No trobo {INPUT_FILE}. Assegura't d'haver executat l'script de preparació de dades.")

df = pd.read_excel(INPUT_FILE, engine='openpyxl')
print(f" -> Dades originals carregades: {len(df)} pacients")

# 2. FILTRATGE INTEL·LIGENT (Gestió de Censurats)
def refinar_target(row):
    estat = str(row['ESTAT_DETALLAT'])
    dies = row['DIES_EVENT']
    target_original = row['ML_TARGET_EXIT (0/1)']
    
    if target_original == 1:
        return 1
    
    if "NO_DMR_ACTIU" in estat:
        if pd.notna(dies) and dies > DIES_TALL_CENSURA:
            return 0 
        else:
            return np.nan 
            
    return 0

df['TARGET_FINAL'] = df.apply(refinar_target, axis=1)
df_model = df.dropna(subset=['TARGET_FINAL']).copy()

print(f" -> Després de filtrar censurats (<2 anys): {len(df_model)} pacients")
print(f" -> Distribució Target: {df_model['TARGET_FINAL'].value_counts().to_dict()}")

# 3. PREPARACIÓ DE VARIABLES (ONE-HOT ENCODING & NETEJA)
cols_excloure = ['ESTAT_DETALLAT', 'ML_TARGET_EXIT (0/1)', 'DIES_EVENT', 'TARGET_FINAL', 'MOSTRA', 'MATCH_ID']
cols_cliniques = ['EDAT_DX', 'SEXE', 'BCR_ABL_INICIAL', 'TIPUS_TRANSCRIT']
cols_genetiques = [c for c in df_model.columns if c not in cols_excloure and c not in cols_cliniques]

# Processament de variables categòriques (Sexe, Transcrit)
df_model = pd.get_dummies(df_model, columns=['SEXE', 'TIPUS_TRANSCRIT'], drop_first=True)

# Processament GENÈTIC (One-Hot Encoding SENSE Agrupació)
processed_genes = pd.DataFrame(index=df_model.index)

print(f" -> Processant genètica (sense agrupació d'al·lels rars)...")

for col in cols_genetiques:
    if pd.api.types.is_numeric_dtype(df_model[col]):
        processed_genes[col] = df_model[col].fillna(0)
    else:
        # One-Hot manual per llistes separades per barres (CORRECCIÓ IMPORTANT: abans era coma)
        # Assegurem que no hi ha espais en blanc que puguin duplicar columnes (ex: " *001" vs "*001")
        dummies = df_model[col].astype(str).str.replace(' ', '').str.get_dummies(sep='/')
        # Prefixem amb el nom del gen
        dummies.columns = [f"{col}_{c}" for c in dummies.columns]
        processed_genes = pd.concat([processed_genes, dummies], axis=1)

print(f" -> Exemples de columnes generades (verificació separació): {list(processed_genes.columns)[:5]}")


# Unim tot per crear la matriu X
# NO IMPUTEM ENCARA (ho farem dins del CV per evitar leakage)
X_clin = df_model[['EDAT_DX', 'BCR_ABL_INICIAL']].copy()
X = pd.concat([X_clin, processed_genes], axis=1)

# Afegim les dummies de sexe/transcrit
cols_dummies_clin = [c for c in df_model.columns if 'SEXE_' in c or 'TIPUS_TRANSCRIT_' in c]
X = pd.concat([X, df_model[cols_dummies_clin]], axis=1)

y = df_model['TARGET_FINAL']

# 4. CROSS-VALIDATION RIGORÓS (Stratified K-Fold) + OPTIMITZACIÓ
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

acc_scores = []
auc_scores = []
shap_values_list = []
X_test_list = [] 

print(f"\n>>> INICIANT CROSS-VALIDATION ({N_FOLDS} Folds) AMB OPTIMITZACIÓ D'HIPERPARÀMETRES...")

fold = 1
for train_index, test_index in skf.split(X, y):
    # 1. Separem Train i Test
    X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 2. IMPUTACIÓ DINS DEL BUCLE (Evita Data Leakage)
    imputer = SimpleImputer(strategy='mean')
    
    # Important: mantenir noms de columnes per SHAP
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns, index=X_test_raw.index)
    
    # 3. CONFIGURACIÓ ROBUSTA (SENSE OPTIMITZACIÓ ALEATÒRIA)
    # Utilitzem els paràmetres que han demostrat ser els millors en l'experiment de Monte Carlo
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    best_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=RANDOM_SEED,
        n_jobs=1
    )
    
    best_model.fit(X_train, y_train)
    
    # 4. Prediccions i Mètriques amb el millor model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.5 
    
    acc_scores.append(acc)
    auc_scores.append(auc)
    
    print(f"   Fold {fold}: Accuracy = {acc:.3f}, AUC = {auc:.3f} | Params: Fixed Robust")
    
    # 5. SHAP
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    
    shap_values_list.append(shap_values)
    X_test_list.append(X_test)
    
    fold += 1

# =============================================================================
# RESULTATS FINALS
# =============================================================================
print("-" * 40)
print("📊 RESULTATS MITJANS CROSS-VALIDATION:")
print(f" -> Accuracy Mitjana: {np.mean(acc_scores):.3f} (+/- {np.std(acc_scores):.3f})")
print(f" -> AUC-ROC Mitjana: {np.mean(auc_scores):.3f} (+/- {np.std(auc_scores):.3f})")
print("-" * 40)

# Concatenar resultats SHAP de tots els folds
all_shap_values = np.concatenate(shap_values_list, axis=0)
all_X_test = pd.concat(X_test_list, axis=0)

print(" -> Generant gràfics SHAP globals...")
plt.figure()
shap.summary_plot(all_shap_values, all_X_test, show=False)
plt.savefig('GRAFIC_SHAP_GLOBAL_CV.png', bbox_inches='tight', dpi=300)
print("✅ Gràfic guardat: GRAFIC_SHAP_GLOBAL_CV.png")

plt.figure()
shap.summary_plot(all_shap_values, all_X_test, plot_type="bar", show=False)
plt.savefig('GRAFIC_SHAP_IMPORTANCIA_CV.png', bbox_inches='tight', dpi=300)
print("✅ Gràfic guardat: GRAFIC_SHAP_IMPORTANCIA_CV.png")

print("\n🎉 ANÀLISI COMPLETADA.")
