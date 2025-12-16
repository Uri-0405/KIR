import pandas as pd
import numpy as np
import sys

if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    INPUT_FILE = 'Taula_imputs_ml.xlsx'

print(f">>> AUDITORIA DE DATA LEAKAGE I INTEGRITAT DE DADES ({INPUT_FILE})")

try:
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')
except Exception as e:
    sys.exit(f"❌ Error llegint fitxer: {e}")

print(f"📂 Fitxer carregat: {INPUT_FILE}")
print(f"   - Files: {len(df)}")
print(f"   - Columnes: {len(df.columns)}")

# ---------------------------------------------------------
# 1. DETECCIÓ DE DUPLICATS (DATA LEAKAGE GREU)
# ---------------------------------------------------------
print("\n🔍 1. COMPROVACIÓ DE PACIENTS DUPLICATS")
col_id = 'MOSTRA' if 'MOSTRA' in df.columns else 'MATCH_ID'

if col_id in df.columns:
    n_unique = df[col_id].nunique()
    n_total = len(df)
    duplicates = n_total - n_unique
    
    if duplicates > 0:
        print(f"⚠️ ALERTA CRÍTICA: Hi ha {duplicates} pacients duplicats (basat en {col_id})!")
        print("   Això causa 'Data Leakage' si un cau a Train i l'altre a Test.")
        print("   Exemples de duplicats:")
        print(df[col_id].value_counts().head())
    else:
        print(f"✅ Correcte: Tots els pacients són únics (basat en {col_id}).")
else:
    print("⚠️ No trobo columna d'identificació (MOSTRA o MATCH_ID).")

# ---------------------------------------------------------
# 2. DETECCIÓ DE LEAKAGE DIRECTE (Correlació amb Target)
# ---------------------------------------------------------
print("\n🔍 2. COMPROVACIÓ DE CORRELACIÓ SOSPITOSA (TARGET LEAKAGE)")

# Repliquem la lògica de target de l'script principal
DIES_TALL_CENSURA = 730
def refinar_target(row):
    estat = str(row['ESTAT_DETALLAT'])
    dies = row['DIES_EVENT']
    target_original = row['ML_TARGET_EXIT (0/1)']
    
    if target_original == 1: return 1
    if "NO_DMR_ACTIU" in estat:
        if pd.notna(dies) and dies > DIES_TALL_CENSURA: return 0 
        else: return np.nan 
    return 0

df['TARGET_FINAL'] = df.apply(refinar_target, axis=1)
df_model = df.dropna(subset=['TARGET_FINAL']).copy()

# Preparem dades numèriques per correlació
# Convertim categòriques a codes per veure si alguna "string" canta massa
df_numeric = df_model.select_dtypes(include=[np.number])

correlations = df_numeric.corrwith(df_model['TARGET_FINAL']).abs().sort_values(ascending=False)

print("   Top 10 Variables més correlacionades amb el Target:")
for col, val in correlations.head(10).items():
    if col == 'TARGET_FINAL': continue
    print(f"   - {col}: {val:.4f}")
    if val > 0.95:
        print(f"     🚨 PERILL: La variable '{col}' sembla ser el mateix Target (Leakage!)")

# ---------------------------------------------------------
# 3. VERIFICACIÓ DE VARIABLES PROHIBIDES
# ---------------------------------------------------------
print("\n🔍 3. VERIFICACIÓ DE VARIABLES 'FUTURES' (Que no haurien de ser-hi)")
# Llista de paraules que indiquen futur o resultat
forbidden_keywords = ['DIES', 'DAYS', 'MESOS', 'MONTHS', 'DATA', 'DATE', 'REMISSIO', 'REMISSION', 'ESTAT', 'STATUS', 'EXIT', 'FRACAS']

cols_sospitoses = []
for col in df_model.columns:
    # Ignorem les que ja sabem que excloem explícitament a l'altre script, 
    # però aquí volem veure si n'hi ha alguna d'amagada.
    if col in ['TARGET_FINAL', 'ML_TARGET_EXIT (0/1)', 'DIES_EVENT', 'ESTAT_DETALLAT']: continue
    
    for kw in forbidden_keywords:
        if kw in str(col).upper():
            cols_sospitoses.append(col)

if cols_sospitoses:
    print(f"⚠️ Atenció: Revisa aquestes columnes, podrien contenir informació del futur:")
    print(cols_sospitoses)
else:
    print("✅ No s'han detectat columnes amb noms sospitosos de 'futur'.")

# ---------------------------------------------------------
# 4. CONSISTÈNCIA DE DADES
# ---------------------------------------------------------
print("\n🔍 4. CONSISTÈNCIA DE DADES")
print(f"   - Pacients totals analitzables: {len(df_model)}")
print(f"   - Target 1 (Èxit): {sum(df_model['TARGET_FINAL'] == 1)}")
print(f"   - Target 0 (Fracàs/No assolit >2a): {sum(df_model['TARGET_FINAL'] == 0)}")

if sum(df_model['TARGET_FINAL'] == 1) < 10 or sum(df_model['TARGET_FINAL'] == 0) < 10:
    print("⚠️ ALERTA: Molt poques mostres d'una classe. El model serà inestable.")

print("\n✅ AUDITORIA FINALITZADA.")
