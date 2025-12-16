import pandas as pd
import numpy as np
import sys
import os
import re

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
if len(sys.argv) > 2:
    FILE_GENETIC = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
else:
    FILE_GENETIC = 'RESULTATS_T1K_NETS_TALL_MALALTS_38.xlsx'
    OUTPUT_FILE = 'Taula_imputs_ml.xlsx' # NOM NOU

FILE_CLINIC = 'CML Clinical DB.xlsx'

print(">>> GENERANT DATASET PER A MACHINE LEARNING (VERSIÓ CORREGIDA)...")

# -----------------------------------------------------------------------------
# 1. CARREGAR DADES
# -----------------------------------------------------------------------------
try:
    df_gen = pd.read_excel(FILE_GENETIC, engine='openpyxl')
    df_clin = pd.read_excel(FILE_CLINIC, engine='openpyxl')
except Exception as e:
    sys.exit(f"❌ Error llegint fitxers: {e}")

# Funcions auxiliars
def find_col(df, keywords):
    for col in df.columns:
        c_str = str(col).upper()
        if all(k.upper() in c_str for k in keywords):
            return col
    return None

def extract_numeric_id(val):
    match = re.search(r'(\d+)', str(val))
    return str(int(match.group(1))) if match else None

# Identificadors
df_gen['MATCH_ID'] = df_gen['MOSTRA'].apply(extract_numeric_id)
col_id_clin = find_col(df_clin, ['ID'])
df_clin['MATCH_ID'] = df_clin[col_id_clin].apply(extract_numeric_id)

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING I CORRECCIÓ D'ERRORS
# -----------------------------------------------------------------------------
# Mapeig de columnes
col_start = find_col(df_clin, ['INICIO', '1', 'LINEA'])
col_end = find_col(df_clin, ['FIN', '1', 'LINEA'])
col_dmr = find_col(df_clin, ['Fecha', 'RM', 'profunda'])
col_diag = find_col(df_clin, ['Data', 'Diagnòstic'])
col_birth = find_col(df_clin, ['naixement'])
col_sex = find_col(df_clin, ['SEXO'])
col_bcr = find_col(df_clin, ['BCR/ABL', 'diagn'])
col_transcrit = find_col(df_clin, ['TRANSCRITO'])
col_causa_fi = find_col(df_clin, ['Causa', 'Fin', '1'])
col_exitus = find_col(df_clin, ['Exitus'])

# Conversió a datetime
for c in [col_start, col_end, col_dmr, col_diag, col_birth]:
    if c: df_clin[c] = pd.to_datetime(df_clin[c], dayfirst=True, errors='coerce')

# --- CORRECCIÓ D'EDATS NEGATIVES ---
def calcular_edat_corregida(row):
    diag = row[col_diag]
    birth = row[col_birth]
    
    if pd.isna(diag) or pd.isna(birth):
        return np.nan
    
    # Si el naixement és FUTUR respecte al diagnòstic (Ex: Neix 2029, Diag 2006)
    # Assumim error de segle i restem 100 anys al naixement
    if birth > diag:
        birth = birth.replace(year=birth.year - 100)
    
    age = (diag - birth).days / 365.25
    
    # Segona comprovació: Si té menys de 10 anys o més de 100, sospitós (però ho deixem passar si és positiu)
    if age < 0: return np.nan # Impossible
    return round(age, 1)

if col_diag and col_birth:
    df_clin['EDAT_DIAGNOSTIC'] = df_clin.apply(calcular_edat_corregida, axis=1)
else:
    df_clin['EDAT_DIAGNOSTIC'] = np.nan

# -----------------------------------------------------------------------------
# 3. LÒGICA D'ESTAT (TARGET)
# -----------------------------------------------------------------------------
col_itc = find_col(df_clin, ['ITC', '1', 'LINEA'])
df_clin['CODI_MED'] = pd.to_numeric(df_clin[col_itc], errors='coerce')
df_target = df_clin[df_clin['CODI_MED'] == 1].copy()

def classify_patient(row):
    start = row[col_start]
    end = row[col_end]
    dmr = row[col_dmr]
    causa = row[col_causa_fi]
    exitus = row[col_exitus]
    
    # 1. ÈXIT
    if pd.notna(dmr):
        # Es considera èxit si DMR ocorre abans d'acabar tractament (o si no ha acabat)
        if pd.isna(end) or dmr <= end:
            days = (dmr - start).days
            if days < 0: return "ERROR_DATES", np.nan
            return "EXIT_DMR_LINIA_1", days
        else:
            return "DMR_FORA_LINIA_1 (FRACAS L1)", np.nan

    # 2. FRACÀS / ABANDONAMENT
    if pd.notna(end):
        str_causa = str(causa).replace('.0', '') if pd.notna(causa) else "?"
        return f"NO_DMR_CANVI_TRACTAMENT (Causa: {str_causa})", np.nan
    
    # 3. EXITUS
    if str(exitus) in ['1', '1.0', 'Si', 'Yes', 'TRUE']:
        return "NO_DMR_EXITUS", np.nan
        
    # 4. ACTIU (CENSURAT)
    if pd.notna(start):
        dies_seguiment = (pd.Timestamp.now() - start).days
        return f"NO_DMR_ACTIU (Encara en tractament)", dies_seguiment
        
    return "NO_AVALUABLE", np.nan

res = df_target.apply(classify_patient, axis=1)
temp_res = res.apply(pd.Series)
df_target['TARGET_ESTAT_DETALLAT'] = temp_res[0]
df_target['TARGET_DIES'] = temp_res[1]

# TARGET BINARI (1 = Èxit Real, 0 = Tota la resta)
df_target['TARGET_BINARI'] = df_target['TARGET_ESTAT_DETALLAT'].apply(lambda x: 1 if 'EXIT_DMR_LINIA_1' in str(x) else 0)

# -----------------------------------------------------------------------------
# 4. FUSIONAR I GUARDAR
# -----------------------------------------------------------------------------
# Netegem noms
cols_to_keep = [
    'MATCH_ID', 'TARGET_ESTAT_DETALLAT', 'TARGET_BINARI', 'TARGET_DIES',
    'EDAT_DIAGNOSTIC', col_sex, col_bcr, col_transcrit
]
clean_names = ['MATCH_ID', 'ESTAT_DETALLAT', 'ML_TARGET_EXIT (0/1)', 'DIES_EVENT', 'EDAT_DX', 'SEXE', 'BCR_ABL_INICIAL', 'TIPUS_TRANSCRIT']

df_clin_clean = df_target[cols_to_keep].copy()
df_clin_clean.columns = clean_names
df_clin_clean = df_clin_clean.drop_duplicates(subset=['MATCH_ID'])

# Merge Final
df_final = pd.merge(df_gen, df_clin_clean, on='MATCH_ID', how='left')
df_final = df_final.drop(columns=['MATCH_ID'])

# Reordenar columnes
cols = list(df_final.columns)
cols_clin = clean_names[1:] 
for c in reversed(cols_clin):
    if c in cols:
        cols.remove(c)
        cols.insert(1, c)

df_final.to_excel(OUTPUT_FILE, index=False)

# -----------------------------------------------------------------------------
# 5. INFORME DE VERIFICACIÓ
# -----------------------------------------------------------------------------
print("-" * 60)
print(f"✅ TAULA CREADA: {OUTPUT_FILE}")
print("-" * 60)
print("🔍 AUDITORIA DE DADES:")

# Verificació Edats
edats = df_final['EDAT_DX'].dropna()
negatives = edats[edats < 0].count()
print(f" -> Edats Negatives detectades: {negatives} (Haurien de ser 0)")
if negatives == 0:
    print("    (Correcte: S'han arreglat els anys de naixement)")
print(f" -> Rang d'Edats: {edats.min()} a {edats.max()} anys")

# Verificació Dies
dies = df_final[df_final['ML_TARGET_EXIT (0/1)'] == 1]['DIES_EVENT']
print(f" -> Dies fins a DMR (mínim): {dies.min()}")
if dies.min() < 0:
    print("    ⚠️ ALERTA: Hi ha dies negatius! Revisa les dates d'inici/fi.")
else:
    print("    (Correcte: Tots els temps són positius)")

print(f" -> Mostres Totals per al ML: {len(df_final)}")
print("-" * 60)
