import pandas as pd
import numpy as np
import sys
import os
import re

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
FILE_CLINIC = 'CML Clinical DB.xlsx'
FILE_GENETIC = 'RESULTATS_T1K_NETS_TALL_MALALTS.xlsx'
OUTPUT_FILE = 'RESULTATS_COMPARACIO_ALELS_MALALTS.xlsx'

print(">>> INICIANT ANÀLISI (DIAGNÒSTIC DE COINCIDÈNCIES)...")

# =============================================================================
# 1. FUNCIONS DE NETEJA
# =============================================================================
def find_col(df, keywords):
    for col in df.columns:
        c_str = str(col).upper()
        if all(k.upper() in c_str for k in keywords):
            return col
    return None

def extract_numeric_id(val):
    """
    Extreu la primera seqüència de números que troba.
    Ex: '71110-SP-P2' -> '71110'
    Ex: '30375.0' -> '30375'
    Ex: 'Muestra 123' -> '123'
    """
    s = str(val).strip()
    match = re.search(r'(\d+)', s)
    if match:
        return str(int(match.group(1))) # Convertim a int i str per treure zeros inicials innecessaris si n'hi ha
    return None

# =============================================================================
# 2. CÀRREGA
# =============================================================================
try:
    df_clin = pd.read_excel(FILE_CLINIC, engine='openpyxl')
    print(f"📄 Fitxer Clínic carregat: {len(df_clin)} files.")
    
    df_gen = pd.read_excel(FILE_GENETIC, engine='openpyxl')
    print(f"📄 Fitxer Genètic carregat: {len(df_gen)} files (Aquest és el màxim teòric de pacients amb ADN).")

except Exception as e:
    print(f"❌ Error llegint fitxers: {e}")
    sys.exit()

# =============================================================================
# 3. PREPARAR CLÍNIC (FILTRE 1a LÍNIA + DATES)
# =============================================================================
COL_ID = find_col(df_clin, ['ID'])
COL_ITC1_TYPE = find_col(df_clin, ['ITC', '1', 'LINEA'])
COL_START_1 = find_col(df_clin, ['INICIO', '1', 'LINEA'])
COL_END_1 = find_col(df_clin, ['FIN', '1', 'LINEA'])
COL_DMR_DATE = find_col(df_clin, ['Fecha', 'RM', 'profunda'])

if not all([COL_ID, COL_ITC1_TYPE, COL_START_1, COL_DMR_DATE]):
    print("❌ ERROR: Falten columnes clau al clínic.")
    sys.exit()

# Arreglem el warning de les dates usant dayfirst=True
print("   Processant dates...")
for c in [COL_START_1, COL_END_1, COL_DMR_DATE]:
    if c: df_clin[c] = pd.to_datetime(df_clin[c], dayfirst=True, errors='coerce')

# Filtrem per Tipus 1
df_clin['CODI_MED'] = pd.to_numeric(df_clin[COL_ITC1_TYPE], errors='coerce')
df_cohort = df_clin[df_clin['CODI_MED'] == 1].copy()

print(f"   Pacients clínics amb medicació Tipus 1: {len(df_cohort)}")

# Calculem Èxit/Fracàs Cronològic
def evaluar_pacient(row):
    start = row[COL_START_1]
    end = row[COL_END_1]
    dmr = row[COL_DMR_DATE]
    
    if pd.isna(dmr) or pd.isna(start): return 0, np.nan # No curat o dades insuficients
    if dmr < start: return 0, np.nan # Incoherent
    
    # Dies fins a curar-se
    days = (dmr - start).days
    
    # Si va deixar la medicació abans de curar-se
    if pd.notna(end) and dmr > end:
        return 0, np.nan # Fracàs de la 1a línia
            
    return 1, days

df_cohort[['REAL_SUCCESS', 'REAL_DAYS']] = df_cohort.apply(
    lambda row: pd.Series(evaluar_pacient(row)), axis=1
)

# =============================================================================
# 4. CREUAMENT D'IDs (MATCHING)
# =============================================================================
print("-" * 50)
print("🔍 DIAGNÒSTIC D'IDs")
print("-" * 50)

# Apliquem la neteja numèrica
df_cohort['MATCH_ID'] = df_cohort[COL_ID].apply(extract_numeric_id)
df_gen['MATCH_ID'] = df_gen['MOSTRA'].apply(extract_numeric_id)

# Eliminem els que no tenen ID vàlid (buits)
df_cohort = df_cohort.dropna(subset=['MATCH_ID'])
df_gen = df_gen.dropna(subset=['MATCH_ID'])

# Fem el merge
df_full = pd.merge(df_cohort, df_gen, on='MATCH_ID', how='inner')

n_clin = len(df_cohort)
n_gen = len(df_gen)
n_match = len(df_full)

print(f"-> IDs Clínics (Tipus 1) disponibles: {n_clin}")
print(f"-> IDs Genètics disponibles:          {n_gen}")
print(f"-> COINCIDÈNCIES FINALS:              {n_match}")

if n_match < n_gen:
    print(f"\n⚠️  ATENCIÓ: Hi ha {n_gen - n_match} mostres genètiques que NO troben amo al Clínic.")
    # Mostrem exemples dels que fallen per si vols investigar
    gen_ids = set(df_gen['MATCH_ID'])
    clin_ids = set(df_cohort['MATCH_ID'])
    missing = list(gen_ids - clin_ids)[:10]
    if missing:
        print(f"   Exemple d'IDs genètics orfes (no són al clínic o no prenen medicació 1): {missing}")
else:
    print("\n✅ ÈXIT TOTAL: Totes les mostres genètiques s'han creuat!")

if n_match == 0:
    print("❌ ERROR FATAL: 0 Coincidències. Revisa si els formats són totalment diferents.")
    sys.exit()

# =============================================================================
# 5. CÀLCULS ESTADÍSTICS (IGUAL QUE ABANS)
# =============================================================================
genes_cols = [c for c in df_gen.columns if c not in ['MOSTRA', 'MATCH_ID', 'ID_CLEAN']]
results = []

print("\n   Calculant estadístiques...")

for gen in genes_cols:
    unique_alleles = set()
    for val in df_full[gen].dropna():
        parts = str(val).split('/')
        for p in parts:
            if p.strip().startswith('*'): unique_alleles.add(p.strip())
            
    for alel in unique_alleles:
        mask = df_full[gen].astype(str).str.contains(re.escape(alel), na=False)
        group_si = df_full[mask]
        group_no = df_full[~mask]

        n_si = len(group_si)
        if n_si < 3: continue 

        pct_si = group_si['REAL_SUCCESS'].mean() * 100
        pct_no = group_no['REAL_SUCCESS'].mean() * 100
        
        days_si = group_si[group_si['REAL_SUCCESS']==1]['REAL_DAYS'].median()
        days_no = group_no[group_no['REAL_SUCCESS']==1]['REAL_DAYS'].median()

        if pd.isna(days_si): days_si = np.nan
        if pd.isna(days_no): days_no = np.nan

        results.append({
            'GEN': gen,
            'ALEL': alel,
            'N_PACIENTS': n_si,
            'EXIT_DMR_PORTADORS (%)': round(pct_si, 1),
            'EXIT_DMR_CONTROL (%)': round(pct_no, 1),
            'DELTA_IMPACTE': round(pct_si - pct_no, 1),
            'DIES_MEDIANA_PORTADORS': days_si,
            'DIES_MEDIANA_CONTROL': days_no
        })

df_res = pd.DataFrame(results).sort_values(by='DELTA_IMPACTE', ascending=False)
df_res.to_excel(OUTPUT_FILE, index=False)

print("-" * 60)
print(f"✅ FITXER GENERAT: {OUTPUT_FILE}")
print("-" * 60)
