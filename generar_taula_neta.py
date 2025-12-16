import pandas as pd
import os
import sys
import re

# ---------------------------------------------------------
# CONFIGURACIÓ
# ---------------------------------------------------------
DIR_RESULTATS = './results_malalts'

# Gestió d'arguments per línia de comandes (per optimització)
if len(sys.argv) > 2:
    ABUNDANCE_CUTOFF = float(sys.argv[1])
    OUTPUT_FILE = sys.argv[2]
else:
    OUTPUT_FILE = 'RESULTATS_T1K_NETS_TALL_MALALTS_38.xlsx'
    ABUNDANCE_CUTOFF = 38.0  # El teu número de tall

# Llista de gens que volem a la taula final (ordre estandar)
# Pots afegir-ne o treure'n segons el teu Excel original
GENS_ORDRE = [
    '3DL3', '2DS2', '2DL2', '2DL3', '2DL5B', '2DS3', '2DP1', 
    '2DL1', '3DP1', '2DL4', '3DL1', '3DS1', '2DL5A', '2DS5', 
    '2DS1', '2DS4', '3DL2'
]

print(f">>> Generant taula neta (Tall Abundància >= {ABUNDANCE_CUTOFF})...")

if not os.path.exists(DIR_RESULTATS):
    print("❌ Error: No trobo la carpeta results.")
    sys.exit()

# ---------------------------------------------------------
# 1. PROCESSAR MOSTRES
# ---------------------------------------------------------
data_matrix = []
samples_list = []

carpetes = sorted([d for d in os.listdir(DIR_RESULTATS) if os.path.isdir(os.path.join(DIR_RESULTATS, d))])

for nom_mostra in carpetes:
    path_carpeta = os.path.join(DIR_RESULTATS, nom_mostra)
    # Busquem el fitxer _genotype.tsv
    fitxers = [f for f in os.listdir(path_carpeta) if 'genotype' in f and f.endswith('.tsv')]
    
    if not fitxers: continue
    
    full_path = os.path.join(path_carpeta, fitxers[0])
    
    # Diccionari per guardar els resultats d'aquesta mostra
    # Clau: Gen (ex: 2DL1), Valor: "*001/*002"
    row_data = {gen: '' for gen in GENS_ORDRE}
    
    try:
        # Llegim sense capçalera
        df = pd.read_csv(full_path, sep='\t', header=None, engine='python', on_bad_lines='skip')
        df = df.dropna(how='all')
        
        # Iterem fila per fila (cada fila és un gen)
        for _, row in df.iterrows():
            # Estructura T1K: 0=Gen, 2=Alel1, 3=Abund1, 5=Alel2, 6=Abund2
            gen_raw = str(row[0]).replace('KIR', '').strip() # ex: 2DL1
            
            aleles_valids = []
            
            # --- AL·LEL 1 ---
            if len(row) > 3:
                a1 = str(row[2])
                try: val1 = float(row[3])
                except: val1 = 0.0
                
                if val1 >= ABUNDANCE_CUTOFF and a1 != '.':
                    # Netejem el nom: KIR2DL1*003 -> *003
                    match = re.search(r'\*(\d+)', a1)
                    if match: aleles_valids.append(f"*{match.group(1)}")

            # --- AL·LEL 2 ---
            if len(row) > 6:
                a2 = str(row[5])
                try: val2 = float(row[6])
                except: val2 = 0.0
                
                if val2 >= ABUNDANCE_CUTOFF and a2 != '.':
                    match = re.search(r'\*(\d+)', a2)
                    if match: aleles_valids.append(f"*{match.group(1)}")
            
            # --- GUARDAR RESULTAT ---
            if gen_raw in row_data:
                # Unim amb barres (ex: *001/*002)
                row_data[gen_raw] = '/'.join(aleles_valids)
                
    except Exception as e:
        print(f"⚠️ Error llegint {nom_mostra}: {e}")
        continue

    # Netegem el nom de la mostra (AMAI-KIR -> AMAI) per fer-ho bonic
    clean_sample = nom_mostra.replace('-KIR', '').replace('_KIR', '')
    
    # Afegim a la llista
    full_row = row_data.copy()
    full_row['MOSTRA'] = clean_sample
    data_matrix.append(full_row)

# ---------------------------------------------------------
# 2. CREAR EXCEL FINAL
# ---------------------------------------------------------
if not data_matrix:
    print("❌ No s'han trobat dades vàlides.")
    sys.exit()

df_final = pd.DataFrame(data_matrix)
df_final = df_final.set_index('MOSTRA')

# Reordenem columnes segons l'ordre estandar
cols_existents = [c for c in GENS_ORDRE if c in df_final.columns]
df_final = df_final[cols_existents]

# Guardem
df_final.to_excel(OUTPUT_FILE)

print("\n" + "="*80)
print(f"✅ TAULA NETA GENERADA: {OUTPUT_FILE}")
print(f"   - Tall aplicat: Abundància >= {ABUNDANCE_CUTOFF}")
print("="*80)
print("ARA: Obre aquest Excel i compara'l amb l'original.")
print("Haurien d'haver desaparegut les llistes boges i els falsos positius.")
