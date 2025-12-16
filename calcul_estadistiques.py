import pandas as pd
import sys
import os

# ---------------------------------------------------------
# CONFIGURACIÓ
# ---------------------------------------------------------
# Posa aquí el nom del teu fitxer final net
FILE_INPUT = 'RESULTATS_T1K_NETS_TALL_MALALTS.xlsx' 
FILE_OUTPUT = 'ESTADISTIQUES_ALELS_MALALTS.xlsx'

# Colors
C_GREEN = '\033[92m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'

print(f"{C_CYAN}>>> CALCULANT ESTADÍSTIQUES DE FREQÜÈNCIES...{C_RESET}")

# 1. CARREGAR DADES
if not os.path.exists(FILE_INPUT):
    print(f"❌ No trobo el fitxer: {FILE_INPUT}")
    sys.exit()

df = pd.read_excel(FILE_INPUT, index_col=0).fillna('')
total_pacients = len(df)
print(f"   Pacients totals analitzats: {total_pacients}")

data_stats = []

# 2. ITERAR PER CADA GEN
columnes_gens = df.columns.tolist()

for gen in columnes_gens:
    # Diccionari per comptar quants pacients tenen cada al·lel
    conteo_alels = {}
    
    # Comptador de quants pacients tenen el gen (positius generals)
    n_positius_gen = 0
    
    for mostra, valor in df[gen].items():
        val_str = str(valor).strip()
        
        if not val_str:
            continue # Pacient negatiu per aquest gen
            
        n_positius_gen += 1
        
        # Separem per '/' i fem set() per comptar només 1 cop per pacient
        # (Si un pacient és *001/*001, compta com a portador de *001 una vegada)
        alels_pacient = set(x for x in val_str.split('/') if x)
        
        for alel in alels_pacient:
            conteo_alels[alel] = conteo_alels.get(alel, 0) + 1
            
    # Guardem dades dels al·lels d'aquest gen
    for alel, count in conteo_alels.items():
        freq_total = (count / total_pacients) * 100
        
        # Opcional: Frequencia relativa (sobre els positius del gen)
        # freq_relativa = (count / n_positius_gen * 100) if n_positius_gen > 0 else 0
        
        data_stats.append({
            'GEN': gen,
            'ALEL': alel,
            'N_PACIENTS': count,
            'TOTAL_COHORT': total_pacients,
            'FREQ_PORTADOR (%)': round(freq_total, 2)
        })

# 3. CREAR DATAFRAME I EXCEL
df_stats = pd.DataFrame(data_stats)

# Ordenem: Primer per Gen, després per Frequencia (de més a menys)
df_stats = df_stats.sort_values(by=['GEN', 'FREQ_PORTADOR (%)'], ascending=[True, False])

# Guardem
df_stats.to_excel(FILE_OUTPUT, index=False)

print(f"\n✅ Estadístiques generades correctament!")
print(f"📁 Resultat guardat a: {C_GREEN}{FILE_OUTPUT}{C_RESET}")
print("\nEXEMPLE DEL TOP 5 ALELS MÉS COMUNS:")
print(df_stats.head(5).to_string(index=False))
