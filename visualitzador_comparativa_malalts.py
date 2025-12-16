import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# =============================================================================
# CONFIGURACIÓ
# =============================================================================
INPUT_FILE = 'RESULTATS_COMPARACIO_ALELS_MALALTS.xlsx'
FOLDER = 'VISUALITZACIO_FINAL_PUBLICACIO'

# LLINDARS DE FILTRATGE
MIN_PACIENTS_BARRES = 5   # Per veure tendències generals
MIN_PACIENTS_MAPA = 10    # MÉS STRICTE: Per assegurar significança estadística al mapa

if not os.path.exists(INPUT_FILE):
    print(f"❌ No trobo l'arxiu {INPUT_FILE}.")
    sys.exit()

if not os.path.exists(FOLDER): os.makedirs(FOLDER)

print(">>> Generant gràfics amb etiquetes i filtres estrictes...")

# Carreguem dades
df = pd.read_excel(INPUT_FILE)
df['ETIQUETA'] = df['GEN'] + ' ' + df['ALEL']

# Estil professional
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# =============================================================================
# GRÀFIC 1: IMPACTE EN PROBABILITAT (AMB ETIQUETES %)
# =============================================================================
# Filtrem pel gràfic de barres
df_viz = df[df['N_PACIENTS'] >= MIN_PACIENTS_BARRES].copy()
df_impacte = df_viz.sort_values(by='DELTA_IMPACTE', ascending=False)

plt.figure(figsize=(12, len(df_impacte) * 0.35 + 2))
colors = ['#27ae60' if x >= 0 else '#c0392b' for x in df_impacte['DELTA_IMPACTE']]

ax = sns.barplot(data=df_impacte, x='DELTA_IMPACTE', y='ETIQUETA', palette=colors)

# AFEGIM ELS PERCENTATGES AL COSTAT DE LES BARRES
for p in ax.patches:
    width = p.get_width()
    # Si és positiu, posem el text a la dreta. Si és negatiu, a l'esquerra.
    x_pos = width + (1 if width > 0 else -1)
    align = 'left' if width > 0 else 'right'
    color_text = 'black'
    
    # Text amb format +0.0%
    ax.text(x_pos, p.get_y() + p.get_height()/2 + 0.1, 
            f"{width:+.1f}%", 
            ha=align, va='center', fontsize=10, color=color_text, fontweight='bold')

plt.axvline(0, color='black', linewidth=1)
# Ampliem marges laterals perquè càpiguen els números
plt.xlim(df_impacte['DELTA_IMPACTE'].min() - 15, df_impacte['DELTA_IMPACTE'].max() + 15)

plt.title(f"PROBABILITAT EXTRA DE CURA (Mínim {MIN_PACIENTS_BARRES} pacients)", fontsize=14, pad=20)
plt.xlabel("% Diferència respecte al grup control")
plt.tight_layout()
plt.savefig(os.path.join(FOLDER, "1_PROBABILITAT_AMB_VALORS.png"), dpi=300)
plt.close()

# =============================================================================
# GRÀFIC 2: VELOCITAT (AMB ETIQUETES DIES)
# =============================================================================
df_temps = df_viz.dropna(subset=['DIES_MEDIANA_PORTADORS']).copy()
if not df_temps.empty:
    plt.figure(figsize=(12, len(df_temps) * 0.35 + 2))
    df_temps = df_temps.sort_values(by='DIES_MEDIANA_PORTADORS', ascending=True) # Ràpids primer
    
    global_median = df_temps['DIES_MEDIANA_PORTADORS'].median()
    
    norm = plt.Normalize(df_temps['DIES_MEDIANA_PORTADORS'].min(), df_temps['DIES_MEDIANA_PORTADORS'].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    palette = [sm.to_rgba(x) for x in df_temps['DIES_MEDIANA_PORTADORS']]

    ax = sns.barplot(data=df_temps, x='DIES_MEDIANA_PORTADORS', y='ETIQUETA', palette=palette)
    
    # ETIQUETES DE DIES
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 5, p.get_y() + p.get_height()/2 + 0.1, 
                f"{int(width)} dies", 
                ha='left', va='center', fontsize=9, color='#333333')

    plt.axvline(global_median, color='blue', linestyle='--', label=f'Mitjana Global ({int(global_median)} dies)')
    plt.legend()
    plt.xlim(0, df_temps['DIES_MEDIANA_PORTADORS'].max() + 100) # Marge extra
    plt.title("VELOCITAT: Dies fins a Remissió Profunda", fontsize=14, pad=20)
    plt.xlabel("Dies (Menys és millor)")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "2_VELOCITAT_AMB_VALORS.png"), dpi=300)
    plt.close()

# =============================================================================
# GRÀFIC 3: MAPA DISPERSIÓ FILTRAT (SIGNIFICACIÓ ESTADÍSTICA)
# =============================================================================
# APLIQUEM EL FILTRE ESTRICTE (N >= 10)
df_mapa = df[df['N_PACIENTS'] >= MIN_PACIENTS_MAPA].copy()
df_mapa = df_mapa.dropna(subset=['DIES_MEDIANA_PORTADORS'])

if not df_mapa.empty:
    print(f"   Punts al mapa (N >= {MIN_PACIENTS_MAPA}): {len(df_mapa)}")
    plt.figure(figsize=(12, 10))
    
    # Scatter plot
    sns.scatterplot(
        data=df_mapa, x='DELTA_IMPACTE', y='DIES_MEDIANA_PORTADORS',
        size='N_PACIENTS', hue='DELTA_IMPACTE', palette='RdYlGn', 
        sizes=(200, 1000), # Boles més grans
        edgecolor='black', alpha=0.85
    )
    
    # Línies centrals (Quadrant)
    plt.axvline(0, color='grey', ls='--')
    # Utilitzem la mediana global de TOTA la cohort per tenir referència real, no només la del subgrup
    mediana_ref = df['DIES_MEDIANA_PORTADORS'].median() 
    plt.axhline(mediana_ref, color='grey', ls='--')

    # Etiquetes: Posem etiquetes a TOTS els punts del mapa filtrat perquè són pocs i importants
    for i, row in df_mapa.iterrows():
        plt.text(row['DELTA_IMPACTE']+0.8, row['DIES_MEDIANA_PORTADORS'], 
                 row['ETIQUETA'], fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    plt.title(f"MAPA D'EFICÀCIA CLÍNICA (Només n ≥ {MIN_PACIENTS_MAPA})", fontsize=16, pad=20)
    plt.xlabel("Probabilitat Extra de Curar-se (%)")
    plt.ylabel("Dies fins a Remissió (Baix és millor)")
    
    # Quadrant Ideal
    plt.text(df_mapa['DELTA_IMPACTE'].max(), df_mapa['DIES_MEDIANA_PORTADORS'].min(), 
             "OBJECTIU IDEAL\n(Ràpid i Eficaç)", 
             horizontalalignment='right', verticalalignment='bottom', 
             color='green', fontweight='bold', fontsize=12,
             bbox=dict(facecolor='#e8f5e9', edgecolor='green', boxstyle='round'))

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "3_MAPA_GLOBAL_FILTRAT.png"), dpi=300)
    plt.close()
else:
    print("⚠️ ALERTA: No queden punts al mapa amb el filtre de pacients tan alt. Baixa 'MIN_PACIENTS_MAPA'.")

print(f"✅ Gràfics generats a: {FOLDER}")
