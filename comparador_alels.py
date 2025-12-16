import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# ---------------------------------------------------------
# CONFIGURACIÓ
# ---------------------------------------------------------
FILE_PACIENTS = 'ESTADISTIQUES_ALELS_MALALTS.xlsx'
FILE_CONTROLS = 'ESTADISTIQUES_ALELS_CONTROLS.xlsx'

# CARPETA DE SORTIDA
OUTPUT_DIR = 'resultatsAlels'

# LLINDARS
LLINDAR_DISCORDANCIA = 5.0   # Mínim % de diferència absoluta
LLINDAR_N_MINIM = 10         # Mínim d'individus totals (Robustesa)
LLINDAR_N_RELATIU = 15       # Mínim d'individus per al gràfic relatiu

# Colors consola
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_RESET = '\033[0m'

print(f"{C_CYAN}>>> GENERANT SET COMPLET DE GRÀFICS A '{OUTPUT_DIR}'...{C_RESET}")

# ---------------------------------------------------------
# 0. PREPARAR CARPETA
# ---------------------------------------------------------
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📁 Carpeta creada: {OUTPUT_DIR}")
else:
    print(f"📁 Carpeta detectada: {OUTPUT_DIR}")

# ---------------------------------------------------------
# 1. CARREGAR I PREPARAR DADES
# ---------------------------------------------------------
try:
    df_pat = pd.read_excel(FILE_PACIENTS)
    df_ctrl = pd.read_excel(FILE_CONTROLS)
except Exception as e:
    print(f"{C_RED}❌ Error: No trobo fitxers.{C_RESET} {e}")
    sys.exit()

df_pat = df_pat[['GEN', 'ALEL', 'FREQ_PORTADOR (%)', 'N_PACIENTS', 'TOTAL_COHORT']]
df_pat.columns = ['GEN', 'ALEL', 'FREQ_PAT', 'N_PAT', 'TOTAL_PAT']

df_ctrl = df_ctrl[['GEN', 'ALEL', 'FREQ_PORTADOR (%)', 'N_PACIENTS', 'TOTAL_COHORT']]
df_ctrl.columns = ['GEN', 'ALEL', 'FREQ_CTRL', 'N_CTRL', 'TOTAL_CTRL']

df = pd.merge(df_pat, df_ctrl, on=['GEN', 'ALEL'], how='outer').fillna(0)

# Càlculs Bàsics
df['DIFERENCIA'] = df['FREQ_PAT'] - df['FREQ_CTRL']
df['ABS_DIF'] = df['DIFERENCIA'].abs()
df['N_TOTAL_OBSERVATS'] = df['N_PAT'] + df['N_CTRL']
df['ETIQUETA'] = df['GEN'] + df['ALEL']

# Càlcul Odds Ratio (Log OR)
a = df['N_PAT'] + 0.5
b = df['TOTAL_PAT'] - df['N_PAT'] + 0.5
c = df['N_CTRL'] + 0.5
d = df['TOTAL_CTRL'] - df['N_CTRL'] + 0.5
df['ODDS_RATIO'] = (a * d) / (b * c)
df['LOG_OR'] = np.log2(df['ODDS_RATIO'])

# ---------------------------------------------------------
# FUNCIÓ 1: GRÀFIC DE BARRES (Comparació directa)
# ---------------------------------------------------------
def plot_barres_freq(dataframe, titol, nom_fitxer):
    if dataframe.empty: return

    save_path = os.path.join(OUTPUT_DIR, nom_fitxer)

    df_sorted = dataframe.sort_values('DIFERENCIA', ascending=False)
    h = len(df_sorted) * 0.35 + 2
    if h < 6: h = 6

    df_melt = df_sorted.melt(id_vars=['ETIQUETA', 'DIFERENCIA'], 
                             value_vars=['FREQ_PAT', 'FREQ_CTRL'], 
                             var_name='GRUP', value_name='FREQÜÈNCIA')
    df_melt['GRUP'] = df_melt['GRUP'].replace({'FREQ_PAT': 'Pacients', 'FREQ_CTRL': 'Controls'})

    plt.figure(figsize=(13, h))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(
        data=df_melt, x='FREQÜÈNCIA', y='ETIQUETA', hue='GRUP',
        palette={'Pacients': '#ff6b6b', 'Controls': '#4ecdc4'},
        edgecolor='none'
    )

    # Etiquetatge manual segur
    for p in ax.patches:
        width = p.get_width()
        if width > 0: 
            ax.text(width + 0.5, p.get_y() + p.get_height()/2, 
                    f'{width:.1f}%', va='center', fontsize=9)

    plt.title(f"FREQÜÈNCIA: {titol}", fontsize=15, fontweight='bold')
    plt.xlabel("Freqüència de Portadors (%)", fontsize=12)
    plt.ylabel("")
    plt.legend(title='Grup', loc='upper right')
    xmax = df_melt['FREQÜÈNCIA'].max()
    plt.xlim(0, xmax * 1.15) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Barres guardades: {nom_fitxer}")
    plt.close()

# ---------------------------------------------------------
# FUNCIÓ 2: GRÀFIC DE RISC (Odds Ratio)
# ---------------------------------------------------------
def plot_risc_or(dataframe, titol, nom_fitxer):
    if dataframe.empty: return
    
    save_path = os.path.join(OUTPUT_DIR, nom_fitxer)
    
    df_sorted = dataframe.sort_values('LOG_OR', ascending=True)
    h = len(df_sorted) * 0.35 + 2
    if h < 6: h = 6

    plt.figure(figsize=(12, h))
    colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in df_sorted['LOG_OR']]
    
    plt.hlines(y=df_sorted['ETIQUETA'], xmin=0, xmax=df_sorted['LOG_OR'], color=colors, alpha=0.6, linewidth=2)
    plt.scatter(df_sorted['LOG_OR'], df_sorted['ETIQUETA'], color=colors, s=80, alpha=1)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    plt.title(f"RISC RELATIU (OR): {titol}", fontsize=15, fontweight='bold')
    plt.xlabel("← Protecció (Controls)      |      Risc (Pacients) →", fontsize=11)
    plt.grid(axis='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   🧬 Risc guardat: {nom_fitxer}")
    plt.close()

# ---------------------------------------------------------
# FUNCIÓ 3: GRÀFIC D'IMPACTE RELATIU (Versió Corregida)
# ---------------------------------------------------------
def plot_impacte_relatiu(dataframe, titol, nom_fitxer):
    save_path = os.path.join(OUTPUT_DIR, nom_fitxer)
    
    # 1. Calcular Variació Relativa
    df_calc = dataframe.copy()
    
    def calc_var(row):
        fp, fc = row['FREQ_PAT'], row['FREQ_CTRL']
        if fc == 0 and fp == 0: return 0
        if fc == 0: return 100.0 # Topall visual
        return ((fp - fc) / fc) * 100

    df_calc['VAR_RELATIVA'] = df_calc.apply(calc_var, axis=1)
    df_calc = df_calc.sort_values('VAR_RELATIVA', ascending=False)
    
    # 2. Assignar colors amb hue
    df_calc['TENDENCIA'] = ['Positiu' if x >= 0 else 'Negatiu' for x in df_calc['VAR_RELATIVA']]

    h = len(df_calc) * 0.35 + 2
    if h < 6: h = 6
    
    plt.figure(figsize=(12, h))
    sns.set_style("whitegrid")
    
    # 3. Plot amb Hue + Dodge=False
    ax = sns.barplot(
        data=df_calc, 
        x='VAR_RELATIVA', 
        y='ETIQUETA',
        hue='TENDENCIA',
        palette={'Positiu': '#ff4d4d', 'Negatiu': '#4d4dff'},
        dodge=False,       
        edgecolor='black', 
        linewidth=0.5
    )
    
    # 4. Etiquetatge MANUAL
    for p in ax.patches:
        width = p.get_width()
        
        if not np.isfinite(width) or width == 0:
            continue
            
        txt = f"{'+' if width > 0 else ''}{width:.0f}%"
        offset = 1 if width > 0 else -1
        
        ax.text(
            width + offset,           
            p.get_y() + p.get_height() / 2, 
            txt, 
            va='center', 
            ha='left' if width > 0 else 'right',
            fontsize=9, 
            fontweight='bold',
            color='black'
        )
    
    plt.title(f"IMPACTE RELATIU (% Variació sobre Control)\n{titol}", fontsize=15, fontweight='bold')
    plt.xlabel("← Disminució (% Caiguda)        |        Augment (% Pujada) →", fontsize=12)
    plt.ylabel("")
    plt.axvline(x=0, color='black', linewidth=1.5)
    plt.legend().set_visible(False) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   🚀 Impacte Relatiu guardat: {nom_fitxer}")
    plt.close()

# ---------------------------------------------------------
# EXECUCIÓ DELS 4 BLOCS
# ---------------------------------------------------------

def processar_bloc(df_in, titol, prefix, fer_relatiu=False):
    print(f"\n--- Processant: {titol} ({len(df_in)} al·lels) ---")
    if df_in.empty:
        print("   ⚠️ Buit.")
        return
    
    # 1. Barres freqüència
    plot_barres_freq(df_in, titol, f"{prefix}_FREQ_BARRES.png")
    # 2. Risc OR
    plot_risc_or(df_in, titol, f"{prefix}_RISC_OR.png")
    
    if fer_relatiu:
        df_rel = df_in[df_in['N_TOTAL_OBSERVATS'] >= LLINDAR_N_RELATIU].copy()
        if not df_rel.empty:
            plot_impacte_relatiu(df_rel, f"(N > {LLINDAR_N_RELATIU})", f"4_IMPACTE_RELATIU.png")
        else:
            print("   ⚠️ No hi ha prous dades per al gràfic relatiu.")

# --- EXECUCIÓ ---
processar_bloc(df, "VISTA GLOBAL", "1_GLOBAL", fer_relatiu=False)

df_disc = df[df['ABS_DIF'] >= LLINDAR_DISCORDANCIA].copy()
processar_bloc(df_disc, f"DISCORDANCES > {LLINDAR_DISCORDANCIA}%", "2_DISCORDANCES", fer_relatiu=False)

df_rob = df[df['N_TOTAL_OBSERVATS'] >= LLINDAR_N_MINIM].copy()
processar_bloc(df_rob, f"ROBUSTS (N > {LLINDAR_N_MINIM})", "3_ROBUSTS", fer_relatiu=True)

print(f"\n✅ Procés finalitzat. Revisa la carpeta '{OUTPUT_DIR}'.")
