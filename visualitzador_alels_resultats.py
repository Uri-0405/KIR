import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# ---------------------------------------------------------
# CONFIGURACIÓ
# ---------------------------------------------------------
FILE_INPUT = 'RESULTATS_CLINICS_SEPARATS.xlsx'
DIR_OUTPUT = 'GRAFICS_AVANCATS'
MIN_PACIENTS = 5  # Mínim de pacients per considerar l'al·lel fiable al gràfic

# Colors i Estils
sns.set_style("whitegrid")
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'

print(f"{C_CYAN}>>> GENERANT VISUALITZACIONS CLÍNIQUES AVANÇADES...{C_RESET}")

# 1. CARREGAR DADES
if not os.path.exists(FILE_INPUT):
    print(f"❌ Error: No trobo {FILE_INPUT}.")
    sys.exit()

# Llegim les dues pestanyes
df_rmm = pd.read_excel(FILE_INPUT, sheet_name='ANALISI_RMM')
df_dmr = pd.read_excel(FILE_INPUT, sheet_name='ANALISI_PROFUNDA')

# Creem etiquetes úniques
df_rmm['ETIQUETA'] = df_rmm['GEN'] + ' ' + df_rmm['ALEL']
df_dmr['ETIQUETA'] = df_dmr['GEN'] + ' ' + df_dmr['ALEL']

# Creem carpeta
if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

# ---------------------------------------------------------
# GRÀFIC 1: RÀNQUING D'IMPACTE EN DMR (El semàfor)
# ---------------------------------------------------------
def plot_impacte_dmr(df):
    # Filtrar soroll
    df_f = df[df['N_PACIENTS'] >= MIN_PACIENTS].copy()
    df_f = df_f.sort_values('DELTA_PROBABILITAT', ascending=False)
    
    if df_f.empty: return

    plt.figure(figsize=(10, len(df_f)*0.35 + 2))
    
    # Colors: Verd per positiu, Vermell per negatiu
    colors = ['#27ae60' if x >= 0 else '#c0392b' for x in df_f['DELTA_PROBABILITAT']]
    
    ax = sns.barplot(data=df_f, x='DELTA_PROBABILITAT', y='ETIQUETA', palette=colors)
    
    # Afegir valors
    for p in ax.patches:
        width = p.get_width()
        val = f"{width:+.1f}%"
        x = width + (1 if width > 0 else -6)
        y = p.get_y() + p.get_height()/2
        ax.text(x, y, val, va='center', fontsize=9, fontweight='bold', color='black')

    plt.title(f"IMPACTE EN REMISSIÓ PROFUNDA (DMR)\n(Delta vs No Portadors | N >= {MIN_PACIENTS})", fontsize=14, fontweight='bold')
    plt.xlabel("← Risc (Pitjor Resposta)          Benefici (Millor Resposta) →", fontsize=12)
    plt.axvline(0, color='black', linewidth=1)
    plt.ylabel("")
    
    path = os.path.join(DIR_OUTPUT, '1_RANQUING_DMR.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gràfic 1 guardat: Rànquing DMR")

# ---------------------------------------------------------
# GRÀFIC 2: VELOCITAT VS EFICÀCIA (Scatter)
# ---------------------------------------------------------
def plot_velocitat_eficacia(df):
    df_f = df[(df['N_PACIENTS'] >= MIN_PACIENTS) & (df['DELTA_PROBABILITAT'].notna()) & (df['DIES_MEDIANA'].notna())].copy()
    
    if df_f.empty: return

    plt.figure(figsize=(12, 10))
    
    # Scatter plot
    sns.scatterplot(
        data=df_f, 
        x='DELTA_PROBABILITAT', 
        y='DIES_MEDIANA', 
        size='N_PACIENTS', 
        sizes=(50, 500),
        hue='DELTA_PROBABILITAT',
        palette='RdYlGn', 
        alpha=0.8,
        edgecolor='black'
    )
    
    # Etiquetes intel·ligents (només els extrems)
    # Etiquetem: Molt bons (>15%), Molt dolents (<-15%), Molt ràpids (<500d)
    for _, row in df_f.iterrows():
        if abs(row['DELTA_PROBABILITAT']) > 10 or row['DIES_MEDIANA'] < 600:
            plt.text(row['DELTA_PROBABILITAT']+0.5, row['DIES_MEDIANA']-10, row['ETIQUETA'], fontsize=8, fontweight='bold')

    # Quadrants
    plt.axvline(0, color='grey', linestyle='--')
    plt.axhline(df_f['DIES_MEDIANA'].median(), color='grey', linestyle='--')

    # Annotacions
    plt.text(df_f['DELTA_PROBABILITAT'].max(), df_f['DIES_MEDIANA'].min(), "TARGET IDEAL\n(Ràpid i Eficaç)", 
             ha='right', va='bottom', color='green', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.text(df_f['DELTA_PROBABILITAT'].min(), df_f['DIES_MEDIANA'].max(), "ZONA RISC\n(Lent i Poc Eficaç)", 
             ha='left', va='top', color='red', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.title("MAPA DE COMPORTAMENT: Velocitat vs Probabilitat (DMR)", fontsize=16)
    plt.xlabel("Probabilitat Extra d'Èxit (%)", fontsize=12)
    plt.ylabel("Dies fins a Remissió (Mediana)", fontsize=12)
    
    path = os.path.join(DIR_OUTPUT, '2_MAPA_VELOCITAT.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gràfic 2 guardat: Mapa Velocitat")

# ---------------------------------------------------------
# GRÀFIC 3: COMPARATIVA RMM (Curt Termini) vs DMR (Llarg Termini)
# ---------------------------------------------------------
def plot_comparativa_fases(df_rmm, df_dmr):
    # Unim les dades
    merged = pd.merge(
        df_rmm[['ETIQUETA', 'DELTA_PROBABILITAT', 'N_PACIENTS']],
        df_dmr[['ETIQUETA', 'DELTA_PROBABILITAT']],
        on='ETIQUETA',
        suffixes=('_RMM', '_DMR')
    )
    
    # Filtrem
    merged = merged[merged['N_PACIENTS'] >= MIN_PACIENTS]
    if merged.empty: return

    plt.figure(figsize=(10, 10))
    
    # Línia diagonal (Si està a sobre, millora amb el temps. Si està sota, empitjora)
    plt.plot([-30, 40], [-30, 40], ls="--", c=".7", label="Progrés Constant")
    
    sns.scatterplot(
        data=merged,
        x='DELTA_PROBABILITAT_RMM',
        y='DELTA_PROBABILITAT_DMR',
        size='N_PACIENTS',
        color='#3498db',
        sizes=(50, 400),
        alpha=0.7,
        edgecolor='black'
    )
    
    # Etiquetes
    for _, row in merged.iterrows():
        # Destaquem els que tenen gran discrepància
        diff = row['DELTA_PROBABILITAT_DMR'] - row['DELTA_PROBABILITAT_RMM']
        if abs(diff) > 10 or abs(row['DELTA_PROBABILITAT_DMR']) > 15:
            plt.text(row['DELTA_PROBABILITAT_RMM']+0.5, row['DELTA_PROBABILITAT_DMR'], row['ETIQUETA'], fontsize=9)

    plt.title("CONSISTÈNCIA DE LA RESPOSTA: RMM vs DMR", fontsize=16)
    plt.xlabel("Impacte en Fase Inicial (RMM - % Delta)", fontsize=12)
    plt.ylabel("Impacte en Fase Profunda (DMR - % Delta)", fontsize=12)
    
    # Annotacions de zones
    plt.text(30, -20, "FALS POSITIU INICIAL\n(Va bé al principi, falla al final)", color='red', ha='right')
    plt.text(-20, 30, "RECUPERADOR TARDÀ\n(Costa arrencar, però es cura)", color='green', ha='left')
    
    plt.axvline(0, color='grey', alpha=0.3)
    plt.axhline(0, color='grey', alpha=0.3)

    path = os.path.join(DIR_OUTPUT, '3_COMPARATIVA_RMM_DMR.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gràfic 3 guardat: RMM vs DMR")

# ---------------------------------------------------------
# GRÀFIC 4: ELS VELOCISTES (Només els bons)
# ---------------------------------------------------------
def plot_velocistes(df):
    # Només al·lels que ajuden (>0%) i tenen pacients suficients
    df_f = df[(df['DELTA_PROBABILITAT'] > 0) & (df['N_PACIENTS'] >= MIN_PACIENTS)].copy()
    df_f = df_f.sort_values('DIES_MEDIANA', ascending=True).head(15) # Top 15
    
    if df_f.empty: return

    plt.figure(figsize=(10, len(df_f)*0.5 + 2))
    
    ax = sns.barplot(data=df_f, x='DIES_MEDIANA', y='ETIQUETA', palette='viridis')
    
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 5, p.get_y() + p.get_height()/2, f"{int(width)} dies", va='center', fontweight='bold')

    plt.title(f"TOP 15 AL·LELS MÉS RÀPIDS EN CURAR (DMR)\n(Només al·lels de bon pronòstic)", fontsize=14, fontweight='bold')
    plt.xlabel("Dies Medians fins a Remissió Profunda", fontsize=12)
    plt.ylabel("")
    
    path = os.path.join(DIR_OUTPUT, '4_TOP_VELOCISTES.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gràfic 4 guardat: Velocistes")

# ---------------------------------------------------------
# EXECUCIÓ
# ---------------------------------------------------------
plot_impacte_dmr(df_dmr)
plot_velocitat_eficacia(df_dmr)
plot_comparativa_fases(df_rmm, df_dmr)
plot_velocistes(df_dmr)

print(f"\n{C_GREEN}Tots els gràfics s'han desat a la carpeta: '{DIR_OUTPUT}'{C_RESET}")
