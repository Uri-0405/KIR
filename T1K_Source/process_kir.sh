#!/bin/bash

# ==========================================
# CONFIGURACIÓ BÀSICA
# ==========================================
FASTQ_DIR="./fastq"              
INDEX="./kir_t1k_index/kir_t1k_index_dna_seq.fa"   
T1K_CMD="./run-t1k"
PYTHON_SCRIPT="generar_taula_format_exacte.py"

# ==========================================
# 0. MENÚ DE SELECCIÓ (NORMAL vs RELAXAT)
# ==========================================
clear
echo "========================================================"
echo "   SELECTOR DE MODE D'ANÀLISI T1K"
echo "========================================================"
echo "Selecciona com vols processar les mostres:"
echo ""
echo "  [n] NORMAL  : Configuració per defecte (Alta fiabilitat)."
echo "  [r] RELAXAT : Baixa exigència (-a 0.01 --weight 0)."
echo "                (Útil per rescatar gens que l'antic veia i el T1K no)"
echo ""
read -p "Tria opció [n/r]: " opcio

# Lògica de selecció
if [[ "$opcio" == "r" || "$opcio" == "R" ]]; then
    echo ""
    echo ">>> ⚠️  MODE RELAXAT ACTIVAT"
    echo ">>> S'aplicaran flags: -a 0.01 --weight 0"
    echo ">>> Els resultats es guardaran a: ./results_relaxat"
    
    # Paràmetres extra per al mode relaxat
    T1K_ARGS="-a 0.01 --weight 0"
    # Canviem la carpeta de sortida per no barrejar resultats
    OUT_DIR="./results_relaxat"
else
    echo ""
    echo ">>> ✅ MODE NORMAL ACTIVAT"
    echo ">>> S'aplicaran els paràmetres estàndard."
    echo ">>> Els resultats es guardaran a: ./results"
    
    # Paràmetres buits per al mode normal
    T1K_ARGS=""
    OUT_DIR="./results"
fi

echo "========================================================"
sleep 2

# ==========================================
# 1. COMPROVACIONS INICIALS
# ==========================================
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ ERROR: No trobo l'script de Python '$PYTHON_SCRIPT'."
    echo "Si us plau, crea'l primer."
    exit 1
fi

mkdir -p $OUT_DIR
echo ">>> [FASE 1] Iniciant anàlisi bioinformàtica (T1K)..."

# ==========================================
# 2. BUCLE DE PROCESSAMENT (T1K)
# ==========================================
count=0
# Busquem els fitxers amb l'extensió exacta
for file1 in $FASTQ_DIR/*_R1_001.fastq.gz; do
    
    [ -e "$file1" ] || continue

    # Deduïm el nom del fitxer parella (R2)
    file2="${file1/_R1_001.fastq.gz/_R2_001.fastq.gz}"
    
    # Netegem el nom del pacient
    filename=$(basename "$file1")
    sample_name=$(echo "$filename" | cut -d_ -f1) 

    echo "   Processant mostra: $sample_name"
    mkdir -p "$OUT_DIR/$sample_name"
    
    # Executem T1K amb els arguments seleccionats ($T1K_ARGS)
    # He afegit $T1K_ARGS just després de la comanda
    $T1K_CMD $T1K_ARGS -1 "$file1" -2 "$file2" -f "$INDEX" -o "$OUT_DIR/$sample_name/$sample_name" -t 8 > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "     -> Fet correctament."
    else
        echo "     -> ❌ Error en processar $sample_name"
    fi

    ((count++))
done

echo ""
echo ">>> [FASE 1 COMPLETADA] S'han processat $count mostres."
echo ">>> Els resultats són a: $OUT_DIR"
