# Predict2Protect - T1K Analysis Pipeline

Aquest repositori conté el flux de treball complet per al genotipatge de KIR/HLA i l'anàlisi clínica associada, utilitzant l'eina T1K.

## Guia d'Execució Pas a Pas

### 1. Execució del T1K ()
Aquest script automatitza l'execució del T1K per a totes les mostres.
- **Entrada**: Fitxers FASTQ situats a la carpeta `fastq/`.
- **Acció**: Processa cada mostra amb T1K (mode Normal o Relaxat).
- **Sortida**: Resultats individuals a la carpeta `results/`.

```bash
./process_kir.sh
```

### 2. Inspecció d'Abundàncies ()
Un cop tenim els resultats bruts, aquest script extreu les abundàncies de tots els al·lels detectats per avaluar la qualitat i decidir punts de tall.
- **Entrada**: Carpeta `results/`.
- **Sortida**: Excel `ANALISI_ABUNDANCIES_TOT_RANK.xlsx`.

```bash
python3 inspector_abundancies.py
```

### 3. Generació de la Taula Neta ()
Aplica un *cutoff* (punt de tall) d'abundància per filtrar el soroll i quedar-se amb els al·lels reals.
- **Entrada**: Resultats de T1K.
- **Configuració**: Es defineix el `ABUNDANCE_CUTOFF` dins l'script.
- **Sortida**: Excel `RESULTATS_T1K_NETS_TALL_3.xlsx` (o similar).

```bash
python3 generar_taula_neta.py
```

### 4. Optimització del Cutoff ()
Utilitza un set de control de **30 casos** coneguts per comparar els resultats de T1K amb la veritat (Golden Standard) i determinar quin és el millor *cutoff* (maximitzant l'F1-score).
- **Entrada**: `ANALISI_ABUNDANCIES_TOT_RANK.xlsx` i `TIPATGES CONTROLS MÈTODE FORMAT BO.xlsx`.
- **Sortida**: `RESULTATS_TRANSPARENTS_FIXED.xlsx` i mètriques de precisió.

### 5. Anàlisi Clínica de Malalts ()
Creua les dades genètiques dels pacients (obtingudes amb el millor cutoff) amb la base de dades clínica per trobar correlacions amb la remissió o resposta al tractament.
- **Entrada**: `RESULTATS_T1K_NETS_TALL_MALALTS.xlsx` i `CML Clinical DB.xlsx`.
- **Sortida**: `RESULTATS_COMPARACIO_ALELS_MALALTS.xlsx`.

### 6. Comparativa Sans vs Malalts ()
Compara les freqüències al·lèliques entre el grup de pacients i un grup de control de ~200 individus sans per identificar al·lels de risc o protecció.
- **Pas previ**: Executar `calcul_estadistiques.py` per generar els fitxers d'estadístiques (`ESTADISTIQUES_ALELS_MALALTS.xlsx` i `ESTADISTIQUES_ALELS_CONTROLS.xlsx`).
- **Sortida**: Gràfics i taules a la carpeta `resultatsAlels/`.

### 7. Visualització de Resultats
Scripts per generar gràfics avançats per a publicació:
- **`visualitzador_alels_resultats.py`**: Genera gràfics d'impacte clínic (RMM, DMR) a `GRAFICS_AVANCATS/`.
- **`visualitzador_comparativa_malalts.py`**: Genera mapes de calor i gràfics comparatius finals a `VISUALITZACIO_FINAL_PUBLICACIO/`.

---

## Estructura de Carpetes

- **`fastq/`**: Fitxers de seqüenciació (input).
- **`results/`**: Sortida bruta del T1K.
- **`results_30_Casos/`**: Resultats del set de validació.
- **`results_malalts/`**: Resultats dels pacients.
- **`results_sans/`**: Resultats dels controls sans.
- **`GRAFICS_AVANCATS/`**, **`resultatsAlels/`**, **`VISUALITZACIO_FINAL_PUBLICACIO/`**: Gràfics generats.

## Instal·lació i Compilació

```bash
git clone https://github.com/pverdura/Predict2Protect.git
cd Predict2Protect
make
```
*Nota: Inclou la compilació local de zlib.*

