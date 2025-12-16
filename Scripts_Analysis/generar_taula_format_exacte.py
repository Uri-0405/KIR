import os
import csv
import glob

# --- CONFIGURACIÓ ---
DIR_RESULTS = "results"
OUTPUT_FILE = "TAULA_GLOBAL_RESULTATS.csv"

def generar_taula_global():
    print(f"--- Escanejant directoris dins de '{DIR_RESULTS}' ---")

    if not os.path.exists(DIR_RESULTS):
        print(f"ERROR: No existeix el directori {DIR_RESULTS}")
        return

    # Estructura per guardar dades: dades[mostra][gen] = al·lel
    dades_globals = {}
    tots_els_gens = set()

    # 1. Busquem totes les carpetes dins de results
    carpetes = [f.path for f in os.scandir(DIR_RESULTS) if f.is_dir()]
    
    if not carpetes:
        print("No s'han trobat carpetes de mostres.")
        return

    print(f"S'han trobat {len(carpetes)} mostres. Processant...")

    # 2. Iterem per cada carpeta de mostra
    for carpeta in carpetes:
        nom_mostra = os.path.basename(carpeta) # El nom de la carpeta és el nom de la mostra
        
        # Busquem qualsevol fitxer que acabi en _genotype.tsv dins d'aquesta carpeta
        patro_busqueda = os.path.join(carpeta, "*_genotype.tsv")
        fitxers_trobats = glob.glob(patro_busqueda)

        if not fitxers_trobats:
            print(f"[!] No s'ha trobat fitxer genotype a: {nom_mostra}")
            continue

        fitxer_genotype = fitxers_trobats[0] # Agafem el primer que trobi
        
        # Inicialitzem el diccionari per aquesta mostra
        dades_globals[nom_mostra] = {}

        # 3. Llegim el fitxer genotype de la mostra
        try:
            with open(fitxer_genotype, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                
                # Intentem detectar capçalera
                header = next(reader, None)
                
                for row in reader:
                    if len(row) < 2: continue
                    
                    # Assumim format T1K: Col 0 = Gen, Col 1..N = Al·lels/Info
                    gen = row[0].strip()
                    tots_els_gens.add(gen)
                    
                    # Busquem la columna que tingui l'al·lel (sol tenir un '*')
                    # Si no troba *, agafa la segona columna per defecte
                    allel = "ND"
                    found_allele = False
                    for col in row[1:]:
                        if '*' in col: # Els al·lels solen tenir format KIR2DL1*001
                            allel = col
                            found_allele = True
                            break
                    
                    if not found_allele and len(row) > 1:
                        allel = row[1] # Si no troba asterisc, agafa el segon valor

                    dades_globals[nom_mostra][gen] = allel

        except Exception as e:
            print(f"Error llegint {nom_mostra}: {e}")

    # 4. Generem la taula matriu final
    if not dades_globals:
        print("No s'han pogut extreure dades.")
        return

    print(f"Generant arxiu final: {OUTPUT_FILE} ...")
    
    gens_ordenats = sorted(list(tots_els_gens)) # Ordenem els gens alfabèticament

    with open(OUTPUT_FILE, 'w') as f_out:
        writer = csv.writer(f_out, delimiter=';')
        
        # Escrivim la capçalera: MOSTRA + Tots els gens trobats
        capcalera = ["MOSTRA"] + gens_ordenats
        writer.writerow(capcalera)

        # Escrivim cada mostra (fila)
        for mostra in sorted(dades_globals.keys()):
            fila = [mostra]
            for gen in gens_ordenats:
                # Si la mostra té el gen, posem l'al·lel, si no, posem '-' o 'Negatiu'
                valor = dades_globals[mostra].get(gen, "-")
                fila.append(valor)
            writer.writerow(fila)

    print("-" * 40)
    print(f"FET! Taula guardada a: {OUTPUT_FILE}")
    print(f"Pots veure-la amb: cat {OUTPUT_FILE}")
    print("-" * 40)

if __name__ == "__main__":
    generar_taula_global()
