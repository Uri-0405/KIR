import pandas as pd
import os

files_to_check = [
    'ABUNDANCIES_MALALTS.xlsx',
    'RESULTATS_T1K_NETS_TALL_MALALTS_38.xlsx',
    'Taula_imputs_ml.xlsx'
]

for f in files_to_check:
    print(f"\n--- INSPECTING: {f} ---")
    if not os.path.exists(f):
        print("❌ File not found.")
        continue
        
    try:
        df = pd.read_excel(f, engine='openpyxl', nrows=5)
        print(df.head())
        print("\nColumn Types:")
        print(df.dtypes)
        
        # Check content of a gene column if possible
        possible_genes = ['3DL1', 'KIR3DL1', '2DL1', 'KIR2DL1']
        for g in possible_genes:
            if g in df.columns:
                print(f"\nSample content of {g}:")
                print(df[g].head())
                break
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")
