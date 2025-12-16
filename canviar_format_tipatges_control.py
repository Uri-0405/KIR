import pandas as pd
import re
import os

# ---------------------------------------------------------
# CONFIGURACIÓ
# ---------------------------------------------------------
FILE_CONTROLS = 'TIPATGES CONTROLS MÈTODE.xlsx'
OUTPUT_FILE = 'CONTROLS_NETS_FORMAT_T1K.xlsx'

print(f">>> Estandarditzant Controls (Mantenint '+'): {FILE_CONTROLS} ...")

try:
    df = pd.read_excel(FILE_CONTROLS, engine='openpyxl')
except Exception as e:
    print(f"❌ Error llegint fitxer: {e}")
    exit()

# ---------------------------------------------------------
# 1. NETEJA DE COLUMNES I ID
# ---------------------------------------------------------
if 'Unnamed' in str(df.columns[0]):
    df.rename(columns={df.columns[0]: 'ID'}, inplace=True)
else:
    df.rename(columns={df.columns[0]: 'ID'}, inplace=True)

GENS_ORDRE = [
    '3DL3', '2DS2', '2DL2', '2DL3', '2DL5B', '2DS3', '2DP1', 
    '2DL1', '3DP1', '2DL4', '3DL1', '3DS1', '2DL5A', '2DS5', 
    '2DS1', '2DS4', '3DL2'
]

data_dict = {}

def netejar_nom_gen(col_name):
    c = str(col_name).strip().upper()
    if '_' in c: c = c.split('_')[0]
    if ' ' in c: c = c.split(' ')[0]
    c = c.replace('KIR', '') 
    return c

cols_map = {}
for c in df.columns:
    if c == 'ID': continue
    gen_net = netejar_nom_gen(c)
    if any(char.isdigit() for char in gen_net):
        cols_map[c] = gen_net

# ---------------------------------------------------------
# 2. PROCESSAMENT FILES
# ---------------------------------------------------------
for _, row in df.iterrows():
    sample_id = str(row['ID']).strip().upper()
    sample_id = sample_id.replace('_KIR', '').strip()
    
    if sample_id in ['NAN', '', 'NONE', '.']: continue
    
    if sample_id not in data_dict:
        data_dict[sample_id] = {}

    for col_orig, gen_net in cols_map.items():
        val = str(row[col_orig]).strip()
        
        # ARA MANTENIM EL '+' I 'POS'
        if val.lower() in ['nan', '', '.', 'neg', 'negative', 'nd']:
            data_dict[sample_id][gen_net] = '' 
            continue
            
        # Normalitzem el positiu a un símbol estàndard '+'
        if val.lower() in ['+', 'pos', 'positive']:
            data_dict[sample_id][gen_net] = '+'
            continue

        parts = re.split(r'[+/,;]', val)
        alels_clean = []
        
        for p in parts:
            p = p.strip()
            if not p: continue
            
            # Si trobem un '+' barrejat amb text, el tractem com a alel "indeterminat"
            if p in ['+', 'POS']:
                # Si ja tenim altres al·lels definits, potser no cal guardar el +
                # Però per seguretat el guardem si és l'únic
                continue 

            if '*' not in p:
                if p.isdigit() or (len(p)>1 and p[0].isdigit()):
                    p = f"*{p}"
            
            # Neteja de prefixos redundants
            if 'KIR' in p or gen_net in p:
                if '*' in p:
                    p = '*' + p.split('*')[1]
            
            alels_clean.append(p)
        
        # Si després de netejar estava buit (i no era +), és buit
        if not alels_clean:
            # Si era un '+' pur, ja l'hem assignat dalt.
            if val.lower() not in ['+', 'pos', 'positive']:
                final_val = ''
            else:
                final_val = '+'
        else:
            alels_clean.sort()
            final_val = '/'.join(alels_clean)
            
        data_dict[sample_id][gen_net] = final_val

# ---------------------------------------------------------
# 3. GUARDAR
# ---------------------------------------------------------
df_out = pd.DataFrame.from_dict(data_dict, orient='index')
cols_presents = [g for g in GENS_ORDRE if g in df_out.columns]
cols_extra = [g for g in df_out.columns if g not in GENS_ORDRE]
df_out = df_out[cols_presents + cols_extra]
df_out.index.name = 'MOSTRA'
df_out.reset_index(inplace=True)
df_out = df_out.sort_values('MOSTRA')

df_out.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Fitxer amb '+' guardat: {OUTPUT_FILE}")
