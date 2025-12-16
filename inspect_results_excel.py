import pandas as pd
try:
    df = pd.read_excel('RESULTATS_T1K_NETS_TALL_MALALTS.xlsx')
    print("Columns:", df.columns.tolist()[:20])
    print("First row:", df.iloc[0].tolist()[:20])
except Exception as e:
    print(e)
