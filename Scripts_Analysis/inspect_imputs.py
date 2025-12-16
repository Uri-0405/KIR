import pandas as pd
try:
    df = pd.read_excel('Taula_imputs_ml.xlsx')
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(e)
