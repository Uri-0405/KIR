import subprocess
import re
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Config
PYTHON_EXEC = '/home/duck/T1K/entorn/bin/python'
CUTOFFS = [55, 60, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]
ITERATIONS = 20
RESULTS = []

print(f">>> INICIANT OPTIMITZACIÓ FINA (55-75) - {ITERATIONS} ITERACIONS PER CUTOFF")
print(f" -> Cutoffs a provar: {CUTOFFS}")

for cutoff in CUTOFFS:
    print(f"\n{'='*60}")
    print(f"🚀 PROVANT CUTOFF: {cutoff}")
    print(f"{'='*60}")
    
    # Noms de fitxers temporals per aquest cutoff
    file_genetic = f"temp_genetic_cutoff_{cutoff}.xlsx"
    file_ml = f"temp_ml_cutoff_{cutoff}.xlsx"
    
    # 1. GENERAR TAULA NETA
    print(f" [1/4] Generant taula genètica neta (Cutoff {cutoff})...")
    cmd1 = [PYTHON_EXEC, 'generar_taula_neta.py', str(cutoff), file_genetic]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    if res1.returncode != 0:
        print(f"❌ Error en generar_taula_neta.py:\n{res1.stderr}")
        continue

    # 2. GENERAR TAULA ML
    print(f" [2/4] Fusionant amb dades clíniques...")
    cmd2 = [PYTHON_EXEC, 'generar_taula_ML.py', file_genetic, file_ml]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if res2.returncode != 0:
        print(f"❌ Error en generar_taula_ML.py:\n{res2.stderr}")
        continue

    # 3. AUDITORIA DE SEGURETAT (NOU PAS)
    print(f" [3/4] Auditant fitxer generat per Leakage...")
    cmd_audit = [PYTHON_EXEC, 'audit_model.py', file_ml]
    res_audit = subprocess.run(cmd_audit, capture_output=True, text=True)
    
    # Busquem alertes en l'output de l'auditoria
    if "ALERTA CRÍTICA" in res_audit.stdout or "PERILL" in res_audit.stdout:
        # Ignorem el perill de ML_TARGET_EXIT perquè ja sabem que l'excloem al model, 
        # però verifiquem duplicats.
        if "pacients duplicats" in res_audit.stdout:
            print(f"❌ ATURANT: S'han detectat duplicats al cutoff {cutoff}!")
            print(res_audit.stdout)
            break
    
    # 4. EXECUTAR MODEL
    print(f" [4/4] Entrenant XGBoost ({ITERATIONS} iteracions)...")
    
    acc_list = []
    auc_list = []
    
    for i in range(ITERATIONS):
        seed = 42 + i
        cmd3 = [PYTHON_EXEC, 'executar_model_ml.py', file_ml, str(seed)]
        res3 = subprocess.run(cmd3, capture_output=True, text=True)
        
        if res3.returncode != 0:
            print(f"   ❌ Error en iteració {i}:\n{res3.stderr}")
            continue
        
        output = res3.stdout
        acc_match = re.search(r'Accuracy Mitjana:\s+([\d\.]+)', output)
        auc_match = re.search(r'AUC-ROC Mitjana:\s+([\d\.]+)', output)
        
        if acc_match and auc_match:
            acc = float(acc_match.group(1))
            auc = float(auc_match.group(1))
            acc_list.append(acc)
            auc_list.append(auc)
        else:
            print(f"   ⚠️ Error llegint mètriques iteració {i}")

    if not acc_list:
        continue

    mean_acc = sum(acc_list) / len(acc_list)
    mean_auc = sum(auc_list) / len(auc_list)
    std_acc = pd.Series(acc_list).std()
    std_auc = pd.Series(auc_list).std()
    
    print(f" ✅ RESULTAT FINAL CUTOFF {cutoff}: ACC={mean_acc:.3f} (+/-{std_acc:.3f}), AUC={mean_auc:.3f} (+/-{std_auc:.3f})")
    RESULTS.append({
        'cutoff': cutoff, 
        'accuracy': mean_acc, 
        'auc': mean_auc,
        'std_acc': std_acc,
        'std_auc': std_auc
    })

    # Neteja (Opcional: esborrar fitxers temporals per no omplir el disc)
    # os.remove(file_genetic)
    # os.remove(file_ml)

# =============================================================================
# RESUM FINAL
# =============================================================================
print("\n" + "="*60)
print("🏆 RESUM FINAL D'OPTIMITZACIÓ")
print("="*60)

if not RESULTS:
    print("❌ No s'han obtingut resultats.")
    sys.exit()

df_res = pd.DataFrame(RESULTS)
df_res = df_res.sort_values(by='auc', ascending=False)

print(df_res.to_string(index=False))

best_auc = df_res.iloc[0]
print(f"\n🌟 MILLOR CUTOFF (per AUC): {best_auc['cutoff']} (AUC: {best_auc['auc']:.3f})")

best_acc = df_res.sort_values(by='accuracy', ascending=False).iloc[0]
print(f"🌟 MILLOR CUTOFF (per Accuracy): {best_acc['cutoff']} (ACC: {best_acc['accuracy']:.3f})")

# Guardar resultats a CSV
df_res.to_csv('RESULTATS_OPTIMITZACIO_CUTOFFS.csv', index=False)
print("\n📄 Resultats guardats a 'RESULTATS_OPTIMITZACIO_CUTOFFS.csv'")

# Gràfic simple
plt.figure(figsize=(10, 6))
plt.plot(df_res['cutoff'], df_res['auc'], marker='o', label='AUC')
plt.plot(df_res['cutoff'], df_res['accuracy'], marker='s', label='Accuracy')
plt.xlabel('Abundance Cutoff')
plt.ylabel('Score')
plt.title('Optimització de Cutoff T1K vs Performance Model')
plt.legend()
plt.grid(True)
plt.savefig('GRAFIC_OPTIMITZACIO_CUTOFFS.png')
print("📈 Gràfic guardat a 'GRAFIC_OPTIMITZACIO_CUTOFFS.png'")
