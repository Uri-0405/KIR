# KIR Analysis Project - Predict2Protect

Aquest repositori conté el codi per a l'anàlisi de dades genètiques KIR i la predicció de remissió en pacients amb LMC utilitzant XGBoost i SHAP.

## 📂 Estructura del Projecte

*   **`optimize_cutoffs.py`**: Script principal per trobar el millor punt de tall (cutoff) d'abundància.
*   **`executar_model_ml.py`**: Entrena el model XGBoost i genera gràfics d'explicabilitat (SHAP).
*   **`generar_taula_neta.py`**: Neteja les dades brutes de T1K aplicant el cutoff.
*   **`generar_taula_ML.py`**: Fusiona les dades genètiques amb les dades clíniques.
*   **`audit_model.py`**: Auditoria de seguretat per evitar Data Leakage.

## 🏆 Resultats Clau

Després d'una optimització exhaustiva (260 iteracions), s'ha determinat que el millor paràmetre de neteja és:

*   **Cutoff d'Abundància:** **68**
*   **AUC Mitjana:** 0.699
*   **Accuracy Mitjana:** 0.644

Aquest cutoff elimina el soroll de seqüenciació i maximitza la capacitat predictiva del model.

## 🚀 Com executar

1.  Instal·lar dependències:
    ```bash
    pip install -r requirements.txt
    ```

2.  Executar l'optimització (opcional):
    ```bash
    python optimize_cutoffs.py
    ```

3.  Executar el model final amb el cutoff òptim (68):
    ```bash
    # 1. Generar taula neta
    python generar_taula_neta.py 68 RESULTATS_NETS_68.xlsx
    
    # 2. Fusionar amb clínic
    python generar_taula_ML.py RESULTATS_NETS_68.xlsx Taula_ML_68.xlsx
    
    # 3. Entrenar model
    python executar_model_ml.py Taula_ML_68.xlsx
    ```

## ⚠️ Dades

Les dades clíniques i genètiques brutes no s'inclouen en aquest repositori per privacitat.
