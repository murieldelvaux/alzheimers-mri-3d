import pandas as pd
import numpy as np
import os

# === PATHS ===
CDR_PATH = "data/oasis/raw/clinical/scans/UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/OASIS3_UDSb4_cdr.csv"
DX_PATH  = "data/oasis/raw/clinical/scans/UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia/resources/csv/files/OASIS3_UDSd1_diagnoses.csv"
OUT_PATH = "data/oasis/oasis_clinical_clean.csv"

os.makedirs("data/oasis", exist_ok=True)

# === MAPEAMENTO dx1 → label ===
CN_VALUES = [
    "Cognitively normal", "No dementia",
    "0.5 in memory only", "Unc: ques. Impairment",
    "Unc: impair reversible", "ProAph w/o dement"
]
MCI_VALUES = [
    "uncertain dementia", "Incipient demt PTP",
    "Incipient Non-AD dem", "uncertain, possible NON AD dem"
]
# Tudo que não for CN ou MCI e não for "." ou "Q" → DEM

LABEL_MAP = {
    0: "CN",
    1: "MCI",
    2: "DEM"
}

def map_dx1_to_label(dx1):
    if pd.isna(dx1) or dx1 in [".", "Q"]:
        return None
    if dx1 in CN_VALUES:
        return 0
    if dx1 in MCI_VALUES:
        return 1
    # qualquer outro valor com conteúdo = DEM
    return 2

# === CARREGAR UDSb4 (CDR + dx1) ===
print("Carregando UDSb4...")
cdr = pd.read_csv(CDR_PATH)
print(f"  Linhas: {len(cdr)}")

# Normalizar IDs
cdr["OASISID"] = cdr["OASISID"].str.strip().str.upper()
cdr["OASIS_session_label"] = cdr["OASIS_session_label"].str.strip()

# Aplicar mapeamento
cdr["label_id"] = cdr["dx1"].apply(map_dx1_to_label)
cdr["label_name"] = cdr["label_id"].map(LABEL_MAP)

# Remover sem diagnóstico
before = len(cdr)
cdr = cdr.dropna(subset=["label_id"])
print(f"  Removidas {before - len(cdr)} linhas sem diagnóstico válido")

# Converter days_to_visit para int
cdr["days_to_visit"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
cdr = cdr.dropna(subset=["days_to_visit"])
cdr["days_to_visit"] = cdr["days_to_visit"].astype(int)

# visit_month e is_baseline (equivalente ao VISCODE2 da ADNI)
cdr["visit_month"] = (cdr["days_to_visit"] / 30).round().astype(int)
cdr["is_baseline"] = cdr["days_to_visit"] == cdr.groupby("OASISID")["days_to_visit"].transform("min")

# Remover duplicatas
cdr = cdr.sort_values(["OASISID", "days_to_visit"])
cdr = cdr.drop_duplicates(subset=["OASISID", "days_to_visit"], keep="last")

# === RENOMEAR para ficar igual à ADNI ===
cdr = cdr.rename(columns={
    "OASISID": "PTID",
    "OASIS_session_label": "VISCODE2",
    "days_to_visit": "days_to_visit",  # mantém original também
    "age at visit": "AGE"
})

# Selecionar colunas relevantes
cols = ["PTID", "VISCODE2", "days_to_visit", "AGE",
        "CDRTOT", "dx1", "label_id", "label_name",
        "visit_month", "is_baseline"]
cdr = cdr[cols]

# === SALVAR ===
cdr.to_csv(OUT_PATH, index=False)
print(f"\n✅ Salvo em {OUT_PATH}")
print(f"   Total de registros: {len(cdr)}")
print(f"\nDistribuição de labels:")
print(cdr["label_name"].value_counts())
print(f"\nPor dataset de visita (baseline vs follow-up):")
print(cdr["is_baseline"].value_counts())
