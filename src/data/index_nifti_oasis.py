import pandas as pd
import os

# === PATHS ===
CDR_PATH = "data/oasis/raw/clinical/scans/MRI-json-MRI_json_information/resources/csv/files/OASIS3_MR_json.csv"
CLIN_PATH = "data/oasis/oasis_clinical_clean.csv"
OUT_INDEX = "data/oasis/imaging_index_oasis.csv"
OUT_SESSIONS = "data/oasis/sessions_to_download.csv"

os.makedirs("data/oasis", exist_ok=True)

# === CARREGAR ===
mri  = pd.read_csv(CDR_PATH)
clin = pd.read_csv(CLIN_PATH)

# === FILTRAR T1w ===
t1 = mri[mri['filename'].str.contains('T1w', na=False)].copy()
print(f"T1w encontrados: {len(t1)}")

# Extrair days_to_visit do label (ex: OAS30096_MR_d2948 → 2948)
t1['days_to_visit'] = t1['label'].str.extract(r'_d(\d+)$')[0].astype(float)

# Extrair session do filename (ex: sub-OAS30096_sess-d2948_T1w.json → d2948)
t1['session'] = t1['label'].str.extract(r'_MR_(d\d+)$')[0]

# Montar path esperado da imagem .nii.gz no formato BIDS
t1['scan_path'] = (
    "data/oasis/raw/images/" +
    "sub-" + t1['subject_id'] + "/" +
    "ses-" + t1['session'] + "/anat/" +
    t1['filename'].str.replace('.json', '.nii.gz', regex=False)
)

# === MERGE COM CLINICAL ===
merged = t1.merge(clin, left_on='subject_id', right_on='PTID', how='inner')
merged = merged.rename(columns={'subject_id': 'OASISID'})

# Selecionar colunas relevantes
cols = [
    'OASISID', 'label', 'session', 'days_to_visit_x',
    'scan_path', 'filename',
    'label_id', 'label_name', 'visit_month', 'is_baseline'
]
merged = merged[cols].rename(columns={'days_to_visit_x': 'days_to_visit'})

# Remover duplicatas (manter primeira ocorrência por sujeito/sessão)
merged = merged.drop_duplicates(subset=['OASISID', 'session'])

print(f"\n✅ Index criado: {len(merged)} scans")
print(f"   Sujeitos únicos: {merged['OASISID'].nunique()}")
print(f"\nDistribuição de labels:")
print(merged['label_name'].value_counts())

# === SALVAR INDEX ===
merged.to_csv(OUT_INDEX, index=False)
print(f"\n✅ Salvo em {OUT_INDEX}")

# === GERAR LISTA DE SESSÕES PARA DOWNLOAD ===
# Formato esperado pelo script download_oasis_scans.sh
sessions = merged[['label']].drop_duplicates()
sessions.columns = ['experiment_id']
sessions.to_csv(OUT_SESSIONS, index=False)
print(f"✅ Lista de sessões salva em {OUT_SESSIONS}")
print(f"   Total de sessões para download: {len(sessions)}")
