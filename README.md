# Alzheimer MRI 3D - PyTorch + MONAI

Projeto de pesquisa (mestrado) para diagnóstico de Alzheimer a partir de imagens de ressonância magnética 3D, organizado em um layout de dados e código reutilizável.

## Objetivo

- Desenvolver e avaliar modelos 3D para classificação cognitiva (por exemplo, CN vs MCI vs AD).
- Usar bases públicas como ADNI e OASIS.
- Garantir reprodutibilidade com configuração centralizada e código modular.

## Estrutura do projeto

- `data/adni/raw/`: arquivos ADNI originais `.nii` / `.nii.gz`
- `data/adni/preprocessed/`: scans ADNI pré-processados
- `data/adni/adni_metadata.csv`: metadados clínicos/demográficos ADNI
- `data/oasis/raw/`: arquivos OASIS originais `.nii` / `.nii.gz`
- `data/oasis/preprocessed/`: scans OASIS pré-processados
- `data/oasis/oasis_metadata.csv`: metadados clínicos/demográficos OASIS
- `data/unified_metadata.csv`: metadados unificados para ambos os datasets

- `src/data_prep/`: preparação de dados e geração de metadados
- `src/datasets/`: classes de dataset PyTorch para MRI
- `src/models/`: arquiteturas 3D (CNNs, etc.)
- `src/utils/`: utilitários gerais
- `src/train.py`: script de treinamento
- `src/evaluate.py`: script de avaliação

- `config.yaml`: configuração central de caminhos e hiperparâmetros
- `requirements.txt`: dependências Python

## Como rodar

Treino:

```bash
python src/train.py --config config.yaml
```

Avaliação:

```bash
python src/evaluate.py --config config.yaml --split val
```
