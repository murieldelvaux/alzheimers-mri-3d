# Alzheimer MRI 3D - PyTorch + MONAI

Projeto de pesquisa (mestrado) para diagnóstico de Alzheimer a partir de imagens de ressonância magnética (MRI) 3D, utilizando modelos de deep learning com PyTorch e MONAI.

## Objetivo

- Desenvolver e avaliar modelos 3D (CNN/Transformers) para classificação de estágios cognitivos (ex.: CN vs MCI vs AD).
- Utilizar bases públicas (ex.: ADNI, OASIS).
- Garantir reprodutibilidade com configs versionadas e código organizado.

## Estrutura

- `src/alz_mri/`: código principal (data, modelos, treinamento)
- `configs/`: arquivos YAML com hiperparâmetros e caminhos
- `notebooks/`: EDA e experimentos exploratórios
- `scripts/`: scripts de linha de comando (treino, preprocessamento)
- `data/`: referência para dados locais (não versionados)

## Como rodar

Em breve: `python scripts/train_3d_baseline.py --config configs/default.yaml`
