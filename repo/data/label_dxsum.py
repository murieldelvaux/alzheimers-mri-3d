#!/usr/bin/env python3
import argparse
import os
import pandas as pd

LABEL_MAP = {1: "CN", 2: "MCI", 3: "DEM"}   # DEM = demência (proxy de AD demência)
LABEL_ID = {"CN": 0, "MCI": 1, "DEM": 2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    # Tipos básicos
    if "EXAMDATE" in df.columns:
        df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df["DIAGNOSIS"] = pd.to_numeric(df["DIAGNOSIS"], errors="coerce")

    # Labels
    df["label_name"] = df["DIAGNOSIS"].map(LABEL_MAP)

    # Mantém só linhas válidas para treino/merge
    needed = ["PTID", "DIAGNOSIS", "label_name"]
    if "VISCODE2" in df.columns: needed.append("VISCODE2")
    if "EXAMDATE" in df.columns: needed.append("EXAMDATE")
    df = df.dropna(subset=["PTID", "label_name"]).copy()

    df["label_id"] = df["label_name"].map(LABEL_ID).astype(int)

    # Ordena (útil pra auditoria)
    sort_cols = [c for c in ["PTID", "EXAMDATE"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Rows (visits):", len(df))
    print("Unique PTID:", df["PTID"].nunique())
    print("Label counts:\n", df["label_name"].value_counts())

if __name__ == "__main__":
    main()