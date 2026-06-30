'''
Data Anonymisation- Drops direct identifiers (PATIENT NAME, PRINCIPAL MEMBER)
                    Hashes MEMBERSHIP NO so per-member patterns are trackable without exposing real IDs
                    Saves the clean version to data/claims_anonymised.csv
'''

import argparse
import hashlib
import os
import pandas as pd


def anonymize(input_path: str, output_path:str) -> None:
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Original DataFrame shape: , {len(df)} rows, {len(df.columns)} columns")

    # Drop direct identifiers
    cols_to_drop = ["PATIENT NAME", "PRINCIPAL MEMBER"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    # df.drop(columns=['PATIENT NAME', 'PRINCIPAL MEMBER'], inplace=True)
    print(f"Dropped columns: {existing}. New DataFrame shape: {len(df)} rows, {len(df.columns)} columns")
