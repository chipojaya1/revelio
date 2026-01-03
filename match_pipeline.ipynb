+#!/usr/bin/env python3
+"""Partition and match Revelio resumes to L2 voter data.
+
+This script follows the matching steps described in the project instructions:
+1) Extract required columns from L2 VM2Uniform files, add year, map ZIP to MSA,
+   clean last names, and partition by last-name initial.
+2) Extract required columns from Revelio (position + user + education),
+   derive birth year from graduation year, clean last names, and partition by
+   state + last-name initial.
+3) Merge partition pairs by cleaned first/last name, apply party-resolution
+   rules, and output user_id -> party matches.
+
+The script is designed for large datasets and uses chunked reads/writes.
+"""
+
+from __future__ import annotations
+
+import argparse
+import re
+from dataclasses import dataclass
+from pathlib import Path
+from typing import Dict, List, Optional
+
+import pandas as pd
+from tqdm import tqdm
+
+
+LASTNAME_BUCKETS = [chr(code) for code in range(ord("a"), ord("z") + 1)] + ["other"]
+
+
+def clean_name(value: str) -> str:
+    if pd.isna(value):
+        return ""
+    value = str(value).lower()
+    value = re.sub(r"[^a-z]", "", value)
+    return value
+
+
+def lastname_bucket(clean_lastname: str) -> str:
+    if not clean_lastname:
+        return "other"
+    first = clean_lastname[0]
+    return first if first in LASTNAME_BUCKETS else "other"
+
+
+def extract_year_from_filename(filename: str) -> Optional[int]:
+    match = re.search(r"(19|20)\d{2}", filename)
+    return int(match.group(0)) if match else None
+
+
+def ensure_dir(path: Path) -> None:
+    path.mkdir(parents=True, exist_ok=True)
+
+
+def load_zip_to_msa(zip_to_msa_path: Optional[Path]) -> Dict[str, str]:
+    if not zip_to_msa_path:
+        return {}
+    mapping = pd.read_csv(zip_to_msa_path, dtype={"zip": "string"})
+    if "zip" not in mapping.columns or "msa" not in mapping.columns:
+        raise ValueError("ZIP-to-MSA file must include 'zip' and 'msa' columns.")
+    return dict(zip(mapping["zip"].str.zfill(5), mapping["msa"]))
+
+
+@dataclass
+class L2Config:
+    input_path: Path
+    output_dir: Path
+    zip_to_msa_path: Optional[Path]
+    chunksize: int
+
+
+@dataclass
+class RevelioConfig:
+    position_path: Path
+    user_path: Path
+    education_path: Path
+    output_dir: Path
+    grad_year_column: str
+    chunksize: int
+
+
+@dataclass
+class MatchConfig:
+    l2_partitions_dir: Path
+    revelio_partitions_dir: Path
+    output_file: Path
+    min_party_share: float
+
+
+def _write_partition(
+    base_dir: Path, state: str, bucket: str, frame: pd.DataFrame
+) -> None:
+    target_dir = base_dir / state / bucket
+    ensure_dir(target_dir)
+    file_path = target_dir / "data.csv"
+    frame.to_csv(file_path, mode="a", index=False, header=not file_path.exists())
+
+
+def partition_l2(config: L2Config) -> None:
+    zip_to_msa = load_zip_to_msa(config.zip_to_msa_path)
+    input_path = config.input_path
+    year = extract_year_from_filename(input_path.name)
+    if year is None:
+        raise ValueError("Could not infer year from L2 filename. Provide a year in the name.")
+
+    def usecols(col: str) -> bool:
+        base_cols = {
+            "LALVOTERID",
+            "Voters_FirstName",
+            "Voters_LastName",
+            "Residence_Addresses_Zip",
+            "Voters_Gender",
+            "Voters_Age",
+            "County",
+            "Parties_Description",
+            "Residence_Addresses_State",
+        }
+        return col in base_cols or col.startswith("PRI_BLT")
+
+    reader = pd.read_csv(
+        input_path,
+        sep="\t",
+        dtype="string",
+        chunksize=config.chunksize,
+        usecols=usecols,
+        low_memory=False,
+    )
+
+    for chunk in tqdm(reader, desc=f"Partitioning L2 {input_path.name}"):
+        chunk["clean_lastname"] = chunk["Voters_LastName"].apply(clean_name)
+        chunk["clean_firstname"] = chunk["Voters_FirstName"].apply(clean_name)
+        chunk["bucket"] = chunk["clean_lastname"].apply(lastname_bucket)
+        chunk["Year"] = str(year)
+        if zip_to_msa:
+            zip_series = chunk["Residence_Addresses_Zip"].fillna("").str.zfill(5)
+            chunk["MSA"] = zip_series.map(zip_to_msa)
+        else:
+            chunk["MSA"] = pd.NA
+
+        for (state, bucket), group in chunk.groupby(
+            ["Residence_Addresses_State", "bucket"], dropna=False
+        ):
+            if pd.isna(state):
+                continue
+            _write_partition(config.output_dir, str(state), str(bucket), group)
+
+
+def _derive_birth_year(education: pd.DataFrame, grad_year_column: str) -> pd.DataFrame:
+    if grad_year_column not in education.columns:
+        raise ValueError(
+            f"Education file missing expected column '{grad_year_column}'."
+        )
+    education = education.copy()
+    education[grad_year_column] = pd.to_numeric(education[grad_year_column], errors="coerce")
+    education = education.dropna(subset=[grad_year_column])
+    education["birth_year"] = education[grad_year_column] - 22
+    education = education[["user_id", "birth_year"]].drop_duplicates("user_id")
+    return education
+
+
+def partition_revelio(config: RevelioConfig) -> None:
+    users = pd.read_csv(config.user_path, dtype="string", low_memory=False)
+    education = pd.read_csv(config.education_path, dtype="string", low_memory=False)
+    birth_year = _derive_birth_year(education, config.grad_year_column)
+
+    positions_reader = pd.read_csv(
+        config.position_path,
+        dtype="string",
+        low_memory=False,
+        chunksize=config.chunksize,
+    )
+
+    for chunk in tqdm(positions_reader, desc="Partitioning Revelio"):
+        revelio = chunk.merge(
+            users[["user_id", "firstname", "lastname", "state", "F_prob"]],
+            on="user_id",
+            how="left",
+        ).merge(birth_year, on="user_id", how="left")
+
+        revelio["clean_lastname"] = revelio["lastname"].apply(clean_name)
+        revelio["clean_firstname"] = revelio["firstname"].apply(clean_name)
+        revelio["bucket"] = revelio["clean_lastname"].apply(lastname_bucket)
+
+        for (state, bucket), group in revelio.groupby(["state", "bucket"], dropna=False):
+            if pd.isna(state):
+                continue
+            _write_partition(config.output_dir, str(state), str(bucket), group)
+
+
+def _resolve_party(group: pd.DataFrame, min_share: float) -> Optional[str]:
+    party_counts = group["Parties_Description"].dropna().value_counts()
+    if party_counts.empty:
+        return None
+    if len(party_counts) == 1:
+        return party_counts.index[0]
+    share = party_counts.iloc[0] / party_counts.sum()
+    if share >= min_share:
+        return party_counts.index[0]
+    return None
+
+
+def _infer_gender_from_fprob(fprob: Optional[str]) -> Optional[str]:
+    if fprob is None or pd.isna(fprob):
+        return None
+    try:
+        prob = float(fprob)
+    except ValueError:
+        return None
+    return "F" if prob >= 0.5 else "M"
+
+
+def match_partitions(config: MatchConfig) -> None:
+    matches: List[pd.DataFrame] = []
+    l2_states = [p for p in config.l2_partitions_dir.iterdir() if p.is_dir()]
+    for state_dir in tqdm(l2_states, desc="Matching states"):
+        state = state_dir.name
+        revelio_state_dir = config.revelio_partitions_dir / state
+        if not revelio_state_dir.exists():
+            continue
+        for bucket in LASTNAME_BUCKETS:
+            l2_file = state_dir / bucket / "data.csv"
+            revelio_file = revelio_state_dir / bucket / "data.csv"
+            if not l2_file.exists() or not revelio_file.exists():
+                continue
+
+            l2_df = pd.read_csv(l2_file, dtype="string", low_memory=False)
+            revelio_df = pd.read_csv(revelio_file, dtype="string", low_memory=False)
+
+            merged = revelio_df.merge(
+                l2_df,
+                left_on=["clean_firstname", "clean_lastname"],
+                right_on=["clean_firstname", "clean_lastname"],
+                how="inner",
+                suffixes=("_rev", "_l2"),
+            )
+            if merged.empty:
+                continue
+
+            merged["rev_gender"] = merged["F_prob"].apply(_infer_gender_from_fprob)
+            merged["l2_birth_year"] = pd.to_numeric(
+                merged["Year"], errors="coerce"
+            ) - pd.to_numeric(merged["Voters_Age"], errors="coerce")
+            merged["rev_birth_year"] = pd.to_numeric(
+                merged["birth_year"], errors="coerce"
+            )
+
+            resolved_rows = []
+            for user_id, group in merged.groupby("user_id"):
+                party = _resolve_party(group, min_share=1.0)
+                if party:
+                    resolved_rows.append(group.iloc[0].assign(resolved_party=party))
+                    continue
+
+                age_filtered = group.copy()
+                age_filtered = age_filtered.dropna(subset=["l2_birth_year", "rev_birth_year"])
+                age_filtered = age_filtered[
+                    (age_filtered["l2_birth_year"] - age_filtered["rev_birth_year"]).abs()
+                    <= 2
+                ]
+
+                if age_filtered.empty:
+                    continue
+
+                rev_gender = age_filtered["rev_gender"].dropna().iloc[0] if not age_filtered["rev_gender"].dropna().empty else None
+                if rev_gender:
+                    age_filtered = age_filtered[
+                        (age_filtered["Voters_Gender"].isna())
+                        | (age_filtered["Voters_Gender"].str.upper() == rev_gender)
+                    ]
+
+                party = _resolve_party(age_filtered, min_share=config.min_party_share)
+                if party:
+                    resolved_rows.append(age_filtered.iloc[0].assign(resolved_party=party))
+
+            if resolved_rows:
+                matches.append(pd.DataFrame(resolved_rows))
+
+    if matches:
+        final = pd.concat(matches, ignore_index=True)
+        final.to_csv(config.output_file, index=False)
+    else:
+        pd.DataFrame().to_csv(config.output_file, index=False)
+
+
+def build_parser() -> argparse.ArgumentParser:
+    parser = argparse.ArgumentParser(description=__doc__)
+    subparsers = parser.add_subparsers(dest="command", required=True)
+
+    l2_parser = subparsers.add_parser("partition-l2", help="Partition L2 VM2Uniform data.")
+    l2_parser.add_argument("--input", required=True, type=Path, help="Path to L2 VM2Uniform file.")
+    l2_parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
+    l2_parser.add_argument("--zip-to-msa", type=Path, help="CSV with zip and msa columns.")
+    l2_parser.add_argument("--chunksize", type=int, default=500_000)
+
+    rev_parser = subparsers.add_parser("partition-revelio", help="Partition Revelio data.")
+    rev_parser.add_argument("--position", required=True, type=Path)
+    rev_parser.add_argument("--user", required=True, type=Path)
+    rev_parser.add_argument("--education", required=True, type=Path)
+    rev_parser.add_argument("--output-dir", required=True, type=Path)
+    rev_parser.add_argument("--grad-year-column", default="enddate", help="Graduation year column.")
+    rev_parser.add_argument("--chunksize", type=int, default=500_000)
+
+    match_parser = subparsers.add_parser("match", help="Match partitioned L2 and Revelio data.")
+    match_parser.add_argument("--l2-partitions", required=True, type=Path)
+    match_parser.add_argument("--revelio-partitions", required=True, type=Path)
+    match_parser.add_argument("--output", required=True, type=Path)
+    match_parser.add_argument("--min-party-share", type=float, default=0.9)
+
+    return parser
+
+
+def main() -> None:
+    parser = build_parser()
+    args = parser.parse_args()
+
+    if args.command == "partition-l2":
+        config = L2Config(
+            input_path=args.input,
+            output_dir=args.output_dir,
+            zip_to_msa_path=args.zip_to_msa,
+            chunksize=args.chunksize,
+        )
+        partition_l2(config)
+    elif args.command == "partition-revelio":
+        config = RevelioConfig(
+            position_path=args.position,
+            user_path=args.user,
+            education_path=args.education,
+            output_dir=args.output_dir,
+            grad_year_column=args.grad_year_column,
+            chunksize=args.chunksize,
+        )
+        partition_revelio(config)
+    elif args.command == "match":
+        config = MatchConfig(
+            l2_partitions_dir=args.l2_partitions,
+            revelio_partitions_dir=args.revelio_partitions,
+            output_file=args.output,
+            min_party_share=args.min_party_share,
+        )
+        ensure_dir(config.output_file.parent)
+        match_partitions(config)
+
+
+if __name__ == "__main__":
+    main()
