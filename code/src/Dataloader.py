"""
Pkg3: UHRI data loader [last updated 2026-04-18]

Reads OHCHR UHRI exports (xlsx or json format) and returns a list of UHRIRecord objects. 
Mimics the small quirks of the export like "Reccomending" typo in OHCHR's column name.

Usage:
    from src.Dataloader import load_records
    records = load_records("../data/search-9_3_2026_592.xlsx")

Open Issues:
    - Only reads 5 of the 16 export columns
    - if adding themes etc more cleaning of the text
    - no error handling on the lstrip chain

Resumption:
    When widening UHRIRecord + this loader update
    uhri_record.py in the same commit, they have to stay in sync.
"""

from pathlib import Path
import json
import pandas as pd
from src.uhri_record import UHRIRecord
 
def load_records(path: str | Path) -> list[UHRIRecord]:
    """
    Load all records from an OHCHR export file.

    Args:
        path: Path to the export file. Suffix determines parsing mode:
              ".xlsx" -> read with pandas.read_excel,
              ".json" -> load as a JSON array of row dicts.

    Returns:
        A list of UHRIRecord, one per row in the export, in file order.
    """
    path = Path(path)
 
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, dtype=str)
    elif path.suffix == ".json":
        df = pd.DataFrame(json.loads(path.read_text()))
    else:
        raise ValueError("unsupported type")
 
    records = []
    for _, row in df.iterrows():
        records.append(UHRIRecord(
            annotation_id=row.get("OHCHR Annotation Id", ""),
            text=row.get("Text", ""),
            country=row.get("Countries Concerned").strip().lstrip("- "),
            recommending_body=row.get("Reccomending Body").strip().lstrip("- "),
            type=row.get("Type").strip().lstrip("- "),
        ))
 
    print(f"Loaded {len(records)} records from {path.name}")
    return records