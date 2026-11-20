"""Build a non-personal-data question catalog from the supplied training CSV."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'training_data.csv'
OUT = Path(__file__).resolve().parent / 'catalog.json'


def main():
    if not CSV.exists():
        raise FileNotFoundError(f'Expected training data at {CSV}')
    df = pd.read_csv(CSV, low_memory=False)
    category_fields = ['Employment_Type','Product_Code']
    catalog = {field: sorted(df[field].dropna().astype(str).str.strip().unique().tolist())
               for field in category_fields}
    OUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {OUT} with {len(catalog["Employment_Type"])} employment and '
          f'{len(catalog["Product_Code"])} vehicle-type choices.')


if __name__ == '__main__':
    main()
