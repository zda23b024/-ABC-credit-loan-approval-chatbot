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
    category_fields = ['Qualifications','Employment_Type','Product_Code','Resident_Type','Final_Tier',
                       'PAST_LOANS_ACTIVE']
    catalog = {field: sorted(df[field].dropna().astype(str).str.strip().unique().tolist())
               for field in category_fields}
    vehicle_fields = ['Model_Description','Product_Code','Make_Code','Model_Variant']
    vehicle_map = df[vehicle_fields].dropna().drop_duplicates()
    counts = vehicle_map.groupby('Model_Description').size()
    if (counts > 1).any():
        raise ValueError('A vehicle description maps to multiple model fields; resolve before use.')
    catalog['vehicles'] = [
        {'description': str(row.Model_Description), 'product_code': str(row.Product_Code),
         'make_code': str(row.Make_Code), 'model_variant': str(row.Model_Variant)}
        for row in vehicle_map.sort_values('Model_Description').itertuples(index=False)
    ]
    OUT.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Wrote {OUT} with {len(catalog["vehicles"])} vehicle choices.')


if __name__ == '__main__':
    main()
