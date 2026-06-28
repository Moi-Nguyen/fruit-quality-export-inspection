import csv
from pathlib import Path
from collections import Counter

files = [
    Path("outputs/features/train_features.csv"),
    Path("outputs/features/test_features.csv"),
]

for file_path in files:
    print("\n" + "=" * 80)
    print(f"Checking: {file_path}")

    if not file_path.exists():
        print("ERROR: File not found")
        continue

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fieldnames = reader.fieldnames or []

    print(f"Rows: {len(rows)}")
    print(f"Columns: {len(fieldnames)}")

    print("\nFirst 10 columns:")
    print(fieldnames[:10])

    class_counts = Counter(row["class_name"] for row in rows)
    fruit_counts = Counter(row["fruit_type"] for row in rows)
    quality_counts = Counter(row["quality"] for row in rows)

    print("\nClass counts:")
    for k, v in sorted(class_counts.items()):
        print(f"  {k}: {v}")

    print("\nFruit type counts:")
    for k, v in sorted(fruit_counts.items()):
        print(f"  {k}: {v}")

    print("\nQuality counts:")
    for k, v in sorted(quality_counts.items()):
        print(f"  {k}: {v}")

    numeric_checks = [
        "area",
        "perimeter",
        "circularity",
        "aspect_ratio",
        "mask_area_ratio",
        "mean_r",
        "mean_g",
        "mean_b",
        "brightness",
        "contrast",
        "noise_level",
        "defect_ratio",
    ]

    print("\nNumeric feature ranges:")
    for col in numeric_checks:
        if col not in fieldnames:
            print(f"  MISSING: {col}")
            continue

        values = []
        bad_values = 0

        for row in rows:
            try:
                values.append(float(row[col]))
            except Exception:
                bad_values += 1

        if values:
            print(
                f"  {col}: min={min(values):.4f}, "
                f"max={max(values):.4f}, "
                f"bad={bad_values}"
            )
        else:
            print(f"  {col}: no numeric values")

    print("\nInvalid value checks:")

    def count_invalid(col, condition):
        if col not in fieldnames:
            return "MISSING"

        count = 0
        for row in rows:
            try:
                value = float(row[col])
                if condition(value):
                    count += 1
            except Exception:
                count += 1
        return count

    print("  area <= 0:", count_invalid("area", lambda x: x <= 0))
    print("  perimeter <= 0:", count_invalid("perimeter", lambda x: x <= 0))
    print("  circularity outside [0,1]:", count_invalid("circularity", lambda x: x < 0 or x > 1))
    print("  mask_area_ratio outside [0,1]:", count_invalid("mask_area_ratio", lambda x: x < 0 or x > 1))
    print("  defect_ratio outside [0,1]:", count_invalid("defect_ratio", lambda x: x < 0 or x > 1))
