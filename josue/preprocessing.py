import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ICD-9 CODE GROUPING
def map_icd9_to_category(code):
    """
    Map an ICD-9 diagnosis code to one of 9 broad clinical categories.
    """
    if pd.isna(code):
        return "Unknown"
    code = str(code).strip()

    # V-codes: supplementary classification
    if code.upper().startswith("V"):
        return "Supplementary"

    # E-codes: external causes of injury/poisoning
    if code.upper().startswith("E"):
        return "External Causes"

    try:
        num = float(code)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 250 <= num < 251:
        return "Diabetes"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif 800 <= num <= 999:
        return "Injury"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    else:
        return "Other"



# MISSING VALUE REPORT
def report_missing(df):
    """
    Print a summary of which features contain missing values and their percentages.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    if report.empty:
        print("No missing values found.")
    else:
        print("\n--- Missing Value Report (after replacing '?' with NaN) ---")
        print(report.to_string())
        print()
    return report


# MAIN PREPROCESSING PIPELINE
def load_and_preprocess(path, random_state=42, apply_smote=True, verbose=True):
    """
    Full preprocessing pipeline for the Diabetes 130-US Hospitals dataset.
    """
    # 1. Load
    df = pd.read_csv(path)
    # Replace '?' placeholders with NaN; suppress pandas 2.x downcasting warning
    pd.set_option("future.no_silent_downcasting", True)
    df = df.replace("?", np.nan).infer_objects(copy=False)
    if verbose:
        print(f"[1] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. Missing value report
    if verbose:
        report_missing(df)

    # 3. Remove non-readmission-eligible discharges
    before = len(df)
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])].copy()
    if verbose:
        print(f"[3] Removed hospice/expired records: {before - len(df):,} rows dropped → {len(df):,} remain")

    # 4. Deduplicate: keep first encounter per patient
    before = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first").copy()
    if verbose:
        print(f"[4] Removed duplicate patient encounters: {before - len(df):,} rows dropped → {len(df):,} remain")

    # 5. Target variable binarization: readmitted <30 days → 1, else 0
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    if verbose:
        counts = df["readmitted"].value_counts()
        pct_pos = counts[1] / len(df) * 100
        print(f"[5] Target distribution → 0 (not readmitted <30d): {counts[0]:,} | 1 (readmitted <30d): {counts[1]:,} ({pct_pos:.1f}%) — class imbalance confirmed")

    # 6. Drop high-missingness / ID columns
    drop_cols = ["weight", "payer_code", "medical_specialty", "encounter_id", "patient_nbr"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    if verbose:
        print(f"[6] Dropped high-missingness & ID columns: {drop_cols}")

    # 7. Impute race (mode) — only ~2% missing; dropping would waste data
    if df["race"].isnull().any():
        mode_race = df["race"].mode()[0]
        df["race"] = df["race"].fillna(mode_race)
        if verbose:
            print(f"[7] Imputed 'race' missing values with mode: '{mode_race}'")

    # 8. Diagnosis code grouping (HIGH CARDINALITY FIX)
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            n_unique_before = df[col].nunique()
            df[col] = df[col].apply(map_icd9_to_category)
            n_unique_after = df[col].nunique()
            if verbose:
                print(f"[8] {col}: {n_unique_before} unique codes → {n_unique_after} clinical categories")

    # 9. One-hot encode all remaining categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    if verbose:
        print(f"[9] One-hot encoded {len(categorical_cols)} categorical columns → {df.shape[1]} total features")

    # 10. Train/test split — stratify to maintain class balance in both sets
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    if verbose:
        print(f"[10] Train/test split → Train: {X_train.shape}, Test: {X_test.shape}")

    # 11. Feature scaling — ONLY on continuous numerical features
    num_cols = [c for c in X_train.columns if X_train[c].nunique() > 2]
    bin_cols = [c for c in X_train.columns if X_train[c].nunique() <= 2]

    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )
    X_test_num = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols, index=X_test.index
    )

    X_train = pd.concat([X_train_num, X_train[bin_cols]], axis=1)
    X_test  = pd.concat([X_test_num,  X_test[bin_cols]],  axis=1)

    if verbose:
        print(f"[11] Scaled {len(num_cols)} continuous features; left {len(bin_cols)} binary features unscaled")

    # 12. SMOTE — oversample the minority class in the training set only
    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_state)
            X_train_arr, y_train_arr = sm.fit_resample(X_train, y_train)
            X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
            y_train = pd.Series(y_train_arr, name="readmitted")
            if verbose:
                counts = pd.Series(y_train).value_counts()
                print(f"[12] SMOTE applied → balanced training set: {counts.to_dict()}")
        except ImportError:
            if verbose:
                print("[12] imbalanced-learn not installed — skipping SMOTE. Install with: pip install imbalanced-learn")
            if verbose:
                print("     Using class_weight='balanced' in the model instead.")
    else:
        if verbose:
            print("[12] SMOTE skipped (apply_smote=False). Using class_weight='balanced' in model.")

    return X_train, X_test, y_train, y_test, scaler, feature_names
