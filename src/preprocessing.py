import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop columns with too many missing values
    df.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove hospice/death discharge cases
    df = df[~df['discharge_disposition_id'].isin([11, 19, 20, 21])]

    # Target variable
    df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

    # Separate features
    X = df.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])
    y = df['readmitted']

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)