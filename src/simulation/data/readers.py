import logging

import numpy as np
import pandas as pd

from src.simulation.data.utils import read_dataset
from src.simulation.data.partitioning import partition_nodes, build_tabular_client_data
from scipy.linalg import qr
from src.simulation.config import DATASET_DIR

dataset_dir = DATASET_DIR / "tabular"

logger = logging.getLogger(__name__)

def _drop_redundant_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated columns and exact linear dependencies (survey skip codes 8/9)."""
    X = X.loc[:, ~X.T.duplicated()]
    Xn = np.hstack([np.ones((len(X), 1)), X.to_numpy(float)])
    _, R, P = qr(Xn, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    r = int((d > d[0] * 1e-10).sum())
    dependent = sorted(p - 1 for p in P[r:] if p > 0)   # p = 0 is the intercept
    return X.drop(columns=X.columns[dependent])

def read_discharges(target, n_nodes):

    # SPARCS 2015 inpatient discharges (2,346,760 samples)
    df = read_dataset(dataset_dir, "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.parquet")

    # Numbers stored as text (pandas >= 3 strict strings): convert BEFORE anything else
    df['Length of Stay'] = pd.to_numeric(df['Length of Stay'].replace('120 +', '120'))
    for col in ["Total Charges", "Total Costs"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str)
                                    .str.replace("$", "", regex=False)
                                    .str.replace(",", "", regex=False))

    # Drop identifiers, high-cardinality codes and Total Costs (near-duplicate of the target)
    df2 = df.drop(columns=['Health Service Area', 'Hospital County',
        'Operating Certificate Number', 'Facility Name',
        'Zip Code - 3 digits',
        'Type of Admission', 'Patient Disposition',
        'Discharge Year', 'CCS Diagnosis Code', 'CCS Diagnosis Description',
        'CCS Procedure Code', 'CCS Procedure Description', 'APR DRG Code',
        'APR DRG Description', 'APR MDC Code', 'APR MDC Description',
        'APR Severity of Illness Code', 'APR Severity of Illness Description',
        'APR Risk of Mortality', 'APR Medical Surgical Description',
        'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3',
        'Attending Provider License Number',
        'Operating Provider License Number', 'Other Provider License Number',
        'Birth Weight', 'Abortion Edit Indicator',
        'Emergency Department Indicator', 'Total Costs'])

    df2 = df2.dropna()

    # Facility Id defines the nodes (partition), it is NOT a predictor
    facility = df2.pop("Facility Id")

    cat_cols = [c for c in df2.columns if c != target and not pd.api.types.is_numeric_dtype(df2[c])]
    df3 = pd.get_dummies(df2, columns=cat_cols, drop_first=True, dtype=float).dropna()

    y = df3[target]
    X = df3.drop(columns=[target])
    
    # Hospital of each remaining row (same index as X, taken from the original df)
    groups = facility.loc[X.index]
    # One node per hospital: the n_nodes largest hospitals
    node_data = partition_nodes(X, y, n_nodes, groups=groups)


    data, n_nodes = build_tabular_client_data(node_data)
    logger.info(f"[Dataset] discharges: {len(data)} nodes, {X.shape[1]} features")
    return data, n_nodes, None

def read_insurance(target, n_nodes):
    # Insurance dataset
    # 1,338 samples, 7 features. Possible target variable: "charges"
    df = read_dataset(dataset_dir, "insurance.parquet")
    df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
    df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
    df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
    X = df.drop(columns=[target])
    y = df[target]

    features_path = None

    node_data = partition_nodes(X, y, n_nodes)
    data, n_nodes = build_tabular_client_data(node_data)

    return data, n_nodes, features_path

def read_housing(target, n_nodes):
    # Housing dataset
    # 20,640 samples, 10 features. Possible target variable: "latitude"
    df = read_dataset(dataset_dir, "housing.parquet").dropna()
    y = df[target]
    X = df.drop(columns=[target])

    X = pd.get_dummies(X, drop_first=True)
    
    features_path = None

    node_data = partition_nodes(X, y, n_nodes)
    data, n_nodes = build_tabular_client_data(node_data)

    return data, n_nodes, features_path

def read_european_survey_adult(target, n_nodes):
    # EESEa dataset; one node per autonomous community (CCAA)
    df = read_dataset(dataset_dir, "eesea.parquet")
    # CMD1 is a weighted average of CMD2 and CMD3 (derived variable)
    X = df.drop(columns=[target, "CCAA", "CMD1"], errors="ignore")
    X = _drop_redundant_columns(X)
    y = df[target]
    logger.info(f"[Dataset] EESE: {X.shape[1]} features after removing redundant columns")

    node_data = partition_nodes(X, y, n_nodes, groups=df["CCAA"])
    data, n_nodes = build_tabular_client_data(node_data)

    features_path = dataset_dir / "tabular" / "fs" / "eesea_fs.json"
    return data, n_nodes, features_path

def read_lengthofstay(target, n_nodes):
    # Length of Stay dataset; one node per facility
    df = read_dataset(dataset_dir, "df_lengthofstay.parquet")
    # eid is a record id and `discharged` is a date (with vdate it encodes the stay length):
    # both are dropped to avoid noise and target leakage
    drop = [c for c in (target, "vdate", "facid", "eid", "discharged") if c in df.columns]
    X = pd.get_dummies(df.drop(columns=drop), drop_first=True, dtype=float)
    y = df[target]

    node_data = partition_nodes(X, y, n_nodes, groups=df["facid"])
    data, n_nodes = build_tabular_client_data(node_data)

    features_path = dataset_dir / "tabular" / "fs" / "lengthofstay_fs.json"
    return data, n_nodes, features_path

def read_anxiety(target, n_nodes):
    # Anxiety dataset
    # 11000 samples, 19 features. Possible target variable: "Sleep Hours"
    df = read_dataset(dataset_dir, "enhanced_anxiety_dataset.parquet")
    X = df.drop(columns=[target])
    y = df[target]
    features_path = dataset_dir / "tabular" / "features" / "selected_rf_anxiety.json"

    node_data = partition_nodes(X, y, n_nodes)
    data, n_nodes = build_tabular_client_data(node_data)

    return data, n_nodes, features_path