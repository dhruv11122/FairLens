"""Data loader, validation, and binning service."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from app.schemas.audit import ColumnSummary, DatasetPreviewResponse, NumericBinConfig

def inspect_dataset(df: pd.DataFrame, dataset_id: str, filename: str) -> DatasetPreviewResponse:
    columns_summary: List[ColumnSummary] = []
    suggested_targets: List[str] = []
    suggested_sensitive: List[str] = []
    
    sensitive_keywords = ["gender", "sex", "age", "race", "ethnicity", "nationality", "marital", 
                          "education", "employment", "income", "region", "disability", "religion"]
    target_keywords = ["target", "label", "approved", "status", "churn", "attrition", "default", "class"]

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique())
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        
        # sample values (convert to JSON-serializable)
        samples = series.dropna().unique()[:5].tolist()
        clean_samples = [int(x) if isinstance(x, (np.integer, np.int64)) 
                         else float(x) if isinstance(x, (np.floating, np.float64)) 
                         else str(x) for x in samples]
        
        columns_summary.append(ColumnSummary(
            name=str(col),
            dtype=dtype_str,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=clean_samples,
            is_numeric=is_numeric
        ))
        
        col_lower = str(col).lower()
        if any(k in col_lower for k in target_keywords) or (unique_count == 2 and not is_numeric):
            suggested_targets.append(str(col))
        if any(k in col_lower for k in sensitive_keywords) or (2 <= unique_count <= 10 and not any(k in col_lower for k in target_keywords)):
            suggested_sensitive.append(str(col))

    head_records = df.head(5).replace({np.nan: None}).to_dict(orient="records")

    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        filename=filename,
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=columns_summary,
        head_rows=head_records,
        suggested_targets=suggested_targets,
        suggested_sensitive=suggested_sensitive
    )

def apply_numeric_binning(df: pd.DataFrame, bin_configs: Optional[List[NumericBinConfig]]) -> pd.DataFrame:
    """Applies quantile or uniform binning to continuous columns for fairness subgroup analysis."""
    if not bin_configs:
        return df
    
    df_copy = df.copy()
    for config in bin_configs:
        col = config.column
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            num_bins = max(2, min(config.bins, 10))
            try:
                if config.strategy == "quantile":
                    df_copy[col] = pd.qcut(
                        df_copy[col], 
                        q=num_bins, 
                        labels=config.labels if config.labels and len(config.labels) == num_bins else False,
                        duplicates="drop"
                    ).astype(str)
                else:
                    df_copy[col] = pd.cut(
                        df_copy[col], 
                        bins=num_bins, 
                        labels=config.labels if config.labels and len(config.labels) == num_bins else False
                    ).astype(str)
            except Exception:
                # Fallback to cut if qcut fails due to low variance
                df_copy[col] = pd.cut(df_copy[col], bins=num_bins).astype(str)
    return df_copy
