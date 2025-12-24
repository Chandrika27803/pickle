"""

Single-run version:
- First run:
    * Trains a model from all CSVs in ./data
    * Saves the model to studentmodel.pkl
    * Immediately analyzes ALL CSVs and creates PDF reports
- Later runs:
    * Loads existing model
    * Analyzes ALL CSVs again and recreates reports

NO hard-coded target column:
- Auto-detects target:
    1) Column named 'target' (any case), else
    2) Last numeric column, else
    3) Last column
"""

import os
import glob
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------- CONFIG ----------
DATAFOLDER = "data"
MODELFILE = "studentmodel.pkl"
# ----------------------------


# ========== Helper: find CSVs ==========

def findcsvfiles(folder: str) -> List[str]:
    """Return a sorted list of CSV file paths in the folder."""
    pattern = os.path.join(folder, "*.csv")
    files = sorted(glob.glob(pattern))
    print(f"[INFO] CSV files found in '{folder}': {files}")
    return files


# ========== Helper: detect column types ==========

def detectcolumntypes(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect numeric and categorical columns."""
    numericcols = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricalcols = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    return {
        "numeric": numericcols,
        "categorical": categoricalcols,
    }


# ========== Helper: auto-detect target column ==========

def autodetecttarget(df: pd.DataFrame) -> str:
    """
    Automatically detect the target column.

    Priority:
      1. A column named 'target' (any case, trimmed)
      2. Last numeric column
      3. Last column in the DataFrame
    """
    # 1. Column literally called "target" (case-insensitive)
    for col in df.columns:
        if col.strip().lower() == "target":
            print(f"[AUTO] Found explicit 'target' column: {col}")
            return col

    # 2. Last numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        chosen = numeric_cols[-1]
        print(f"[AUTO] Using last numeric column as target: {chosen}")
        return chosen

    # 3. Fallback: last column
    chosen = df.columns[-1]
    print(f"[AUTO] Using last column as target: {chosen}")
    return chosen


# ========== Training phase ==========

def loadtrainingdata(csvfiles: List[str]) -> pd.DataFrame:
    """Load all CSVs and concatenate into one DataFrame."""
    dfs = []
    for path in csvfiles:
        df = pd.read_csv(path)
        print(f"[TRAIN] Loading {path} with columns: {list(df.columns)}")
        dfs.append(df)

    if not dfs:
        raise ValueError("No training CSVs found.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[TRAIN] Combined training shape: {combined.shape}")
    return combined


def buildandtrainmodel(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a model using auto-detected column types and target."""
    # --- Auto-detect target column ---
    targetcolumn = autodetecttarget(df)
    if targetcolumn not in df.columns:
        raise ValueError(f"Detected target column '{targetcolumn}' not found in DataFrame.")

    y = df[targetcolumn]

    # Detect types on full df
    coltypes = detectcolumntypes(df)

    numericfeatures = [c for c in coltypes["numeric"] if c != targetcolumn]
    categoricalfeatures = [c for c in coltypes["categorical"] if c != targetcolumn]

    featurecols = numericfeatures + categoricalfeatures
    if not featurecols:
        raise ValueError("No feature columns (numeric/categorical) found for training.")

    print(f"[TRAIN] Target column: {targetcolumn}")
    print(f"[TRAIN] Numeric features: {numericfeatures}")
    print(f"[TRAIN] Categorical features: {categoricalfeatures}")

    X = df[featurecols]

    # Drop rows with NaN in features or target
    mask = X.notna().all(axis=1) & y.notna()
    Xclean = X[mask]
    yclean = y[mask]

    print(f"[TRAIN] Rows before cleaning: {len(df)}, after cleaning: {len(Xclean)}")

    if Xclean.empty:
        raise ValueError("No rows left for training after dropping NaNs.")

    # Decide regression vs classification
    if np.issubdtype(yclean.dtype, np.number) and yclean.nunique() > 10:
        tasktype = "regression"
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        tasktype = "classification"
        # For classification, turn y into integer codes if needed
        if not np.issubdtype(yclean.dtype, np.number):
            yclean = yclean.astype("category").cat.codes
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    print(f"[TRAIN] Task type: {tasktype}")

    # Preprocessor: numeric passthrough, categorical one-hot
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numericfeatures),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricalfeatures),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(Xclean, yclean)

    print("[TRAIN] Model training complete.")

    modelinfo = {
        "pipeline": pipeline,
        "tasktype": tasktype,
        "numericfeatures": numericfeatures,
        "categoricalfeatures": categoricalfeatures,
        "targetcolumn": targetcolumn,
    }
    return modelinfo


def savemodel(modelinfo: Dict[str, Any]) -> None:
    """Save model info with pickle."""
    with open(MODELFILE, "wb") as f:
        pickle.dump(modelinfo, f)
    print(f"[MODEL] Saved model to {MODELFILE}")


def loadmodel() -> Dict[str, Any]:
    """Load model info with pickle."""
    with open(MODELFILE, "rb") as f:
        modelinfo = pickle.load(f)
    print(f"[MODEL] Loaded model from {MODELFILE}")
    return modelinfo


# ========== Reporting phase: PDF generation ==========

def generatepdfreport(
    df: pd.DataFrame,
    coltypes: Dict[str, List[str]],
    predictions: Optional[pd.Series],
    csvpath: str,
    tasktype: str,
    targetcolumn: str,
) -> None:
    """Create <csvname>report.pdf with overview + plots."""
    basename = os.path.splitext(os.path.basename(csvpath))[0]
    pdfname = f"{basename}report.pdf"

    numcols = coltypes["numeric"]
    catcols = coltypes["categorical"]

    hastarget = targetcolumn in df.columns
    ytrue = df[targetcolumn] if hastarget else None

    print(f"[REPORT] Generating {pdfname} (target present: {hastarget})")

    with PdfPages(pdfname) as pdf:
        # ---- Page 1: Overview ----
        fig = plt.figure(figsize=(8, 6))
        plt.axis("off")

        lines = [
            f"Report for file: {csvpath}",
            "",
            f"Task type: {tasktype}",
            f"Rows: {len(df)}",
            "",
            f"Target column (from training): {targetcolumn}",
            "",
            "Columns:",
            f"  Numeric     : {', '.join(numcols) if numcols else '(none)'}",
            f"  Categorical : {', '.join(catcols) if catcols else '(none)'}",
            "",
            f"Target column present in this file: {'Yes' if hastarget else 'No'}",
        ]
        text = "\n".join(lines)
        fig.text(0.05, 0.95, "Dataset Report", fontsize=16, weight="bold", va="top")
        fig.text(0.05, 0.80, text, fontsize=10, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Numeric histograms (up to 4) ----
        for col in numcols[:4]:
            fig = plt.figure(figsize=(8, 6))
            plt.hist(df[col].dropna(), bins=30)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Categorical bar charts (up to 3, top 10 categories) ----
        for col in catcols[:3]:
            fig = plt.figure(figsize=(8, 6))
            counts = df[col].value_counts().nlargest(10)
            plt.bar(counts.index.astype(str), counts.values)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Top categories in {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            pdf.savefig(fig)
            plt.close(fig)

        # ---- Prediction plots ----
        if predictions is not None:
            if tasktype == "regression" and hastarget and np.issubdtype(ytrue.dtype, np.number):
                # True vs Predicted
                fig = plt.figure(figsize=(8, 6))
                plt.scatter(ytrue, predictions, alpha=0.6)
                plt.xlabel("True Target")
                plt.ylabel("Predicted Target")
                plt.title("True vs Predicted (Regression)")
                pdf.savefig(fig)
                plt.close(fig)

                # Error distribution
                fig = plt.figure(figsize=(8, 6))
                errors = predictions - ytrue
                errors = pd.Series(errors).dropna()
                plt.hist(errors, bins=30)
                plt.title("Prediction Error Distribution")
                plt.xlabel("Error (pred - true)")
                plt.ylabel("Frequency")
                pdf.savefig(fig)
                plt.close(fig)
            else:
                # For classification or no true target:
                fig = plt.figure(figsize=(8, 6))
                predcounts = predictions.value_counts(dropna=True)
                plt.bar(predcounts.index.astype(str), predcounts.values)
                plt.title("Prediction Distribution")
                plt.xlabel("Predicted value / class")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[REPORT] PDF report created: {pdfname}")


# ========== Main logic ==========

def main():
    print("[INFO] Starting studentfolderanalyzer.py")
    os.makedirs(DATAFOLDER, exist_ok=True)
    print(f"[INFO] Ensured data folder exists: {DATAFOLDER}")

    csvfiles = findcsvfiles(DATAFOLDER)
    if not csvfiles:
        print(f"[WARN] No CSV files found in '{DATAFOLDER}' folder. Nothing to do.")
        return

    # ---- Train model if needed ----
    if not os.path.exists(MODELFILE):
        print("[INFO] No model found. Training a new model from all CSVs...")
        dftrain = loadtrainingdata(csvfiles)
        modelinfo = buildandtrainmodel(dftrain)
        savemodel(modelinfo)
    else:
        print("[INFO] Existing model found. Skipping training.")
        modelinfo = loadmodel()

    # ---- Analyze all CSVs and create reports ----
    print("[INFO] Analyzing all CSV files and generating reports...")
    pipeline = modelinfo["pipeline"]
    tasktype = modelinfo["tasktype"]
    numericfeatures = modelinfo["numericfeatures"]
    categoricalfeatures = modelinfo["categoricalfeatures"]
    targetcolumn = modelinfo["targetcolumn"]
    featurecols = numericfeatures + categoricalfeatures

    for path in csvfiles:
        print(f"\n[ANALYZE] Processing: {path}")
        df = pd.read_csv(path)
        print(f"[ANALYZE] Shape: {df.shape}")

        # Auto-detect types for THIS file (for report)
        coltypes = detectcolumntypes(df)
        print(f"[ANALYZE] Numeric cols: {coltypes['numeric']}")
        print(f"[ANALYZE] Categorical cols: {coltypes['categorical']}")

        # Make sure all feature columns exist in the new data
        for col in featurecols:
            if col not in df.columns:
                if col in numericfeatures:
                    print(f"[ANALYZE] Adding missing numeric column '{col}' with 0.")
                    df[col] = 0
                else:
                    print(f"[ANALYZE] Adding missing categorical column '{col}' with 'missing'.")
                    df[col] = "missing"

        X = df[featurecols]

        # Drop rows with NaN in features before prediction
        mask = X.notna().all(axis=1)
        Xclean = X[mask]
        print(f"[ANALYZE] Rows valid for prediction: {len(Xclean)} / {len(df)}")

        if Xclean.empty:
            print("[ANALYZE] No valid rows (after removing NaNs). Skipping predictions.")
            predictionsfull = None
        else:
            preds = pipeline.predict(Xclean)
            predictionsfull = pd.Series(np.nan, index=df.index)
            predictionsfull.loc[mask] = preds

        generatepdfreport(
            df=df,
            coltypes=coltypes,
            predictions=predictionsfull,
            csvpath=path,
            tasktype=tasktype,
            targetcolumn=targetcolumn,
        )

    print("\n[INFO] All CSVs processed. Done.")


if __name__ == "__main__":
    main()


