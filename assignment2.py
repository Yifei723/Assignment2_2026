from __future__ import annotations

import itertools
import json
import math
import os
import re
import shutil
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    calinski_harabasz_score,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "assignment2_distributed"
DATA_PATH = DATA_DIR / "Indigenous_Voters_pp.csv"
DICTIONARY_PATH = DATA_DIR / "1_IndigenousVoters_2025_Data_Dictionary_100297_GENERAL.xlsx"
OUTPUT_DIR = BASE_DIR / "assignment2_outputs"
REPORT_SOURCE_PATH = Path(r"D:\Indigenous_Voters_Data_Mining_Report.docx")
REPORT_UPDATED_PATH = BASE_DIR / "Indigenous_Voters_Data_Mining_Report_updated.docx"

REPORT_LEAKAGE_COLUMNS = {
    "Serial",
    "wave",
    "W1",
    "Q15",
    "Q15_pp",
    "Q15_pp_corrected",
    "Q16",
    "Q17",
    "Q18",
    "Q19",
    "StateMap",
}
DIRECT_GREEN_SIGNAL_COLUMNS = {"Q8", "Q20", "Q25_4", "Q26_4"}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=8410)
SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_greens": "precision",
    "recall_greens": "recall",
    "f1_greens": "f1",
}
TREE_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5],
    "min_samples_leaf": [20, 35, 50],
}

THERMOMETER_COLUMNS = {
    "Q25_1",
    "Q25_2",
    "Q25_3",
    "Q25_4",
    "Q26_1",
    "Q26_2",
    "Q26_3",
    "Q26_4",
}
ASSOCIATION_COLUMNS = [
    "Gender_sdc",
    "AGE_sdc",
    "education_sdc",
    "state_sdc",
    "Q1",
    "Q7",
    "Q8",
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q20",
    "Q21",
    "Q23",
    "Q24",
    "Q25_1",
    "Q25_2",
    "Q25_3",
    "Q25_4",
    "Q26_1",
    "Q26_2",
    "Q26_3",
    "Q26_4",
    "Q27",
    "Q28",
    "Q29",
    "Q30_1",
    "Q30_2",
    "Q30_3",
    "Q30_4",
    "Q30_5",
    "Q30_6",
    "Q37_1",
    "Q37_2",
    "Q37_3",
    "Q37_4",
    "Q37_5",
    "Q37_6",
    "Q37_7",
    "Q37_8",
    "Q37_9",
    "Q37_10",
    "Q38_1",
    "Q38_2",
    "Q38_3",
    "Q38_4",
    "Q38_5",
    "Q38_6",
    "Q39",
    "Q41",
    "Q42",
    "Q43",
    "Q45",
    "Q46",
    "Q50",
]
CLUSTER_COLUMNS = [
    "Q1",
    "Q7",
    "Q10",
    "Q13",
    "Q23",
    "Q24",
    "Q27",
    "Q28",
    "Q29",
    "Q31",
    "Q32",
    "Q33",
    "Q34_2",
    "Q38_6",
    "Q39",
    "Q41",
    "Q42",
    "Q43",
    "Q45",
    "Q46",
    "Q30_3",
    "Q40_3",
]
CLUSTER_MISSING_CODES = {"", "-97", "96", "97", "98", "99", "998", "999", "9999"}


def normalise_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return str(int(float(text)))
    return text


def load_preprocessed_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    for column in df.columns:
        series = df[column].fillna("").astype(str).str.strip()
        df[column] = series.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return df


def read_data_dictionary(xlsx_path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    if not xlsx_path.exists():
        return {}, {}

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(xlsx_path) as workbook:
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in item.iter(f"{ns}t")) for item in shared_root]
        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet_root.iter(f"{ns}row"):
        values = []
        for cell in row.iter(f"{ns}c"):
            cell_type = cell.attrib.get("t")
            value_node = cell.find(f"{ns}v")
            if cell_type == "s" and value_node is not None:
                values.append(shared[int(value_node.text)])
            elif value_node is not None:
                values.append(value_node.text or "")
            else:
                values.append("")
        rows.append(values)

    header = rows[0]
    variable_labels: dict[str, str] = {}
    value_labels: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        record = dict(zip(header, row + [""] * (len(header) - len(row))))
        variable = record.get("Variable", "")
        if not variable:
            continue
        variable_labels.setdefault(variable, record.get("Label", ""))
        code = normalise_code(record.get("Value", ""))
        if code:
            value_labels.setdefault(variable, {})[code] = record.get("Value_labels", "")
    return variable_labels, value_labels


def label_for_value(variable: str, value: object, value_labels: dict[str, dict[str, str]]) -> str:
    code = normalise_code(value)
    return value_labels.get(variable, {}).get(code, code)


def prepare_supervised_data(df: pd.DataFrame, exclude_direct_signals: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    supervised_df = df.loc[df["Q15"].notna()].copy()
    supervised_df["Q15_pp_corrected"] = (supervised_df["Q15"].map(normalise_code) == "4").astype(int)

    excluded = set(REPORT_LEAKAGE_COLUMNS)
    if exclude_direct_signals:
        excluded |= DIRECT_GREEN_SIGNAL_COLUMNS

    feature_columns = [column for column in supervised_df.columns if column not in excluded]
    X = supervised_df[feature_columns].copy()
    for column in X.columns:
        X[column] = X[column].where(pd.notna(X[column]), "Missing").map(normalise_code)

    return X, supervised_df["Q15_pp_corrected"], supervised_df


def build_tree_pipeline(feature_columns: list[str], tree_params: dict[str, object]) -> Pipeline:
    model_params = dict(tree_params)
    model_params.setdefault("random_state", 42)
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_columns),
                    ]
                ),
            ),
            ("model", DecisionTreeClassifier(class_weight="balanced", **model_params)),
        ]
    )


def tune_tree(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    for params in ParameterGrid(TREE_GRID):
        pipeline = build_tree_pipeline(list(X.columns), params)
        scores = cross_validate(pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False)
        row = dict(params)
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=0)
        rows.append(row)

    tuning_df = pd.DataFrame(rows).sort_values(
        by=["roc_auc_mean", "balanced_accuracy_mean", "f1_greens_mean"],
        ascending=False,
        ignore_index=True,
    )
    selected_params = {
        "criterion": tuning_df.loc[0, "criterion"],
        "max_depth": int(tuning_df.loc[0, "max_depth"]),
        "min_samples_leaf": int(tuning_df.loc[0, "min_samples_leaf"]),
    }
    return selected_params, tuning_df


def classification_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    selected_params: dict[str, object],
    value_labels: dict[str, dict[str, str]],
) -> dict[str, object]:
    pipeline = build_tree_pipeline(list(X.columns), selected_params)
    cv_results = cross_validate(pipeline, X, y, cv=CV, scoring=SCORING, return_train_score=False)
    fold_metrics = pd.DataFrame(
        {
            "fold": np.arange(1, CV.get_n_splits() + 1),
            **{metric: cv_results[f"test_{metric}"] for metric in SCORING},
        }
    )

    metric_names = {
        "roc_auc": "ROC-AUC",
        "average_precision": "Average precision",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "precision_greens": "Precision for Greens",
        "recall_greens": "Recall for Greens",
        "f1_greens": "F1 for Greens",
    }
    table3_rows = [
        {"Measure": "Valid cases", "Five-fold cross-validated result": f"{len(y)}"},
        {"Measure": "Greens class rate", "Five-fold cross-validated result": f"{y.mean() * 100:.1f}%"},
        {
            "Measure": "Naive all-non-Greens accuracy",
            "Five-fold cross-validated result": f"{(1 - y.mean()) * 100:.1f}%",
        },
    ]
    for metric, label in metric_names.items():
        values = fold_metrics[metric]
        table3_rows.append(
            {
                "Measure": label,
                "Five-fold cross-validated result": f"{values.mean():.3f} +/- {values.std(ddof=0):.3f}",
            }
        )

    probabilities = cross_val_predict(pipeline, X, y, cv=CV, method="predict_proba")[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    confusion = pd.DataFrame(
        confusion_matrix(y, predicted, labels=[0, 1]),
        index=["actual_non_greens", "actual_greens"],
        columns=["pred_non_greens", "pred_greens"],
    )

    pipeline.fit(X, y)
    encoded_feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importances = decode_feature_importances(
        encoded_feature_names,
        importances,
        list(X.columns),
        value_labels,
    )

    return {
        "table3": pd.DataFrame(table3_rows),
        "fold_metrics": fold_metrics,
        "confusion_matrix": confusion,
        "feature_importances": feature_importances,
        "tree_text": export_text(
            pipeline.named_steps["model"],
            feature_names=list(encoded_feature_names),
            max_depth=int(selected_params["max_depth"]),
        ),
    }


def decode_feature_importances(
    encoded_feature_names: np.ndarray,
    importances: np.ndarray,
    feature_columns: list[str],
    value_labels: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for encoded_name, importance in sorted(zip(encoded_feature_names, importances), key=lambda row: row[1], reverse=True):
        if importance <= 0:
            continue
        stripped = encoded_name.replace("categorical__", "", 1)
        variable = stripped
        code = ""
        for column in sorted(feature_columns, key=len, reverse=True):
            prefix = f"{column}_"
            if stripped.startswith(prefix):
                variable = column
                code = stripped[len(prefix) :]
                break
        rows.append(
            {
                "encoded_feature": stripped,
                "variable": variable,
                "code": code,
                "value_label": label_for_value(variable, code, value_labels),
                "importance": round(float(importance), 3),
            }
        )
    return pd.DataFrame(rows)


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    supervised = df.loc[df["Q15"].notna()].copy()
    greens = int((supervised["Q15"].map(normalise_code) == "4").sum())
    q15_pp = df["Q15_pp"].fillna("").map(normalise_code).eq("1")
    q15_corrected = df["Q15"].fillna("").map(normalise_code).eq("4")
    rows = [
        ("Rows in supplied pre-processed file", len(df)),
        ("Attributes in supplied pre-processed file", df.shape[1]),
        ("Attributes after recomputing target for analysis", df.shape[1] + 1),
        ("Valid House vote records for supervised mining", len(supervised)),
        ("Missing House vote records excluded from supervised mining", int(df["Q15"].isna().sum())),
        ("Greens first preference after target correction", f"{greens} ({greens / len(supervised) * 100:.1f}% of valid Q15)"),
        ("Non-Greens first preference after target correction", f"{len(supervised) - greens} ({(1 - greens / len(supervised)) * 100:.1f}% of valid Q15)"),
        ("Original Q15_pp inconsistencies with metadata definition", int((q15_pp != q15_corrected).sum())),
    ]
    return pd.DataFrame(rows, columns=["Data characteristic", "Value"])


def thermometer_bin(value: object) -> str | None:
    number = pd.to_numeric(normalise_code(value), errors="coerce")
    if pd.isna(number) or number < 0 or number > 10:
        return None
    if number <= 2:
        return "very low 0-2"
    if number <= 4:
        return "low 3-4"
    if number == 5:
        return "neutral 5"
    if number <= 7:
        return "warm 6-7"
    return "very warm 8-10"


def left_right_bin(value: object) -> str | None:
    number = pd.to_numeric(normalise_code(value), errors="coerce")
    if pd.isna(number) or number < 0 or number > 10:
        return None
    if number <= 3:
        return "left 0-3"
    if number <= 6:
        return "centre 4-6"
    return "right 7-10"


def equality_bin(value: object) -> str | None:
    number = pd.to_numeric(normalise_code(value), errors="coerce")
    if pd.isna(number) or number < 0 or number > 10:
        return None
    if number <= 3:
        return "pro-equality 0-3"
    if number <= 6:
        return "middle 4-6"
    return "less redistribution 7-10"


def item_for_value(variable: str, value: object, value_labels: dict[str, dict[str, str]]) -> str | None:
    code = normalise_code(value)
    if not code:
        return None
    if variable in THERMOMETER_COLUMNS:
        label = thermometer_bin(code)
    elif variable == "Q13":
        label = left_right_bin(code)
    elif variable == "Q39":
        label = equality_bin(code)
    else:
        label = label_for_value(variable, code, value_labels)

    if not label:
        return None
    return f"{variable}={label}"


def build_transactions(
    supervised_df: pd.DataFrame,
    value_labels: dict[str, dict[str, str]],
) -> list[set[str]]:
    variables = [column for column in ASSOCIATION_COLUMNS if column in supervised_df.columns]
    transactions = []
    for _, row in supervised_df.iterrows():
        items = set()
        for variable in variables:
            item = item_for_value(variable, row.get(variable), value_labels)
            if item:
                items.add(item)
        transactions.append(items)
    return transactions


def mine_greens_rules(
    transactions: list[set[str]],
    y: pd.Series,
    excluded_prefixes: set[str] | None = None,
    min_antecedent_count: int = 20,
    min_greens_count: int = 8,
    max_len: int = 3,
) -> pd.DataFrame:
    excluded_prefixes = excluded_prefixes or set()
    y_array = y.to_numpy(dtype=bool)
    base_rate = float(y_array.mean())
    all_items = sorted(set().union(*transactions))
    all_items = [
        item
        for item in all_items
        if not any(item.startswith(f"{prefix}=") for prefix in excluded_prefixes)
    ]
    item_masks = {
        item: np.array([item in transaction for transaction in transactions], dtype=bool)
        for item in all_items
    }
    item_masks = {
        item: mask
        for item, mask in item_masks.items()
        if int(mask.sum()) >= min_antecedent_count
    }

    rules = []
    frequent_itemsets: dict[int, dict[tuple[str, ...], np.ndarray]] = {}

    def maybe_add_rule(itemset: tuple[str, ...], mask: np.ndarray) -> None:
        antecedent_count = int(mask.sum())
        greens_count = int((mask & y_array).sum())
        if antecedent_count < min_antecedent_count or greens_count < min_greens_count:
            return
        confidence = greens_count / antecedent_count
        rules.append(
            {
                "antecedent": " AND ".join(itemset),
                "antecedent_size": len(itemset),
                "antecedent_count": antecedent_count,
                "n_supporting_rule": greens_count,
                "confidence": confidence,
                "lift": confidence / base_rate,
                "support_pct": greens_count / len(y_array),
            }
        )

    frequent_itemsets[1] = {}
    for item, mask in item_masks.items():
        maybe_add_rule((item,), mask)
        if int(mask.sum()) >= min_antecedent_count:
            frequent_itemsets[1][(item,)] = mask

    current = frequent_itemsets[1]
    for size in range(2, max_len + 1):
        candidates: dict[tuple[str, ...], np.ndarray] = {}
        keys = sorted(current)
        seen = set()
        for left, right in itertools.combinations(keys, 2):
            union = tuple(sorted(set(left) | set(right)))
            if len(union) != size or union in seen:
                continue
            if size > 2:
                subsets_ok = all(tuple(sorted(subset)) in current for subset in itertools.combinations(union, size - 1))
                if not subsets_ok:
                    continue
            mask = np.ones(len(y_array), dtype=bool)
            for item in union:
                mask &= item_masks[item]
            if int(mask.sum()) >= min_antecedent_count and int((mask & y_array).sum()) >= min_greens_count:
                candidates[union] = mask
                maybe_add_rule(union, mask)
            seen.add(union)
        frequent_itemsets[size] = candidates
        current = candidates
        if not current:
            break

    rules_df = pd.DataFrame(rules)
    if rules_df.empty:
        return rules_df
    rules_df = rules_df.sort_values(
        by=["lift", "confidence", "n_supporting_rule", "antecedent_count"],
        ascending=[False, False, False, False],
        ignore_index=True,
    )
    rules_df["confidence_pct"] = (rules_df["confidence"] * 100).round(1)
    rules_df["lift"] = rules_df["lift"].round(2)
    rules_df["support_pct"] = (rules_df["support_pct"] * 100).round(1)
    return rules_df


def numeric_clean(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").map(normalise_code).replace(list(CLUSTER_MISSING_CODES), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def oriented_cluster_frame(df: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame({column: numeric_clean(df[column]) for column in CLUSTER_COLUMNS if column in df.columns})
    oriented = pd.DataFrame(index=df.index)
    oriented["political_interest"] = 5 - raw["Q1"]
    oriented["social_media_news_days"] = raw["Q7"] - 1
    oriented["cares_who_wins"] = 4 - raw["Q10"]
    oriented["left_position"] = 10 - raw["Q13"]
    oriented["parties_do_not_care"] = raw["Q23"]
    oriented["parties_not_needed"] = raw["Q24"]
    oriented["democracy_dissatisfaction"] = raw["Q27"]
    oriented["government_self_interest"] = 5 - raw["Q28"]
    oriented["big_interests"] = 6 - raw["Q29"]
    oriented["politicians_do_not_know"] = raw["Q31"]
    oriented["power_matters"] = 6 - raw["Q32"]
    oriented["vote_matters"] = 6 - raw["Q33"]
    oriented["voting_should_matter"] = 6 - raw["Q34_2"]
    oriented["income_difference_reduction"] = 6 - raw["Q38_6"]
    oriented["pro_equality_scale"] = 10 - raw["Q39"]
    oriented["social_services_over_tax"] = raw["Q41"]
    oriented["abortion_liberal"] = 4 - raw["Q42"]
    oriented["marriage_equality_support"] = 5 - raw["Q43"]
    oriented["pro_immigration"] = 6 - raw["Q45"]
    oriented["warming_serious"] = 5 - raw["Q46"]
    oriented["protest_participation"] = 3 - raw["Q30_3"]
    oriented["self_determination_importance"] = 5 - raw["Q40_3"]
    return oriented


def run_clustering(df: pd.DataFrame) -> dict[str, object]:
    oriented = oriented_cluster_frame(df)
    imputed = oriented.fillna(oriented.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(imputed)
    scaled_df = pd.DataFrame(scaled, columns=imputed.columns, index=imputed.index)

    k_rows = []
    labels_by_k = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=8410, n_init=50)
        labels = model.fit_predict(scaled)
        labels_by_k[k] = labels
        k_rows.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(scaled, labels)), 3),
                "calinski_harabasz": round(float(calinski_harabasz_score(scaled, labels)), 1),
            }
        )
    k_search = pd.DataFrame(k_rows)
    best_k = int(k_search.sort_values("silhouette", ascending=False).iloc[0]["k"])

    profiles = {}
    for k in [best_k, 3]:
        labels = labels_by_k[k]
        profile_df = cluster_profile(df, oriented, labels)
        profiles[k] = profile_df

    standardized_profile_k3 = scaled_df.groupby(labels_by_k[3]).mean().round(2)
    standardized_profile_k3.index.name = "cluster"

    return {
        "k_search": k_search,
        "best_k": best_k,
        "profile_best": profiles[best_k],
        "profile_k3": profiles[3],
        "standardized_profile_k3": standardized_profile_k3.reset_index(),
    }


def percentage_true(series: pd.Series, true_code: str, valid_codes: set[str] | None = None) -> float:
    codes = series.map(normalise_code)
    if valid_codes:
        codes = codes[codes.isin(valid_codes)]
    else:
        codes = codes[codes != ""]
    if len(codes) == 0:
        return math.nan
    return float((codes == true_code).mean() * 100)


def cluster_profile(df: pd.DataFrame, oriented: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    profile_rows = []
    cluster_series = pd.Series(labels, index=df.index, name="cluster")
    for cluster_id in sorted(cluster_series.unique()):
        mask = cluster_series == cluster_id
        subset = df.loc[mask]
        means = oriented.loc[mask].mean(numeric_only=True)
        profile_rows.append(
            {
                "cluster": int(cluster_id),
                "n": int(mask.sum()),
                "sample_pct": round(float(mask.mean() * 100), 1),
                "greens_pct": round(percentage_true(subset["Q15"], "4"), 1),
                "labor_pct": round(percentage_true(subset["Q15"], "2"), 1),
                "voice_yes_pct": round(percentage_true(subset["Q21"], "1", {"1", "2", "3", "4", "97"}), 1),
                "interest": round(float(means["political_interest"]), 2),
                "social_media_days": round(float(means["social_media_news_days"]), 2),
                "pro_equality": round(float(means["pro_equality_scale"]), 2),
                "warming_serious": round(float(means["warming_serious"]), 2),
                "democracy_dissatisfaction": round(float(means["democracy_dissatisfaction"]), 2),
            }
        )
    return pd.DataFrame(profile_rows)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_height: int,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def new_canvas(width: int = 1600, height: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "#fbfaf7")
    draw = ImageDraw.Draw(image)
    return image, draw


def save_horizontal_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    x_label: str,
    color: str,
) -> None:
    image, draw = new_canvas()
    title_font = load_font(38, bold=True)
    label_font = load_font(20)
    small_font = load_font(18)
    value_font = load_font(20, bold=True)

    draw.text((70, 42), title, font=title_font, fill="#1d2b34")
    left = 660
    top = 145
    right = 1470
    bar_h = 48
    gap = 28
    max_value = max(values) if values else 1.0

    axis_y = top + len(values) * (bar_h + gap) + 10
    draw.line((left, top - 18, left, axis_y), fill="#2d3a3f", width=2)
    draw.line((left, axis_y, right, axis_y), fill="#2d3a3f", width=2)
    draw.text((left, axis_y + 24), x_label, font=small_font, fill="#4d5b60")

    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * (bar_h + gap)
        draw_wrapped_text(draw, (70, y + 2), label, label_font, "#1f2a2e", left - 100, 25)
        bar_w = int((right - left) * value / max_value)
        draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=6, fill=color)
        draw.text((left + bar_w + 14, y + 12), f"{value:.2f}", font=value_font, fill="#1d2b34")

    image.save(path)


def save_vertical_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    color: str,
) -> None:
    image, draw = new_canvas()
    title_font = load_font(38, bold=True)
    label_font = load_font(22)
    small_font = load_font(18)
    value_font = load_font(22, bold=True)

    draw.text((70, 42), title, font=title_font, fill="#1d2b34")
    left, top, right, bottom = 170, 145, 1480, 760
    max_value = max(values) * 1.15 if values else 1.0
    draw.line((left, top, left, bottom), fill="#2d3a3f", width=2)
    draw.line((left, bottom, right, bottom), fill="#2d3a3f", width=2)
    draw.text((70, top - 42), y_label, font=small_font, fill="#4d5b60")

    bar_width = min(190, int((right - left) / max(len(values), 1) * 0.55))
    step = (right - left) / max(len(values), 1)
    for i, (label, value) in enumerate(zip(labels, values)):
        center = int(left + step * (i + 0.5))
        bar_left = center - bar_width // 2
        bar_right = center + bar_width // 2
        bar_top = int(bottom - (bottom - top) * value / max_value)
        draw.rounded_rectangle((bar_left, bar_top, bar_right, bottom), radius=8, fill=color)
        draw.text((center - text_width(draw, f"{value:.1f}%", value_font) // 2, bar_top - 34), f"{value:.1f}%", font=value_font, fill="#1d2b34")
        draw.text((center - text_width(draw, label, label_font) // 2, bottom + 22), label, font=label_font, fill="#1f2a2e")

    image.save(path)


def save_profile_line_chart(path: Path, title: str, profile: pd.DataFrame) -> None:
    image, draw = new_canvas()
    title_font = load_font(38, bold=True)
    label_font = load_font(17)
    small_font = load_font(16)
    legend_font = load_font(20, bold=True)

    draw.text((70, 42), title, font=title_font, fill="#1d2b34")
    columns = [
        ("political_interest", "Interest"),
        ("social_media_news_days", "Social media"),
        ("pro_equality_scale", "Equality"),
        ("warming_serious", "Climate"),
        ("democracy_dissatisfaction", "Democracy dis."),
        ("government_self_interest", "Gov self-int."),
    ]
    plot = profile[["cluster"] + [col for col, _ in columns]].copy()
    left, top, right, bottom = 130, 150, 1460, 720
    values = plot[[col for col, _ in columns]].to_numpy(dtype=float)
    y_min = min(-1.5, float(np.nanmin(values)) - 0.2)
    y_max = max(1.5, float(np.nanmax(values)) + 0.2)

    for tick in np.linspace(y_min, y_max, 7):
        y = int(bottom - (tick - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill="#e3ded5", width=1)
        draw.text((58, y - 10), f"{tick:.1f}", font=small_font, fill="#667176")
    draw.line((left, top, left, bottom), fill="#2d3a3f", width=2)
    draw.line((left, bottom, right, bottom), fill="#2d3a3f", width=2)

    x_positions = np.linspace(left, right, len(columns))
    for x, (_, label) in zip(x_positions, columns):
        draw.line((int(x), bottom, int(x), bottom + 8), fill="#2d3a3f", width=2)
        draw_wrapped_text(draw, (int(x) - 65, bottom + 22), label, label_font, "#1f2a2e", 130, 21)

    colors = ["#126d7a", "#cc6b32", "#5f7f36"]
    for row_idx, (_, row) in enumerate(plot.iterrows()):
        points = []
        for x, (column, _) in zip(x_positions, columns):
            value = float(row[column])
            y = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
            points.append((int(x), y))
        color = colors[row_idx % len(colors)]
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        legend_x = 970
        legend_y = 90 + row_idx * 32
        draw.rectangle((legend_x, legend_y + 5, legend_x + 24, legend_y + 19), fill=color)
        draw.text((legend_x + 34, legend_y), f"Cluster {int(row['cluster'])}", font=legend_font, fill="#1d2b34")

    draw.text((70, 792), "Values are standardised means within the k=3 descriptive clustering solution.", font=small_font, fill="#4d5b60")
    image.save(path)


def generate_figures(
    full_classification: dict[str, object],
    association_restricted: pd.DataFrame,
    clustering: dict[str, object],
) -> dict[str, Path]:
    figure_dir = OUTPUT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    importance = full_classification["feature_importances"].head(6)
    figure1 = figure_dir / "figure1_feature_importance.png"
    save_horizontal_bar_chart(
        figure1,
        "Figure 1. Most influential one-hot conditions in the selected decision tree",
        [f"{row.variable} = {row.value_label}" for row in importance.itertuples()],
        [float(value) for value in importance["importance"]],
        "Impurity importance",
        "#126d7a",
    )

    rules = association_restricted.head(8)
    figure2 = figure_dir / "figure2_restricted_rule_lift.png"
    save_horizontal_bar_chart(
        figure2,
        "Figure 2. Lift of restricted association rules for Greens first preference",
        list(rules["antecedent"]),
        [float(value) for value in rules["lift"]],
        "Lift over the base Greens rate",
        "#cc6b32",
    )

    figure3 = figure_dir / "figure3_cluster_profiles.png"
    save_profile_line_chart(
        figure3,
        "Figure 3. Standardised mean profiles for the k=3 clusters",
        clustering["standardized_profile_k3"],
    )

    profile = clustering["profile_k3"]
    figure4 = figure_dir / "figure4_greens_by_cluster.png"
    save_vertical_bar_chart(
        figure4,
        "Figure 4. Greens first-preference rate by cluster",
        [f"Cluster {int(value)}" for value in profile["cluster"]],
        [float(value) for value in profile["greens_pct"]],
        "Greens first-preference voters (%)",
        "#5f7f36",
    )

    return {
        "figure1": figure1,
        "figure2": figure2,
        "figure3": figure3,
        "figure4": figure4,
    }


def docx_paragraph_text(paragraph: ET.Element) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", ns)).strip()


def simple_docx_paragraph(text: str) -> ET.Element:
    return ET.fromstring(
        f'''<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:r><w:t>{escape_xml(text)}</w:t></w:r>
            </w:p>'''
    )


def escape_xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def image_paragraph(relationship_id: str, name: str, image_path: Path, doc_pr_id: int) -> ET.Element:
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    width_emu = int(6.2 * 914400)
    height_emu = int(width_emu * height_px / width_px)
    return ET.fromstring(
        f'''<w:p
              xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
              xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
              xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
              xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
              xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <w:r>
                <w:drawing>
                  <wp:inline distT="0" distB="0" distL="0" distR="0">
                    <wp:extent cx="{width_emu}" cy="{height_emu}"/>
                    <wp:docPr id="{doc_pr_id}" name="{escape_xml(name)}"/>
                    <wp:cNvGraphicFramePr>
                      <a:graphicFrameLocks noChangeAspect="1"/>
                    </wp:cNvGraphicFramePr>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                        <pic:pic>
                          <pic:nvPicPr>
                            <pic:cNvPr id="0" name="{escape_xml(image_path.name)}"/>
                            <pic:cNvPicPr/>
                          </pic:nvPicPr>
                          <pic:blipFill>
                            <a:blip r:embed="{relationship_id}"/>
                            <a:stretch><a:fillRect/></a:stretch>
                          </pic:blipFill>
                          <pic:spPr>
                            <a:xfrm>
                              <a:off x="0" y="0"/>
                              <a:ext cx="{width_emu}" cy="{height_emu}"/>
                            </a:xfrm>
                            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                          </pic:spPr>
                        </pic:pic>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing>
              </w:r>
            </w:p>'''
    )


def add_png_content_type(content_types_root: ET.Element) -> None:
    ns = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    has_png = any(
        child.tag == f"{ns}Default" and child.attrib.get("Extension") == "png"
        for child in content_types_root
    )
    if not has_png:
        ET.SubElement(
            content_types_root,
            f"{ns}Default",
            {"Extension": "png", "ContentType": "image/png"},
        )


def add_image_relationships(rels_root: ET.Element, figures: dict[str, Path]) -> dict[str, str]:
    rel_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    existing_ids = [child.attrib.get("Id", "") for child in rels_root]
    numeric_ids = [
        int(match.group(1))
        for rel_id in existing_ids
        if (match := re.fullmatch(r"rId(\d+)", rel_id))
    ]
    next_id = max(numeric_ids or [0]) + 1
    relationship_ids = {}
    for figure_name in figures:
        rel_id = f"rId{next_id}"
        next_id += 1
        relationship_ids[figure_name] = rel_id
        ET.SubElement(
            rels_root,
            f"{rel_ns}Relationship",
            {
                "Id": rel_id,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"media/generated_{figure_name}.png",
            },
        )
    return relationship_ids


def insert_after_anchor(body: ET.Element, anchor_prefix: str, elements: list[ET.Element]) -> None:
    children = list(body)
    for index, child in enumerate(children):
        if child.tag.endswith("}p") and docx_paragraph_text(child).startswith(anchor_prefix):
            for offset, element in enumerate(elements, start=1):
                body.insert(index + offset, element)
            return


def refresh_report_figures(figures: dict[str, Path]) -> Path | None:
    if not REPORT_UPDATED_PATH.exists():
        return None

    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }
    with ZipFile(REPORT_UPDATED_PATH, "r") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}

    document_root = ET.fromstring(files["word/document.xml"])
    rels_root = ET.fromstring(files["word/_rels/document.xml.rels"])
    content_types_root = ET.fromstring(files["[Content_Types].xml"])
    body = document_root.find("w:body", ns)
    if body is None:
        return None

    for paragraph in list(body.findall("./w:p", ns)):
        text = docx_paragraph_text(paragraph)
        if paragraph.find(".//w:drawing", ns) is not None or text.startswith("Figure "):
            body.remove(paragraph)

    relationship_ids = add_image_relationships(rels_root, figures)
    figure_blocks = {
        "The cross-validated confusion matrix": [
            image_paragraph(relationship_ids["figure1"], "Figure 1", figures["figure1"], 1001),
            simple_docx_paragraph("Figure 1. Most influential one-hot conditions in the selected decision tree."),
        ],
        "Table 6.": [
            image_paragraph(relationship_ids["figure2"], "Figure 2", figures["figure2"], 1002),
            simple_docx_paragraph("Figure 2. Lift of the top restricted association rules with Greens first preference as consequent."),
        ],
        "Table 8.": [
            image_paragraph(relationship_ids["figure3"], "Figure 3", figures["figure3"], 1003),
            simple_docx_paragraph("Figure 3. Standardised mean profiles for the three k-means clusters."),
            image_paragraph(relationship_ids["figure4"], "Figure 4", figures["figure4"], 1004),
            simple_docx_paragraph("Figure 4. Greens first-preference rate by cluster."),
        ],
    }
    for anchor, elements in figure_blocks.items():
        insert_after_anchor(body, anchor, elements)

    add_png_content_type(content_types_root)
    files["word/document.xml"] = ET.tostring(document_root, encoding="utf-8", xml_declaration=True)
    files["word/_rels/document.xml.rels"] = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)
    files["[Content_Types].xml"] = ET.tostring(content_types_root, encoding="utf-8", xml_declaration=True)
    for figure_name, path in figures.items():
        files[f"word/media/generated_{figure_name}.png"] = path.read_bytes()

    temp_path = REPORT_UPDATED_PATH.with_suffix(".tmp.docx")
    if temp_path.exists():
        temp_path.unlink()
    with ZipFile(temp_path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    try:
        os.replace(temp_path, REPORT_UPDATED_PATH)
        return REPORT_UPDATED_PATH
    except PermissionError:
        fallback_path = REPORT_UPDATED_PATH.with_name(f"{REPORT_UPDATED_PATH.stem}_with_figures.docx")
        if fallback_path.exists():
            try:
                fallback_path.unlink()
            except PermissionError:
                fallback_path = REPORT_UPDATED_PATH.with_name(f"{REPORT_UPDATED_PATH.stem}_with_figures_new.docx")
        os.replace(temp_path, fallback_path)
        return fallback_path


def sensitivity_note(
    full_table: pd.DataFrame,
    restricted_table: pd.DataFrame,
    best_k: int,
    k_search: pd.DataFrame,
) -> str:
    def lookup(table: pd.DataFrame, measure: str) -> str:
        return str(table.loc[table["Measure"] == measure, "Five-fold cross-validated result"].iloc[0])

    k2_sil = float(k_search.loc[k_search["k"] == 2, "silhouette"].iloc[0])
    k3_sil = float(k_search.loc[k_search["k"] == 3, "silhouette"].iloc[0])

    lines = [
        "# Corrected Report Notes",
        "",
        "These outputs follow the report's main idea, but recalculate the values from the current CSV rather than trusting the numbers already written in the Word document.",
        "",
        "## Classification",
        f"- Full Greens model ROC-AUC: {lookup(full_table, 'ROC-AUC')}.",
        f"- Full Greens model balanced accuracy: {lookup(full_table, 'Balanced accuracy')}.",
        f"- Sensitivity model without Q8, Q20, Q25_4 and Q26_4 ROC-AUC: {lookup(restricted_table, 'ROC-AUC')}.",
        f"- Sensitivity model without those direct party/warmth signals balanced accuracy: {lookup(restricted_table, 'Balanced accuracy')}.",
        "- Improvement to the report: keep the full tree as a pattern-discovery model, but explicitly say that much of its strength comes from party identity, previous vote and direct Greens/Adam Bandt warmth. The restricted model is the better evidence for non-obvious attitudinal signal.",
        "",
        "## Association Rules",
        "- Rules are recalculated from transactions built from the survey columns. Thermometers are binned into 0-2, 3-4, 5, 6-7 and 8-10; Q13 and Q39 are binned into three ordered groups.",
        "- Improvement to the report: report both direct-signal rules and restricted rules. The direct-signal rules are interpretable but close to tautological.",
        "",
        "## Clustering",
        f"- Best k by silhouette in the recalculated run: k={best_k}.",
        f"- Silhouette for k=2: {k2_sil:.3f}; silhouette for k=3: {k3_sil:.3f}.",
        "- Improvement to the report: if k=3 is used for interpretability, describe it as a descriptive segmentation rather than as evidence of sharply separated natural groups. If the marking focus is formal quality, prefer the best-silhouette solution.",
        "",
        "## General Caveats",
        "- The analysis is unweighted; W1 is not used in model fitting.",
        "- Accuracy is not the best headline metric because Greens voters are a minority class.",
        "- Do not present the model as causal or suitable for individual-level political targeting.",
    ]
    return "\n".join(lines) + "\n"


def save_outputs(
    target_df: pd.DataFrame,
    full_classification: dict[str, object],
    restricted_classification: dict[str, object],
    full_tuning: pd.DataFrame,
    restricted_tuning: pd.DataFrame,
    association_main: pd.DataFrame,
    association_restricted: pd.DataFrame,
    clustering: dict[str, object],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(OUTPUT_DIR / "corrected_table1_data_summary.csv", index=False)

    full_classification["table3"].to_csv(OUTPUT_DIR / "report_reference_table3.csv", index=False)
    full_classification["fold_metrics"].to_csv(OUTPUT_DIR / "report_reference_table3_fold_metrics.csv", index=False)
    full_tuning.to_csv(OUTPUT_DIR / "report_reference_table3_tuning.csv", index=False)
    full_classification["confusion_matrix"].to_csv(OUTPUT_DIR / "report_reference_table3_confusion_matrix.csv")
    full_classification["feature_importances"].to_csv(OUTPUT_DIR / "report_reference_table4_feature_importances.csv", index=False)
    (OUTPUT_DIR / "report_reference_tree.txt").write_text(full_classification["tree_text"], encoding="utf-8")

    restricted_classification["table3"].to_csv(
        OUTPUT_DIR / "improved_greens_without_direct_signals_metrics.csv",
        index=False,
    )
    restricted_classification["fold_metrics"].to_csv(
        OUTPUT_DIR / "improved_greens_without_direct_signals_fold_metrics.csv",
        index=False,
    )
    restricted_tuning.to_csv(OUTPUT_DIR / "improved_greens_without_direct_signals_tuning.csv", index=False)
    restricted_classification["confusion_matrix"].to_csv(
        OUTPUT_DIR / "improved_greens_without_direct_signals_confusion_matrix.csv"
    )
    restricted_classification["feature_importances"].to_csv(
        OUTPUT_DIR / "improved_greens_without_direct_signals_feature_importances.csv",
        index=False,
    )

    association_main.head(20).to_csv(OUTPUT_DIR / "corrected_table5_association_rules_direct_allowed.csv", index=False)
    association_restricted.head(20).to_csv(OUTPUT_DIR / "corrected_table6_association_rules_restricted.csv", index=False)

    clustering["k_search"].to_csv(OUTPUT_DIR / "corrected_table7_cluster_validity.csv", index=False)
    clustering["profile_best"].to_csv(OUTPUT_DIR / "corrected_cluster_profile_best_k.csv", index=False)
    clustering["profile_k3"].to_csv(OUTPUT_DIR / "corrected_table8_cluster_profile_k3.csv", index=False)
    clustering["standardized_profile_k3"].to_csv(
        OUTPUT_DIR / "corrected_cluster_standardized_profile_k3.csv",
        index=False,
    )
    figures = generate_figures(full_classification, association_restricted, clustering)

    notes = sensitivity_note(
        full_classification["table3"],
        restricted_classification["table3"],
        clustering["best_k"],
        clustering["k_search"],
    )
    (OUTPUT_DIR / "corrected_report_notes.md").write_text(notes, encoding="utf-8")
    refreshed_report = refresh_report_figures(figures)

    summary = {
        "classification_selected_params": full_classification["selected_params"],
        "restricted_selected_params": restricted_classification["selected_params"],
        "best_cluster_k_by_silhouette": clustering["best_k"],
        "association_rules_direct_allowed": int(len(association_main)),
        "association_rules_restricted": int(len(association_restricted)),
        "figures": {name: str(path) for name, path in figures.items()},
        "updated_report_with_figures": str(refreshed_report) if refreshed_report else None,
    }
    (OUTPUT_DIR / "corrected_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = load_preprocessed_data(DATA_PATH)
    _, value_labels = read_data_dictionary(DICTIONARY_PATH)
    target_df = target_summary(df)

    X_full, y, supervised_df = prepare_supervised_data(df, exclude_direct_signals=False)
    full_params, full_tuning = tune_tree(X_full, y)
    full_classification = classification_metrics(X_full, y, full_params, value_labels)
    full_classification["selected_params"] = full_params

    X_restricted, y_restricted, _ = prepare_supervised_data(df, exclude_direct_signals=True)
    restricted_params, restricted_tuning = tune_tree(X_restricted, y_restricted)
    restricted_classification = classification_metrics(X_restricted, y_restricted, restricted_params, value_labels)
    restricted_classification["selected_params"] = restricted_params

    transactions = build_transactions(supervised_df, value_labels)
    association_main = mine_greens_rules(transactions, y)
    association_restricted = mine_greens_rules(transactions, y, excluded_prefixes=DIRECT_GREEN_SIGNAL_COLUMNS)
    clustering = run_clustering(df)

    save_outputs(
        target_df,
        full_classification,
        restricted_classification,
        full_tuning,
        restricted_tuning,
        association_main,
        association_restricted,
        clustering,
    )

    print(f"Valid supervised records: {len(supervised_df)}")
    print(f"Greens first-preference voters: {int(y.sum())}")
    print(f"Full model selected params: {full_params}")
    print(f"Restricted model selected params: {restricted_params}")
    print(f"Best cluster k by silhouette: {clustering['best_k']}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
