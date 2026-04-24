from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

load_dotenv()


def _env_path(name: str, default: str) -> Path:
    value = (os.getenv(name) or default).strip()
    return Path(value).expanduser()

RANDOM_STATE = 42

DEFAULT_LABELS_PATH = _env_path("RECIPE_LABELS_PATH", "dataset/recipes_dataset.csv")
DEFAULT_RECIPES_PATH = _env_path("RECIPE_RECIPES_PATH", "dataset/recipes_processed.json")
DEFAULT_ARTIFACT_PATH = _env_path("RANKER_ARTIFACT_PATH", "artifacts/recipe_ranker.joblib")
DEFAULT_METRICS_PATH = _env_path("RANKER_METRICS_PATH", "artifacts/recipe_ranker_metrics.json")

STOPWORDS = {
    "toi", "muon", "hay", "cho", "biet", "tim", "kiem", "giup", "goi", "y",
    "mot", "mon", "an", "cong", "thuc", "lam", "nau", "chi", "huong", "dan",
    "co", "nao", "khong", "thoi", "gian", "moi", "ngay", "hom", "nay",
    "nhung", "cac", "la", "voi", "cho", "nguoi", "phan",
    "quan", "huyen", "tp", "thanh", "pho",
}

FEATURE_COLUMNS = [
    "ingredient_match_ratio",
    "serving_fit",
    "difficulty_fit",
    "prep_time_fit",
    "cook_time_fit",
    "total_time_fit",
    "prep_time_score",
    "cook_time_score",
    "category_match",
    "dish_name_match",
    "rating_score",
    "view_score",
    "text_match",
]

MODEL_PIPELINE = Pipeline(
    [
        (
            "model",
            lgb.LGBMRanker(
                objective="rank_xendcg",
                learning_rate=0.05,
                min_child_samples=70,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
        ),
    ]
)
MODEL_PARAMS = {
    "model__max_bin": 31,
    "model__max_depth": 4,
    "model__n_estimators": 150,
    "model__num_leaves": 20,
}

QUERY_PHRASE_REPLACEMENTS = {
    "toida": "toi da",
    "khongqua": "khong qua",
    "khauphan": "khau phan",
    "baonhieu": "bao nhieu",
}

QUERY_TOKEN_REPLACEMENTS = {
    "ng": "nguoi",
    "nguo": "nguoi",
    "nguoii": "nguoi",
    "ph": "phut",
    "p": "phut",
    "phu": "phut",
    "phutt": "phut",
    "gioo": "gio",
    "tiengg": "tieng",
    "nhanhh": "nhanh",
    "dee": "de",
    "khoo": "kho",
}


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_query_text(query: Any) -> str:
    q = normalize_text(query)
    if not q:
        return ""

    q = re.sub(r"(\d)([a-z])", r"\1 \2", q)
    q = re.sub(r"([a-z])(\d)", r"\1 \2", q)

    for old, new in QUERY_PHRASE_REPLACEMENTS.items():
        q = re.sub(rf"\b{re.escape(old)}\b", new, q)

    normalized_tokens: list[str] = []
    for token in q.split():
        token = re.sub(r"(.)\1{2,}", r"\1\1", token)
        token = QUERY_TOKEN_REPLACEMENTS.get(token, token)
        for piece in token.split():
            if piece in {"h", "gio", "tieng"}:
                normalized_tokens.append("gio")
            elif piece in {"p", "ph", "phut"}:
                normalized_tokens.append("phut")
            elif piece in {"ng", "nguoi"}:
                normalized_tokens.append("nguoi")
            else:
                normalized_tokens.append(piece)

    deduped_tokens: list[str] = []
    for token in normalized_tokens:
        if not deduped_tokens or deduped_tokens[-1] != token:
            deduped_tokens.append(token)

    return " ".join(deduped_tokens).strip()


def normalize_recipe_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except Exception:
        return text


def tokenize(text: Any) -> list[str]:
    return [tok for tok in normalize_text(text).split() if tok and tok not in STOPWORDS]


def tokenize_query(text: Any) -> list[str]:
    return [tok for tok in normalize_query_text(text).split() if tok and tok not in STOPWORDS]


def parse_ingredients(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def extract_ingredient_names(value: Any) -> list[str]:
    names: list[str] = []
    for item in parse_ingredients(value):
        if isinstance(item, dict):
            name = normalize_text(item.get("name", ""))
            if name:
                names.append(name)
    return names


def normalize_category(value: Any) -> str:
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value is not None else ""


def normalize_instructions(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for step in value:
            if isinstance(step, dict):
                title = str(step.get("step_title", "")).strip()
                content = str(step.get("content", "")).strip()
                text = " ".join(part for part in [title, content] if part)
                if text:
                    parts.append(text)
            else:
                raw = str(step).strip()
                if raw:
                    parts.append(raw)
        return " ".join(parts)
    return str(value).strip() if value is not None else ""


def parse_servings_bin(text: Any) -> tuple[float, float]:
    s = normalize_text(text)
    if not s or s == "uncertain":
        return np.nan, np.nan

    range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*nguoi", s)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    single_match = re.search(r"(\d+)\s*nguoi", s)
    if single_match:
        value = float(single_match.group(1))
        return value, value

    return np.nan, np.nan


def parse_duration_to_minutes(text: Any) -> float:
    s = normalize_query_text(text)
    if not s:
        return np.nan

    if re.search(r"\bnua\s*(gio|tieng)\b", s):
        return 30.0

    hour_min_match = re.search(r"(\d+(?:\.\d+)?)\s*gio\s*(\d{1,2})?\s*(?:phut)?", s)
    if hour_min_match:
        hour = float(hour_min_match.group(1))
        minute = float(hour_min_match.group(2)) if hour_min_match.group(2) else 0.0
        return hour * 60.0 + minute

    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*phut", s)
    if minute_match:
        return float(minute_match.group(1))

    return np.nan


def extract_requested_people(query: Any) -> float:
    query_norm = normalize_query_text(query)

    range_patterns = [
        r"(\d{1,2})\s*-\s*(\d{1,2})\s*(?:nguoi|khau\s*phan|phan|suat)?",
        r"(\d{1,2})\s*(?:den|toi)\s*(\d{1,2})\s*(?:nguoi|khau\s*phan|phan|suat)?",
    ]
    for pattern in range_patterns:
        match = re.search(pattern, query_norm)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            return (low + high) / 2.0

    single_patterns = [
        r"(?:cho|an|khau\s*phan|phan\s*an|suat)\s*(\d{1,2})\b",
        r"(\d{1,2})\s*(?:nguoi|khau\s*phan|phan|suat)\b",
    ]
    for pattern in single_patterns:
        match = re.search(pattern, query_norm)
        if match:
            return float(match.group(1))

    return np.nan


def extract_requested_max_minutes(query: Any) -> float:
    query_norm = normalize_query_text(query)

    constrained_patterns = [
        r"(?:duoi|toi\s*da|khong\s*qua|it\s*hon|trong)\s*([a-z0-9\s]+)",
        r"(?:<=|<)\s*([a-z0-9\s]+)",
    ]
    for pattern in constrained_patterns:
        match = re.search(pattern, query_norm)
        if not match:
            continue
        tail_text = match.group(1).strip()
        minutes = parse_duration_to_minutes(tail_text)
        if not pd.isna(minutes):
            return float(minutes)
        number_only_match = re.search(r"\b(\d{1,3})\b", tail_text)
        if number_only_match:
            return float(number_only_match.group(1))

    direct_minutes = parse_duration_to_minutes(query_norm)
    if not pd.isna(direct_minutes):
        return float(direct_minutes)

    return np.nan


def popularity_to_rating(popularity: Any) -> float:
    popularity_norm = normalize_text(popularity)
    mapping = {
        "rat thap": 1.0,
        "thap": 2.0,
        "trung binh": 3.0,
        "cao": 4.0,
        "rat cao": 5.0,
        "thinh hanh": 4.5,
    }
    return mapping.get(popularity_norm, 2.5)


def build_recipe_frame(recipes_path: str | Path = DEFAULT_RECIPES_PATH) -> pd.DataFrame:
    with Path(recipes_path).open("r", encoding="utf-8") as file:
        records = json.load(file)

    recipes_df = pd.DataFrame(records)
    recipes_df["recipe_id"] = [str(i) for i in range(len(recipes_df))]

    default_columns = {
        "dish_name": "",
        "category": "",
        "instructions": "",
        "ingredients": None,
        "servings_bin": "",
        "popularity": "",
        "views": 0.0,
        "difficulty": 0.0,
        "prep_time_min": 0.0,
        "cook_time_min": 0.0,
        "url": "",
    }
    for column, default_value in default_columns.items():
        if column not in recipes_df.columns:
            if column == "ingredients":
                recipes_df[column] = [[] for _ in range(len(recipes_df))]
            else:
                recipes_df[column] = default_value

    recipes_df["dish_name"] = recipes_df["dish_name"].fillna("").astype(str)
    recipes_df["category"] = recipes_df["category"].apply(normalize_category)
    recipes_df["instructions"] = recipes_df["instructions"].apply(normalize_instructions)
    recipes_df["ingredients_names"] = recipes_df["ingredients"].apply(extract_ingredient_names)
    recipes_df["views"] = pd.to_numeric(recipes_df["views"], errors="coerce").fillna(0.0)
    recipes_df["difficulty"] = pd.to_numeric(recipes_df["difficulty"], errors="coerce").fillna(0.0)
    recipes_df["prep_time_min"] = pd.to_numeric(recipes_df["prep_time_min"], errors="coerce").fillna(0.0)
    recipes_df["cook_time_min"] = pd.to_numeric(recipes_df["cook_time_min"], errors="coerce").fillna(0.0)
    recipes_df["url"] = recipes_df["url"].fillna("").astype(str)
    recipes_df["dish_name_norm"] = recipes_df["dish_name"].map(normalize_text)
    return recipes_df


def load_merged_recipe_ranking(
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    recipes_path: str | Path = DEFAULT_RECIPES_PATH,
) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path).copy()
    if "query" not in labels_df.columns or "recipe_id" not in labels_df.columns:
        raise ValueError("labels file must contain at least 'query' and 'recipe_id' columns.")

    labels_df["query"] = labels_df["query"].fillna("").astype(str)
    if "dish_name" not in labels_df.columns:
        labels_df["dish_name"] = ""
    else:
        labels_df["dish_name"] = labels_df["dish_name"].fillna("").astype(str)
    labels_df["dish_name_norm"] = labels_df["dish_name"].map(normalize_text)
    labels_df["label"] = pd.to_numeric(labels_df.get("label", 0.0), errors="coerce").fillna(0.0)
    labels_df["recipe_id"] = labels_df["recipe_id"].map(normalize_recipe_id)

    recipes_df = build_recipe_frame(recipes_path)
    merged_df = labels_df.merge(
        recipes_df,
        on="recipe_id",
        how="left",
        suffixes=("_label", ""),
    )

    if "dish_name_label" in merged_df.columns:
        blank_mask = merged_df["dish_name"].fillna("").str.len() == 0
        merged_df.loc[blank_mask, "dish_name"] = merged_df.loc[blank_mask, "dish_name_label"]

    return merged_df


def prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = ["prep_time_min", "cook_time_min", "difficulty", "views", "label"]
    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    text_columns = ["query", "dish_name", "category", "instructions", "servings_bin", "popularity", "url"]
    for col in text_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if "ingredients_names" not in df.columns:
        df["ingredients_names"] = [[] for _ in range(len(df))]
    df["ingredients_names"] = df["ingredients_names"].apply(
        lambda value: value if isinstance(value, list) else extract_ingredient_names(value)
    )

    if "recipe_id" not in df.columns:
        df["recipe_id"] = np.arange(len(df), dtype=int).astype(str)
    else:
        df["recipe_id"] = df["recipe_id"].map(normalize_recipe_id)

    df["query_raw_norm"] = df["query"].map(normalize_text)
    df["query_norm"] = df["query"].map(normalize_query_text)
    empty_query_mask = df["query_norm"].str.len() == 0
    df.loc[empty_query_mask, "query_norm"] = df.loc[empty_query_mask, "query_raw_norm"]

    df["dish_name_norm"] = df["dish_name"].map(normalize_text)
    df["category_norm"] = df["category"].map(normalize_text)
    df["recipe_text"] = (
        df["dish_name"].fillna("")
        + " | "
        + df["category"].fillna("")
        + " | "
        + df["ingredients_names"].apply(lambda items: " ".join(items))
    )
    df["recipe_text_norm"] = df["recipe_text"].map(normalize_text)

    servings_bounds = df["servings_bin"].apply(parse_servings_bin)
    df[["serving_low", "serving_high"]] = pd.DataFrame(servings_bounds.tolist(), index=df.index)
    df["serving_mid"] = (df["serving_low"] + df["serving_high"]) / 2.0
    df["requested_people"] = df["query_norm"].map(extract_requested_people)
    df["requested_max_minutes"] = df["query_norm"].map(extract_requested_max_minutes)
    return df


def add_text_match_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["text_match"] = pd.Series(index=df.index, dtype=float)
        return df

    recipes = df[["recipe_id", "recipe_text_norm"]].drop_duplicates("recipe_id").reset_index(drop=True)
    if recipes.empty:
        df["text_match"] = 0.0
        return df

    min_df = 2 if len(recipes) >= 2 else 1
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)

    try:
        recipe_word_matrix = word_vectorizer.fit_transform(recipes["recipe_text_norm"].fillna(""))
        query_word_matrix = word_vectorizer.transform(df["query_norm"].fillna(""))
    except ValueError:
        df["text_match"] = 0.0
        return df

    recipe_index = {recipe_id: idx for idx, recipe_id in enumerate(recipes["recipe_id"])}
    df["text_match"] = [
        float(cosine_similarity(query_word_matrix[idx], recipe_word_matrix[recipe_index[recipe_id]])[0, 0])
        for idx, recipe_id in enumerate(df["recipe_id"])
    ]
    return df


def token_overlap_ratio(query_text: Any, target_text: Any) -> float:
    query_tokens = set(tokenize_query(query_text))
    target_tokens = set(tokenize(target_text))
    if not query_tokens or not target_tokens:
        return 0.0
    return float(len(query_tokens.intersection(target_tokens)) / len(query_tokens))


def compute_ingredient_match_ratio(row: pd.Series) -> float:
    query_tokens = set(tokenize_query(row.get("query_norm", row.get("query", ""))))
    if not query_tokens:
        return 0.0

    ingredient_tokens: set[str] = set()
    for name in row.get("ingredients_names", []):
        ingredient_tokens.update(tokenize(name))

    if not ingredient_tokens:
        return 0.0

    overlap = len(query_tokens.intersection(ingredient_tokens))
    return float(overlap / max(len(query_tokens), 1))


def compute_serving_fit(row: pd.Series) -> float:
    requested = row.get("requested_people", np.nan)
    low = row.get("serving_low", np.nan)
    high = row.get("serving_high", np.nan)

    if pd.isna(requested) or pd.isna(low) or pd.isna(high):
        return 0.0
    if low <= requested <= high:
        return 1.0

    gap = min(abs(requested - low), abs(requested - high))
    return float(max(0.0, 1.0 - gap / max(requested, 1.0)))


def add_ingredient_and_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    query_series = df["query_norm"] if "query_norm" in df.columns else df["query"]
    df["ingredient_match_ratio"] = df.apply(compute_ingredient_match_ratio, axis=1)
    df["serving_fit"] = df.apply(compute_serving_fit, axis=1)
    df["category_match"] = [
        token_overlap_ratio(query, category)
        for query, category in zip(query_series, df["category"])
    ]
    df["dish_name_match"] = [
        token_overlap_ratio(query, dish_name)
        for query, dish_name in zip(query_series, df["dish_name"])
    ]
    return df


def extract_requested_difficulty(query: Any) -> float:
    query_norm = normalize_query_text(query)
    easy_terms = ["de nau", "don gian", "nhanh", "it buoc", "de lam"]
    hard_terms = ["kho", "phuc tap", "cau ky"]

    if any(term in query_norm for term in easy_terms):
        return 1.0
    if any(term in query_norm for term in hard_terms):
        return 3.0
    return np.nan


def infer_effective_max_minutes(query: Any, requested_max_minutes: float) -> float:
    if not pd.isna(requested_max_minutes) and requested_max_minutes > 0:
        return float(requested_max_minutes)

    query_norm = normalize_query_text(query)
    quick_terms = ["nhanh", "de nau", "it thoi gian", "duoi 30", "30 phut"]
    if any(term in query_norm for term in quick_terms):
        return 30.0
    return np.nan


def compute_difficulty_fit(row: pd.Series) -> float:
    requested = extract_requested_difficulty(row.get("query_norm", row.get("query", "")))
    recipe_difficulty = row.get("difficulty", np.nan)
    if pd.isna(requested) or pd.isna(recipe_difficulty):
        return 0.0
    return float(max(0.0, 1.0 - abs(recipe_difficulty - requested) / 3.0))


def compute_time_fit(time_value: float, requested_max: float) -> float:
    if pd.isna(requested_max) or requested_max <= 0 or pd.isna(time_value) or time_value <= 0:
        return 0.0
    if time_value <= requested_max:
        return 1.0
    overflow = time_value - requested_max
    return float(max(0.0, 1.0 - overflow / requested_max))


def inverse_rank_score(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if len(numeric_values) <= 1:
        return pd.Series(np.ones(len(numeric_values)), index=values.index)
    ranks = numeric_values.rank(method="average", pct=True)
    return (1.0 - ranks).clip(0.0, 1.0)


def add_difficulty_and_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    query_series = df["query_norm"] if "query_norm" in df.columns else df["query"]
    df["difficulty_fit"] = df.apply(compute_difficulty_fit, axis=1)
    df["effective_max_minutes"] = [
        infer_effective_max_minutes(query, requested_max)
        for query, requested_max in zip(query_series, df["requested_max_minutes"])
    ]

    df["total_time_min"] = (
        pd.to_numeric(df["prep_time_min"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["cook_time_min"], errors="coerce").fillna(0.0)
    )
    df["prep_time_fit"] = [
        compute_time_fit(prep, max_time)
        for prep, max_time in zip(df["prep_time_min"], df["effective_max_minutes"])
    ]
    df["cook_time_fit"] = [
        compute_time_fit(cook, max_time)
        for cook, max_time in zip(df["cook_time_min"], df["effective_max_minutes"])
    ]
    df["total_time_fit"] = [
        compute_time_fit(total, max_time)
        for total, max_time in zip(df["total_time_min"], df["effective_max_minutes"])
    ]
    df["prep_time_score"] = inverse_rank_score(df["prep_time_min"])
    df["cook_time_score"] = inverse_rank_score(df["cook_time_min"])
    return df


def add_rating_score_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rating_score"] = df["popularity"].map(popularity_to_rating) / 5.0
    return df


def add_view_score_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    view_signal = np.log1p(pd.to_numeric(df["views"], errors="coerce").fillna(0.0))
    max_signal = float(view_signal.max()) if len(view_signal) else 0.0
    df["view_score"] = view_signal / max_signal if max_signal > 0 else 0.0
    return df


def build_recipe_features(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_base_frame(df)
    df = add_text_match_feature(df)
    df = add_ingredient_and_serving_features(df)
    df = add_difficulty_and_time_features(df)
    df = add_rating_score_feature(df)
    df = add_view_score_feature(df)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def build_features(df: pd.DataFrame, _art: Any | None = None) -> pd.DataFrame:
    return build_recipe_features(df)


def build_grouped_train_test_split(
    df: pd.DataFrame,
    test_percent: float = 0.3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_queries = np.array(pd.unique(df["query"]), dtype=object)
    if len(unique_queries) < 2:
        return df.sort_values(["query", "recipe_id"]).reset_index(drop=True), df.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_queries)

    n_test = max(1, int(round(len(unique_queries) * test_percent)))
    n_test = min(n_test, len(unique_queries) - 1)
    test_queries = set(unique_queries[:n_test])

    train_df = df[~df["query"].isin(test_queries)].sort_values(["query", "recipe_id"]).reset_index(drop=True)
    test_df = df[df["query"].isin(test_queries)].sort_values(["query", "recipe_id"]).reset_index(drop=True)
    return train_df, test_df


def build_group_fit_kwargs(df: pd.DataFrame) -> dict[str, list[int]]:
    sorted_df = df.sort_values(["query", "recipe_id"]).reset_index(drop=True)
    return {
        "model__group": sorted_df.groupby("query", sort=False).size().tolist(),
    }


def compute_group_metrics(
    eval_df: pd.DataFrame,
    score_col: str = "pred_score",
    top_ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    if eval_df.empty:
        metrics = {f"NDCG@{k}": 0.0 for k in top_ks}
        metrics["MRR"] = 0.0
        for k in top_ks:
            metrics[f"HIT@{k}"] = 0.0
        return metrics

    ndcg_store = {k: [] for k in top_ks}
    mrr_store: list[float] = []
    hit_store = {k: [] for k in top_ks}

    for _, group in eval_df.groupby("query"):
        ranked = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        y_true = ranked["label"].to_numpy(dtype=float)
        y_score = ranked[score_col].to_numpy(dtype=float)

        for k in top_ks:
            ndcg_store[k].append(float(ndcg_score([y_true], [y_score], k=min(k, len(ranked)))))

        relevant_positions = np.where(y_true >= 3)[0]
        mrr_store.append(float(1.0 / (relevant_positions[0] + 1)) if len(relevant_positions) else 0.0)

        for k in top_ks:
            hit_store[k].append(float((ranked["label"].head(k) >= 3).any()))

    metrics = {f"NDCG@{k}": float(np.mean(values)) for k, values in ndcg_store.items()}
    metrics["MRR"] = float(np.mean(mrr_store))
    for k in top_ks:
        metrics[f"HIT@{k}"] = float(np.mean(hit_store[k]))
    return metrics


def load_data(
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    recipes_path: str | Path = DEFAULT_RECIPES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_merged_recipe_ranking(labels_path, recipes_path), build_recipe_frame(recipes_path)


def train(
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    recipes_path: str | Path = DEFAULT_RECIPES_PATH,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_df = load_merged_recipe_ranking(labels_path, recipes_path)
    feature_df = build_recipe_features(raw_df).sort_values(["query", "recipe_id"]).reset_index(drop=True)
    train_df, test_df = build_grouped_train_test_split(feature_df)

    eval_model = clone(MODEL_PIPELINE)
    eval_model.set_params(**MODEL_PARAMS)

    eval_fit_time = 0.0
    eval_predict_time = 0.0
    test_metrics: dict[str, float] = {}

    if not train_df.empty and not test_df.empty:
        start_time = time.perf_counter()
        eval_model.fit(
            train_df[FEATURE_COLUMNS],
            train_df["label"],
            **build_group_fit_kwargs(train_df),
        )
        eval_fit_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        test_predictions = np.asarray(eval_model.predict(test_df[FEATURE_COLUMNS]), dtype=float).reshape(-1)
        eval_predict_time = time.perf_counter() - start_time

        scored_test_df = test_df.copy()
        scored_test_df["pred_score"] = test_predictions
        test_metrics = compute_group_metrics(scored_test_df)

    final_model = clone(MODEL_PIPELINE)
    final_model.set_params(**MODEL_PARAMS)

    start_time = time.perf_counter()
    final_model.fit(
        feature_df[FEATURE_COLUMNS],
        feature_df["label"],
        **build_group_fit_kwargs(feature_df),
    )
    full_fit_time = time.perf_counter() - start_time

    recipes_df = build_recipe_frame(recipes_path)

    artifact = {
        "model": final_model,
        "recipes_df": recipes_df,
        "feature_columns": FEATURE_COLUMNS,
        "model_params": MODEL_PARAMS,
        "model_name": "LightGBM Ranker",
        "created_at_unix": float(time.time()),
    }

    artifact_path = Path(artifact_path)
    metrics_path = Path(metrics_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, artifact_path)

    metrics_payload = {
        "artifact_path": str(artifact_path),
        "labels_path": str(Path(labels_path)),
        "recipes_path": str(Path(recipes_path)),
        "rows": int(len(feature_df)),
        "queries": int(feature_df["query"].nunique()) if "query" in feature_df.columns else 0,
        "catalog_recipes": int(recipes_df["recipe_id"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_queries": int(train_df["query"].nunique()) if not train_df.empty else 0,
        "test_queries": int(test_df["query"].nunique()) if not test_df.empty else 0,
        "eval_fit_time_sec": float(eval_fit_time),
        "eval_predict_time_sec": float(eval_predict_time),
        "full_fit_time_sec": float(full_fit_time),
        "test_metrics": {name: float(value) for name, value in test_metrics.items()},
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact, metrics_payload

if __name__ == "__main__":
    _, metrics = train()
    print(f"Saved artifact to: {metrics['artifact_path']}")
    print(json.dumps(metrics["test_metrics"], ensure_ascii=False, indent=2))
