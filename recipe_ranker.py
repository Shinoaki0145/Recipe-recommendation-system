from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker

RANDOM_STATE = 42

DEFAULT_LABELS_PATH = Path("dataset") / "recipes_dataset_ver1.csv"
DEFAULT_RECIPES_PATH = Path("dataset") / "recipes_processed.json"
DEFAULT_ARTIFACT_PATH = Path("artifacts") / "recipe_ranker.joblib"
DEFAULT_METRICS_PATH = Path("artifacts") / "recipe_ranker_metrics.json"

STOPWORDS = {
    "toi", "la", "cho", "va", "1", "mon", "hay", "nao", "co", "khong", "thi",
    "de", "nau", "lam", "gi", "huong", "dan", "goi", "y", "can", "mot", "tu",
    "bang", "duoi", "tren", "deu", "nhung", "chi",
}

POPULARITY_MAP = {
    "rat thap": 1,
    "thap": 2,
    "trung binh": 3,
    "cao": 4,
    "rat cao": 5,
}

CATEGORY_KEYWORDS = {
    "mon nuong": ["nuong", "quay", "bbq"],
    "mon canh": ["canh", "canh chua", "canh ngot"],
    "mon chao": ["chao"],
    "mon xao": ["xao"],
    "mon kho": ["kho"],
    "mon chien": ["chien", "ran", "gion"],
    "mon hap": ["hap"],
    "mon banh": ["banh", "panna cotta"],
    "mon che": ["che"],
    "mon nuoc": ["nuoc", "thuc uong", "do uong", "nuoc mat", "tra", "tra bi dao"],
    "mon kem": ["kem"],
    "mon goi salad": ["goi", "salad", "tron"],
    "mon bun": ["bun"],
    "mon lau": ["lau"],
    "mon chay": ["chay"],
    "mon an vat": ["an vat", "nham nhi"],
    "mon trang mieng": ["trang mieng"],
    "mon kho mam": ["kho mam"],
    "mon cuon tron": ["cuon", "banh trang", "cuon tron"],
    "sinh to": ["sinh to"],
    "tra sua": ["tra sua"],
    "nuoc ep": ["nuoc ep", "ep trai cay"],
    "nuoc cham": ["nuoc cham", "nuoc mam cham", "sot", "sauce"],
    "meo vao bep": ["meo", "bao quan", "so che", "khu mui"],
}

DIFFICULTY_KEYWORDS = {
    1: ["de", "don gian", "nhanh", "it buoc", "nguoi moi", "de lam", "de nau", "de thanh cong"],
    2: ["trung binh", "vua tay", "hon mot chut"],
    3: ["kho", "thu suc", "cau ky", "tay nghe"],
}

INTENT_KEYWORDS = {
    "loai_yeu_cau": {
        "meo": ["meo", "bi kip"],
        "cong_thuc": ["cong thuc", "cach lam", "huong dan", "goi y", "chi toi"],
    },
    "dung_cu": {
        "noi_chien_khong_dau": ["noi chien khong dau"],
        "lo_nuong": ["lo nuong"],
        "lo_vi_song": ["lo vi song"],
    },
    "doi_tuong": {
        "cho_be": ["cho be", "tre em", "tre nho"],
        "nguoi_om": ["nguoi om"],
        "nguoi_moi": ["nguoi moi", "moi bat dau"],
        "gia_dinh": ["gia dinh", "bua com gia dinh"],
        "dai_khach": ["dai khach", "tiep khach"],
    },
    "boi_canh": {
        "cuoi_tuan": ["cuoi tuan"],
        "tet": ["tet", "dau nam", "ngay le tet", "mam co tet"],
        "ngay_nong": ["ngay nong", "mua he", "giai nhiet"],
        "troi_mua": ["troi mua", "buoi toi mua"],
        "buoi_sang": ["buoi sang", "bua sang"],
        "buoi_toi": ["buoi toi", "bua toi"],
    },
    "phong_cach": {
        "lanh_manh": ["lanh manh", "it dau mo", "thanh dam", "bo duong", "nhe bung", "it calo", "du chat"],
        "ngot_nhe": ["ngot nhe", "ngot vua"],
        "cay_nhe": ["cay nhe"],
        "dep_mat": ["dep", "dep mat", "nhin dep", "bat mat", "trinh bay dep", "nhin sang"],
        "truyen_thong": ["truyen thong", "viet nam"],
        "pho_bien": ["pho bien", "thinh hanh", "noi tieng", "nhieu nguoi xem", "duoc nhieu nguoi thich", "hien nay"],
        "it_gap": ["hiem nguoi biet", "khong nhieu nguoi biet"],
    },
}

GENERIC_QUERY_TOKENS = {
    "toi", "muon", "nguoi", "phut", "mon", "nau", "lam", "an", "uong", "thoi",
    "gian", "chuan", "bi", "cong", "thuc", "goi", "y", "chi", "huong", "dan",
    "de", "kho", "nhanh", "cham", "khong", "ngoai", "hon", "qua", "duoc", "it",
    "nhieu", "nguyen", "lieu", "pho", "bien", "thinh", "hanh", "noi", "tieng",
    "ngon", "tai", "nha", "cuoi", "tuan", "gia", "dinh", "dai", "khach", "bua",
    "sang", "toi", "ngay", "nong", "tet", "giai", "nhiet", "thanh", "dam", "dep",
    "nhe", "bung", "cay", "ngot", "gion", "truyen", "thong", "viet", "nam",
    "bat", "dau", "moi", "nuoc", "dang", "kieu", "loai", "nay", "cho", "ma",
}

SHORT_INGREDIENT_TOKENS = {
    "ca", "bo", "ga", "heo", "vit", "tom", "muc", "oc", "be", "ngo", "so", "bi", "sen", "le",
}

CORR_FEATURE_COLUMNS = [
    "title_bm25_score",
    "title_overlap_ratio",
    "dish_exact_mention",
    "ingredient_match_ratio",
    "ingredient_exact_hit",
    "category_match_score",
    "vector_sim_score",
    "prep_time_min",
    "cook_time_min",
    "total_time_min",
    "prep_time_margin",
    "cook_time_margin",
    "total_time_margin",
    "serving_match",
    "serving_distance",
    "difficulty_match",
    "time_constraint_satisfied",
    "rating",
    "log_review_counts",
    "quick_match",
    "popularity_match",
]

FEATURE_COLUMNS = CORR_FEATURE_COLUMNS + [
    "serving_mid",
    "requested_people",
    "difficulty_level",
    "time_limit_min",
    "category_exact_hit",
]

RECIPE_RESPONSE_COLUMNS = [
    "recipe_id",
    "dish_name",
    "url",
    "difficulty",
    "views",
    "ingredients",
    "instructions",
    "category",
    "prep_time_min",
    "cook_time_min",
    "servings_bin",
    "cook_time_source",
    "popularity",
]


@dataclass
class TextArtifacts:
    bm25_params: dict[str, Any]
    tfidf: TfidfVectorizer
    recipe_ids: list[int]
    recipe_index_by_id: dict[int, int]
    recipe_tfidf_matrix: Any


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: Any) -> list[str]:
    return [tok for tok in normalize_text(text).split() if tok and tok not in STOPWORDS]


def contains_phrase(text: Any, phrase: Any) -> bool:
    norm_text = normalize_text(text)
    norm_phrase = normalize_text(phrase)
    if not norm_text or not norm_phrase:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(norm_phrase)}(?![a-z0-9])"
    return re.search(pattern, norm_text) is not None


def match_keyword_groups(text: Any, mapping: dict[str, list[str]]) -> list[str]:
    return [label for label, keywords in mapping.items() if any(contains_phrase(text, kw) for kw in keywords)]


def first_number(patterns: Iterable[str], text: str) -> float:
    numbers: list[int] = []
    for pattern in patterns:
        numbers.extend(int(x) for x in re.findall(pattern, text))
    return float(min(numbers)) if numbers else np.nan


def extract_people(query: str) -> float:
    q = normalize_text(query)
    for pattern in [
        r"(\d+)\s*[-–]\s*(\d+)\s*nguoi",
        r"(\d+)\s*den\s*(\d+)\s*nguoi",
        r"(\d+)\s*nguoi",
    ]:
        match = re.search(pattern, q)
        if match:
            nums = [int(x) for x in match.groups() if x is not None]
            return float(sum(nums) / len(nums))
    return np.nan


def parse_servings_bin(value: Any) -> float:
    nums = [int(x) for x in re.findall(r"\d+", normalize_text(value))]
    if not nums:
        return np.nan
    return float(nums[0]) if len(nums) == 1 else float(sum(nums[:2]) / 2)


def extract_time_info(query: str) -> dict[str, Optional[float]]:
    q = normalize_text(query)
    prep_time = first_number(
        [r"chuan\s*bi\s*duoi\s*(\d+)\s*phut", r"chuan\s*bi\s*khong\s*qua\s*(\d+)\s*phut"],
        q,
    )
    cook_time = first_number(
        [
            r"thoi\s*gian\s*nau\s*duoi\s*(\d+)\s*phut",
            r"nau\s*duoi\s*(\d+)\s*phut",
            r"nau\s*khong\s*qua\s*(\d+)\s*phut",
        ],
        q,
    )
    total_time = first_number(
        [r"duoi\s*(\d+)\s*phut", r"khong\s*qua\s*(\d+)\s*phut", r"lam\s*trong\s*(\d+)\s*phut"],
        q,
    )
    values = [value for value in [prep_time, cook_time, total_time] if not pd.isna(value)]
    min_time = float(min(values)) if values else np.nan

    def clean(value: float) -> Optional[float]:
        return None if pd.isna(value) else float(value)

    return {
        "prep_time_limit_min": clean(prep_time),
        "cook_time_limit_min": clean(cook_time),
        "total_time_limit_min": clean(total_time),
        "time_limit_min": clean(min_time),
    }


def extract_max_ingredient_count(query: str) -> float:
    q = normalize_text(query)
    value = first_number(
        [
            r"it\s*hon\s*(\d+)\s*nguyen\s*lieu",
            r"duoi\s*(\d+)\s*nguyen\s*lieu",
            r"khong\s*qua\s*(\d+)\s*nguyen\s*lieu",
        ],
        q,
    )
    if not pd.isna(value):
        return float(value)
    return 8.0 if "it nguyen lieu" in q else np.nan


def extract_negative_constraints(query: str) -> list[str]:
    q = normalize_text(query)
    negatives: list[str] = []
    for pattern in [r"ngoai\s+([^,]+)", r"khong\s*phai\s+([^,]+)", r"khong\s+([^,]+)"]:
        for match in re.findall(pattern, q):
            cleaned = match.strip()
            if cleaned and cleaned not in {"qua", "co", "can"}:
                negatives.append(cleaned)
    return negatives


def extract_explicit_ingredients(query: str) -> set[str]:
    q_tokens = set(tokenize(query))
    category_tokens: set[str] = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        category_tokens.update(tokenize(category))
        for keyword in keywords:
            category_tokens.update(tokenize(keyword))

    blocked = GENERIC_QUERY_TOKENS | category_tokens
    selected: list[str] = []
    for token in sorted(q_tokens):
        if token in blocked:
            continue
        if len(token) >= 3 or token in SHORT_INGREDIENT_TOKENS:
            selected.append(token)
    return set(selected)


def query_category_targets(query: str) -> list[str]:
    return match_keyword_groups(normalize_text(query), CATEGORY_KEYWORDS)


def query_difficulty_target(query: str) -> float:
    q = normalize_text(query)
    for level, keywords in DIFFICULTY_KEYWORDS.items():
        if any(contains_phrase(q, kw) for kw in keywords):
            return float(level)
    return np.nan


def build_recipe_text(row: pd.Series) -> str:
    ingredient_names: list[str] = []
    for item in row.get("ingredients", []) or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                ingredient_names.append(name)
    parts = [str(row.get("dish_name", "")), str(row.get("category", "")), " ".join(ingredient_names)]
    return " | ".join(part for part in parts if part)


def query_category_signal(query: str, category: Any) -> float:
    category_text = normalize_text(category if isinstance(category, str) else " ".join(category) if isinstance(category, list) else "")
    matched = query_category_targets(query)
    if not matched:
        return 0.0

    matched_norm = [normalize_text(item) for item in matched]
    if category_text in matched_norm:
        return 1.0

    for item in matched_norm:
        if item and (item in category_text or category_text in item):
            return 0.5
    return 0.0


def ingredient_features(query: str, ingredients: Any) -> tuple[float, float]:
    ingredient_names = [normalize_text(item.get("name", "")) for item in (ingredients or []) if isinstance(item, dict)]
    ingredient_tokens: set[str] = set()
    for name in ingredient_names:
        ingredient_tokens.update(tokenize(name))

    query_tokens = extract_explicit_ingredients(query)
    if not query_tokens:
        return 0.0, 0.0

    overlap = query_tokens & ingredient_tokens
    ratio = len(overlap) / max(len(query_tokens), 1)
    exact_hit = 1.0 if overlap else 0.0
    return float(ratio), exact_hit


@lru_cache(maxsize=4096)
def parse_query_constraints(query: str) -> dict[str, Any]:
    q = normalize_text(query)
    time_info = extract_time_info(query)

    request_types = match_keyword_groups(q, INTENT_KEYWORDS["loai_yeu_cau"])
    tools = match_keyword_groups(q, INTENT_KEYWORDS["dung_cu"])
    audience = match_keyword_groups(q, INTENT_KEYWORDS["doi_tuong"])
    occasion = match_keyword_groups(q, INTENT_KEYWORDS["boi_canh"])
    style = match_keyword_groups(q, INTENT_KEYWORDS["phong_cach"])
    categories = query_category_targets(query)
    ingredients = sorted(extract_explicit_ingredients(query))
    negatives = extract_negative_constraints(query)

    difficulty_target = query_difficulty_target(query)
    servings_target = extract_people(query)
    max_ingredients = extract_max_ingredient_count(query)

    return {
        "query": query,
        "query_norm": q,
        "request_type": ", ".join(request_types or ["chung"]),
        "categories": ", ".join(categories),
        "ingredients": ", ".join(ingredients),
        "excluded_terms": ", ".join(negatives),
        "tools": ", ".join(tools),
        "difficulty_target": None if pd.isna(difficulty_target) else int(difficulty_target),
        "servings_target": None if pd.isna(servings_target) else float(servings_target),
        "max_ingredients": None if pd.isna(max_ingredients) else float(max_ingredients),
        "audience_tags": ", ".join(audience),
        "occasion_tags": ", ".join(occasion),
        "style_tags": ", ".join(style),
        "wants_popular": float("pho_bien" in style),
        **time_info,
    }


def build_bm25(corpus_tokens: list[list[str]]) -> dict[str, Any]:
    n_docs = len(corpus_tokens)
    doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=float)
    avgdl = doc_lens.mean() if n_docs else 0.0

    doc_freq: dict[str, int] = {}
    for doc in corpus_tokens:
        for token in set(doc):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    return {"idf": idf, "avgdl": avgdl, "k1": 1.5, "b": 0.75}


def bm25_score(query_tokens: list[str], doc_tokens: list[str], bm25_params: dict[str, Any]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    tf: dict[str, int] = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1

    score = 0.0
    doc_len = len(doc_tokens)
    avgdl = bm25_params["avgdl"] or 1.0
    k1 = bm25_params["k1"]
    b = bm25_params["b"]

    for token in query_tokens:
        idf = bm25_params["idf"].get(token)
        if idf is None:
            continue
        freq = tf.get(token, 0)
        denom = freq + k1 * (1 - b + b * doc_len / avgdl)
        if denom > 0:
            score += idf * ((freq * (k1 + 1)) / denom)
    return float(score)


def load_labels_dataframe(labels_path: Path | str) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path, encoding="utf-8")
    required = {"query", "recipe_id", "label"}
    missing = required - set(labels_df.columns)
    if missing:
        raise KeyError(f"Missing required label columns: {sorted(missing)}")
    return labels_df


def load_recipes_dataframe(recipes_path: Path | str) -> pd.DataFrame:
    with open(recipes_path, "r", encoding="utf-8") as handle:
        recipes = json.load(handle)
    recipes_df = pd.DataFrame(recipes).copy()
    recipes_df["recipe_id"] = recipes_df.index.astype(int)
    return recipes_df


def prepare_recipe_catalog(recipes_df: pd.DataFrame) -> pd.DataFrame:
    catalog = recipes_df.copy()
    for column in ["ingredients", "instructions"]:
        catalog[column] = catalog[column].apply(lambda value: value if isinstance(value, list) else [])

    catalog["recipe_text"] = catalog.apply(build_recipe_text, axis=1)
    catalog["dish_name_norm"] = catalog["dish_name"].map(normalize_text)
    catalog["title_tokens"] = catalog["dish_name"].map(tokenize)
    catalog["serving_mid"] = catalog["servings_bin"].map(parse_servings_bin)
    catalog["total_time_min"] = catalog["prep_time_min"].fillna(0) + catalog["cook_time_min"].fillna(0)
    catalog["difficulty_level"] = pd.to_numeric(catalog["difficulty"], errors="coerce").fillna(0.0)
    catalog["views"] = pd.to_numeric(catalog["views"], errors="coerce").fillna(0.0)
    catalog["log_review_counts"] = np.log1p(catalog["views"])
    catalog["rating"] = catalog["popularity"].map(lambda value: POPULARITY_MAP.get(normalize_text(value), 0)).astype(float)
    catalog = catalog.sort_values("recipe_id").reset_index(drop=True)
    return catalog


def fit_text_artifacts(
    recipe_catalog: pd.DataFrame,
    fit_queries: Iterable[str],
    fit_recipe_ids: Optional[Iterable[int]] = None,
) -> TextArtifacts:
    fit_query_texts = list(dict.fromkeys(str(query) for query in fit_queries if str(query).strip()))
    if not fit_query_texts:
        raise ValueError("fit_queries must contain at least one query")

    if fit_recipe_ids is None:
        fit_recipe_catalog = recipe_catalog
    else:
        fit_recipe_id_set = {int(recipe_id) for recipe_id in fit_recipe_ids}
        fit_recipe_catalog = recipe_catalog[recipe_catalog["recipe_id"].isin(fit_recipe_id_set)].copy()

    if fit_recipe_catalog.empty:
        raise ValueError("No recipes available to fit text artifacts")

    bm25_params = build_bm25(fit_recipe_catalog["title_tokens"].tolist())

    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=20000)
    tfidf.fit(fit_query_texts + fit_recipe_catalog["recipe_text"].tolist())

    recipe_ids = recipe_catalog["recipe_id"].astype(int).tolist()
    recipe_tfidf_matrix = tfidf.transform(recipe_catalog["recipe_text"].tolist())
    recipe_index_by_id = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}

    return TextArtifacts(
        bm25_params=bm25_params,
        tfidf=tfidf,
        recipe_ids=recipe_ids,
        recipe_index_by_id=recipe_index_by_id,
        recipe_tfidf_matrix=recipe_tfidf_matrix,
    )


def build_query_from_needs(needs: list[str], query: Optional[str] = None) -> str:
    cleaned_needs = [str(item).strip() for item in (needs or []) if str(item).strip()]
    if query and cleaned_needs:
        return f"{query.strip()}, {', '.join(cleaned_needs)}"
    if query:
        return query.strip()
    if not cleaned_needs:
        raise ValueError("Either query or needs must be provided")
    return "Toi muon lam mon " + ", ".join(cleaned_needs)


def build_feature_frame(
    pairs_df: pd.DataFrame,
    recipe_catalog: pd.DataFrame,
    text_artifacts: TextArtifacts,
) -> pd.DataFrame:
    required = {"query", "recipe_id"}
    missing = required - set(pairs_df.columns)
    if missing:
        raise KeyError(f"Missing required pair columns: {sorted(missing)}")

    frame = pairs_df.copy()
    frame["query"] = frame["query"].astype(str)
    frame["recipe_id"] = pd.to_numeric(frame["recipe_id"], errors="raise").astype(int)
    frame = frame.sort_values(["query", "recipe_id"]).reset_index(drop=True)

    query_features = pd.DataFrame([parse_query_constraints(query) for query in frame["query"].drop_duplicates()])
    query_features = query_features.rename(columns={"ingredients": "query_ingredients"})
    frame = frame.merge(query_features, on="query", how="left", validate="many_to_one")

    merge_columns = [
        "recipe_id",
        "dish_name",
        "dish_name_norm",
        "ingredients",
        "instructions",
        "category",
        "prep_time_min",
        "cook_time_min",
        "servings_bin",
        "cook_time_source",
        "popularity",
        "views",
        "difficulty",
        "difficulty_level",
        "recipe_text",
        "title_tokens",
        "serving_mid",
        "total_time_min",
        "log_review_counts",
        "rating",
        "url",
    ]
    frame = frame.merge(recipe_catalog[merge_columns], on="recipe_id", how="left", validate="many_to_one")

    frame["query_tokens"] = frame["query"].map(tokenize)
    frame["title_bm25_score"] = [
        bm25_score(query_tokens, title_tokens, text_artifacts.bm25_params)
        for query_tokens, title_tokens in zip(frame["query_tokens"], frame["title_tokens"])
    ]

    vector_scores = np.zeros(len(frame), dtype=float)
    for query, row_index in frame.groupby("query").groups.items():
        row_positions = list(row_index)
        candidate_recipe_ids = frame.loc[row_positions, "recipe_id"].astype(int).tolist()
        candidate_matrix_positions = [text_artifacts.recipe_index_by_id[recipe_id] for recipe_id in candidate_recipe_ids]
        query_vector = text_artifacts.tfidf.transform([query])
        vector_scores[row_positions] = cosine_similarity(
            query_vector,
            text_artifacts.recipe_tfidf_matrix[candidate_matrix_positions],
        ).ravel()
    frame["vector_sim_score"] = vector_scores

    frame["category_match_score"] = [
        query_category_signal(query, category)
        for query, category in zip(frame["query"], frame["category"])
    ]
    frame["title_overlap_ratio"] = [
        len(set(query_tokens) & set(title_tokens)) / max(len(set(title_tokens)), 1)
        for query_tokens, title_tokens in zip(frame["query_tokens"], frame["title_tokens"])
    ]
    frame["dish_exact_mention"] = [
        float(len(set(query_tokens) & set(title_tokens)) >= 2)
        for query_tokens, title_tokens in zip(frame["query_tokens"], frame["title_tokens"])
    ]

    ingredient_matches = [
        ingredient_features(query, ingredients)
        for query, ingredients in zip(frame["query"], frame["ingredients"])
    ]
    frame["ingredient_match_ratio"] = [value[0] for value in ingredient_matches]
    frame["ingredient_exact_hit"] = [value[1] for value in ingredient_matches]

    frame["requested_people"] = pd.to_numeric(frame["servings_target"], errors="coerce")
    frame["prep_time_min"] = pd.to_numeric(frame["prep_time_min"], errors="coerce")
    frame["cook_time_min"] = pd.to_numeric(frame["cook_time_min"], errors="coerce")
    frame["total_time_min"] = pd.to_numeric(frame["total_time_min"], errors="coerce")
    frame["serving_mid"] = pd.to_numeric(frame["serving_mid"], errors="coerce")
    frame["difficulty_level"] = pd.to_numeric(frame["difficulty_level"], errors="coerce")
    frame["time_limit_min"] = pd.to_numeric(frame["time_limit_min"], errors="coerce")

    frame["prep_time_margin"] = np.where(
        frame["time_limit_min"].notna(),
        frame["time_limit_min"] - frame["prep_time_min"],
        0.0,
    )
    frame["cook_time_margin"] = np.where(
        frame["time_limit_min"].notna(),
        frame["time_limit_min"] - frame["cook_time_min"],
        0.0,
    )
    frame["total_time_margin"] = np.where(
        frame["time_limit_min"].notna(),
        frame["time_limit_min"] - frame["total_time_min"],
        0.0,
    )
    frame["serving_match"] = (
        frame["requested_people"].notna()
        & frame["serving_mid"].notna()
        & (np.abs(frame["requested_people"] - frame["serving_mid"]) <= 1.0)
    ).astype(float)
    frame["serving_distance"] = np.where(
        frame["requested_people"].notna() & frame["serving_mid"].notna(),
        np.abs(frame["requested_people"] - frame["serving_mid"]),
        99.0,
    )
    frame["difficulty_match"] = (
        pd.to_numeric(frame["difficulty_target"], errors="coerce").notna()
        & (pd.to_numeric(frame["difficulty_target"], errors="coerce") == frame["difficulty_level"])
    ).astype(float)
    frame["category_exact_hit"] = (frame["category_match_score"] >= 1.0).astype(float)
    frame["time_constraint_satisfied"] = np.where(
        frame["time_limit_min"].notna(),
        (frame["prep_time_min"].fillna(10**6) <= frame["time_limit_min"])
        & (frame["cook_time_min"].fillna(10**6) <= frame["time_limit_min"]),
        0,
    ).astype(float)
    frame["quick_match"] = np.where(
        frame["query_norm"].fillna("").str.contains(r"\bnhanh\b|it thoi gian|lam gap", regex=True),
        1.0 / (1.0 + frame["total_time_min"].fillna(0.0)),
        0.0,
    )
    frame["popularity_match"] = np.where(
        frame["style_tags"].fillna("").str.contains("pho_bien"),
        (frame["rating"] >= 4).astype(float),
        0.0,
    )

    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def make_group_arrays(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[int]]:
    ordered = frame.sort_values(["query", "recipe_id"]).reset_index(drop=True)
    X = ordered[FEATURE_COLUMNS]
    y = ordered["label"].astype(float)
    group = ordered.groupby("query").size().tolist()
    return ordered, X, y, group


def compute_group_metrics(eval_df: pd.DataFrame, score_col: str = "pred_score", top_ks: tuple[int, int] = (5, 10)) -> dict[str, float]:
    ndcg_store = {k: [] for k in top_ks}
    mrr_store: list[float] = []
    hit_store = {k: [] for k in top_ks}

    for _, group in eval_df.groupby("query"):
        ranked = group.sort_values(score_col, ascending=False).reset_index(drop=True)
        y_true = ranked["label"].to_numpy(dtype=float)
        y_score = ranked[score_col].to_numpy(dtype=float)

        for k in top_ks:
            ndcg_value = ndcg_score([y_true], [y_score], k=min(k, len(ranked)))
            ndcg_store[k].append(float(ndcg_value))

        relevant_positions = np.where(ranked["label"].to_numpy(dtype=float) >= 4)[0]
        mrr_store.append(1.0 / (relevant_positions[0] + 1) if len(relevant_positions) > 0 else 0.0)

        for k in top_ks:
            hit_store[k].append(float((ranked["label"].head(k) >= 4).any()))

    metrics: dict[str, float] = {}
    for k in top_ks:
        metrics[f"NDCG@{k}"] = float(np.mean(ndcg_store[k]))
    metrics["MRR"] = float(np.mean(mrr_store))
    for k in top_ks:
        metrics[f"HIT@{k}"] = float(np.mean(hit_store[k]))
    return metrics


def train_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: list[int],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    group_val: Optional[list[int]] = None,
) -> XGBRanker:
    ranker = XGBRanker(
        objective="rank:pairwise",
        eval_metric=["ndcg@5", "ndcg@10"],
        learning_rate=0.05,
        n_estimators=400,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
    )
    fit_kwargs: dict[str, Any] = {"group": group_train, "verbose": False}
    if X_val is not None and y_val is not None and group_val is not None:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
        fit_kwargs["eval_group"] = [group_train, group_val]
    ranker.fit(X_train, y_train, **fit_kwargs)
    return ranker


class RecipeRankerService:
    def __init__(
        self,
        model: XGBRanker,
        recipe_catalog: pd.DataFrame,
        text_artifacts: TextArtifacts,
        feature_columns: list[str],
        evaluation_summary: dict[str, Any],
        artifact_path: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.recipe_catalog = recipe_catalog.sort_values("recipe_id").reset_index(drop=True)
        self.text_artifacts = text_artifacts
        self.feature_columns = feature_columns
        self.evaluation_summary = evaluation_summary
        self.artifact_path = artifact_path
        self.recipe_records_by_id = {
            int(row["recipe_id"]): {column: row[column] for column in RECIPE_RESPONSE_COLUMNS}
            for _, row in self.recipe_catalog[RECIPE_RESPONSE_COLUMNS].iterrows()
        }

    @classmethod
    def train_from_paths(
        cls,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        recipes_path: Path | str = DEFAULT_RECIPES_PATH,
        artifact_path: Optional[Path | str] = DEFAULT_ARTIFACT_PATH,
        metrics_path: Optional[Path | str] = DEFAULT_METRICS_PATH,
    ) -> "RecipeRankerService":
        labels_path = Path(labels_path)
        recipes_path = Path(recipes_path)
        artifact_path = Path(artifact_path) if artifact_path is not None else None
        metrics_path = Path(metrics_path) if metrics_path is not None else None

        labels_df = load_labels_dataframe(labels_path)
        recipes_df = load_recipes_dataframe(recipes_path)
        recipe_catalog = prepare_recipe_catalog(recipes_df)

        labeled_pairs = labels_df[["query", "recipe_id", "label"]].copy()
        labeled_pairs["recipe_id"] = pd.to_numeric(labeled_pairs["recipe_id"], errors="raise").astype(int)
        labeled_pairs = labeled_pairs[labeled_pairs["recipe_id"].isin(recipe_catalog["recipe_id"])].copy()

        query_series = labeled_pairs["query"].drop_duplicates()
        train_queries, temp_queries = train_test_split(
            query_series,
            test_size=0.30,
            random_state=RANDOM_STATE,
        )
        val_queries, test_queries = train_test_split(
            temp_queries,
            test_size=0.50,
            random_state=RANDOM_STATE,
        )

        train_pairs = labeled_pairs[labeled_pairs["query"].isin(train_queries)].copy()
        val_pairs = labeled_pairs[labeled_pairs["query"].isin(val_queries)].copy()
        test_pairs = labeled_pairs[labeled_pairs["query"].isin(test_queries)].copy()

        eval_text_artifacts = fit_text_artifacts(
            recipe_catalog=recipe_catalog,
            fit_queries=train_queries.tolist(),
            fit_recipe_ids=train_pairs["recipe_id"].unique().tolist(),
        )

        train_feature_df = build_feature_frame(train_pairs, recipe_catalog, eval_text_artifacts)
        val_feature_df = build_feature_frame(val_pairs, recipe_catalog, eval_text_artifacts)
        test_feature_df = build_feature_frame(test_pairs, recipe_catalog, eval_text_artifacts)

        train_df, X_train, y_train, group_train = make_group_arrays(train_feature_df)
        val_df, X_val, y_val, group_val = make_group_arrays(val_feature_df)
        test_df, X_test, y_test, group_test = make_group_arrays(test_feature_df)

        eval_model = train_ranker(X_train, y_train, group_train, X_val, y_val, group_val)
        test_scores = eval_model.predict(X_test)
        test_df = test_df.copy()
        test_df["pred_score"] = test_scores
        test_metrics = compute_group_metrics(test_df)

        feature_importance = (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": eval_model.feature_importances_})
            .sort_values("importance", ascending=False)
            .to_dict(orient="records")
        )

        deploy_text_artifacts = fit_text_artifacts(
            recipe_catalog=recipe_catalog,
            fit_queries=labeled_pairs["query"].drop_duplicates().tolist(),
            fit_recipe_ids=recipe_catalog["recipe_id"].tolist(),
        )
        all_feature_df = build_feature_frame(labeled_pairs, recipe_catalog, deploy_text_artifacts)
        all_df, X_all, y_all, group_all = make_group_arrays(all_feature_df)
        final_model = train_ranker(X_all, y_all, group_all)

        evaluation_summary = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "labels_path": str(labels_path),
            "recipes_path": str(recipes_path),
            "n_catalog_recipes": int(len(recipe_catalog)),
            "n_labeled_pairs": int(len(labeled_pairs)),
            "n_labeled_recipes": int(labeled_pairs["recipe_id"].nunique()),
            "feature_count": len(FEATURE_COLUMNS),
            "split_summary": {
                "train_queries": int(len(train_queries)),
                "val_queries": int(len(val_queries)),
                "test_queries": int(len(test_queries)),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "test_groups": int(len(group_test)),
            },
            "clean_test_metrics": test_metrics,
            "top_feature_importance": feature_importance[:15],
            "final_training_rows": int(len(all_df)),
            "unused_eval_targets": {
                "y_test_size": int(len(y_test)),
            },
        }

        service = cls(
            model=final_model,
            recipe_catalog=recipe_catalog,
            text_artifacts=deploy_text_artifacts,
            feature_columns=FEATURE_COLUMNS,
            evaluation_summary=evaluation_summary,
            artifact_path=artifact_path,
        )

        if artifact_path is not None:
            service.save(artifact_path)
        if metrics_path is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(evaluation_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return service

    @classmethod
    def load(cls, artifact_path: Path | str) -> "RecipeRankerService":
        loaded = joblib.load(artifact_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {artifact_path} is not a {cls.__name__} instance")
        loaded.artifact_path = Path(artifact_path)
        return loaded

    @classmethod
    def load_or_train(
        cls,
        artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
        labels_path: Path | str = DEFAULT_LABELS_PATH,
        recipes_path: Path | str = DEFAULT_RECIPES_PATH,
        metrics_path: Path | str = DEFAULT_METRICS_PATH,
    ) -> "RecipeRankerService":
        artifact_path = Path(artifact_path)
        if artifact_path.exists():
            return cls.load(artifact_path)
        return cls.train_from_paths(
            labels_path=labels_path,
            recipes_path=recipes_path,
            artifact_path=artifact_path,
            metrics_path=metrics_path,
        )

    def save(self, artifact_path: Path | str) -> None:
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact_path)
        self.artifact_path = artifact_path

    def rank(
        self,
        query: str,
        candidate_recipe_ids: Optional[list[int]] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not str(query).strip():
            raise ValueError("query must not be empty")

        if candidate_recipe_ids:
            candidate_ids: list[int] = []
            seen: set[int] = set()
            for recipe_id in candidate_recipe_ids:
                value = int(recipe_id)
                if value in self.recipe_records_by_id and value not in seen:
                    candidate_ids.append(value)
                    seen.add(value)
        else:
            candidate_ids = self.recipe_catalog["recipe_id"].astype(int).tolist()

        if not candidate_ids:
            raise ValueError("No valid candidate recipes were provided")

        candidate_pairs = pd.DataFrame({"query": [query] * len(candidate_ids), "recipe_id": candidate_ids})
        feature_df = build_feature_frame(candidate_pairs, self.recipe_catalog, self.text_artifacts)
        scores = self.model.predict(feature_df[self.feature_columns])

        ranked = feature_df[["recipe_id", "dish_name", "url", "category"]].copy()
        ranked["score"] = scores
        ranked = ranked.sort_values(["score", "recipe_id"], ascending=[False, True]).head(top_k).reset_index(drop=True)

        results: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            recipe_id = int(row["recipe_id"])
            results.append(
                {
                    "recipe_id": recipe_id,
                    "score": float(row["score"]),
                    "recipe": self.recipe_records_by_id[recipe_id],
                }
            )
        return results

    def rank_from_needs(
        self,
        needs: list[str],
        candidate_recipe_ids: Optional[list[int]] = None,
        top_k: int = 5,
        query: Optional[str] = None,
    ) -> dict[str, Any]:
        resolved_query = build_query_from_needs(needs=needs, query=query)
        results = self.rank(query=resolved_query, candidate_recipe_ids=candidate_recipe_ids, top_k=top_k)
        return {
            "query": resolved_query,
            "top_k": min(int(top_k), len(results)),
            "candidate_count": len(candidate_recipe_ids) if candidate_recipe_ids else int(len(self.recipe_catalog)),
            "results": results,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "feature_count": len(self.feature_columns),
            "catalog_size": int(len(self.recipe_catalog)),
            "evaluation_summary": self.evaluation_summary,
        }
