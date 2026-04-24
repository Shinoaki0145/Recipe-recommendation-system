from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pymongo import MongoClient
from transformers import AutoModel, AutoTokenizer

try:
    from backend.recipe_ranker import (
        build_recipe_features,
        normalize_recipe_id,
    )
except ModuleNotFoundError:
    from recipe_ranker import (
        build_recipe_features,
        normalize_recipe_id,
    )

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or os.getenv("API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
MODEL_NAME = os.getenv("MODEL_NAME")

_artifact_env = (os.getenv("RANKER_ARTIFACT_PATH") or "").strip()
RANKER_ARTIFACT_PATH = Path(_artifact_env).expanduser() if _artifact_env else Path("artifacts/recipe_ranker.joblib")

app = FastAPI(title="Recipe Ranking API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RankRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language recipe query")
    candidate_top_k: int = Field(default=30, ge=1, le=100)
    result_top_k: int = Field(default=5, ge=1, le=20)


def get_config_errors() -> list[str]:
    errors: list[str] = []
    if not PINECONE_API_KEY:
        errors.append("Missing PINECONE_API_KEY or API_KEY")
    if not PINECONE_INDEX:
        errors.append("Missing PINECONE_INDEX")
    if not MONGO_URI:
        errors.append("Missing MONGO_URI")
    if not MONGO_DB:
        errors.append("Missing MONGO_DB")
    if not MONGO_COLLECTION:
        errors.append("Missing MONGO_COLLECTION")
    if not MODEL_NAME:
        errors.append("Missing MODEL_NAME")
    return errors


def ensure_configured() -> None:
    errors = get_config_errors()
    if errors:
        raise RuntimeError("; ".join(errors))


def _serialize_for_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _serialize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _frontend_recipe_payload(row: dict[str, Any]) -> dict[str, Any]:
    preferred_fields = [
        "recipe_id",
        "dish_name",
        "url",
        "score",
        "pinecone_score",
        "difficulty",
        "views",
        "prep_time_min",
        "cook_time_min",
        "total_time_min",
        "servings_bin",
        "popularity",
        "category",
        "ingredients",
        "ingredients_names",
        "instructions",
        "description",
        "image",
        "thumbnail",
        "source",
    ]

    payload: dict[str, Any] = {}
    for field in preferred_fields:
        if field in row:
            payload[field] = _serialize_for_json(row[field])

    for fallback_field in ["_id", "pinecone_id"]:
        if fallback_field in row and fallback_field not in payload:
            payload[fallback_field] = _serialize_for_json(row[fallback_field])

    return payload


def _match_get(match: Any, key: str, default: Any = None) -> Any:
    if isinstance(match, dict):
        return match.get(key, default)
    return getattr(match, key, default)


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    return metadata if isinstance(metadata, dict) else {}


def _to_object_id(value: str) -> ObjectId | None:
    if not value:
        return None
    try:
        return ObjectId(value)
    except Exception:
        return None


def _expand_id_variants(values: list[str]) -> list[Any]:
    expanded: list[Any] = []
    seen: set[tuple[str, str]] = set()

    for raw_value in values:
        value = str(raw_value).strip()
        if not value:
            continue

        string_key = ("str", value)
        if string_key not in seen:
            expanded.append(value)
            seen.add(string_key)

        if value.isdigit():
            int_key = ("int", value)
            if int_key not in seen:
                expanded.append(int(value))
                seen.add(int_key)

    return expanded


@lru_cache(maxsize=1)
def get_embedding_components():
    ensure_configured()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(MODEL_NAME)
    embedding_model.eval()
    return tokenizer, embedding_model


@lru_cache(maxsize=1)
def get_pinecone_index():
    ensure_configured()
    client = Pinecone(api_key=PINECONE_API_KEY)
    return client.Index(PINECONE_INDEX)


@lru_cache(maxsize=1)
def get_mongo_collection():
    ensure_configured()
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_COLLECTION]


@lru_cache(maxsize=1)
def get_ranker_artifact() -> dict[str, Any]:
    artifact_path = RANKER_ARTIFACT_PATH.expanduser()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing ranker artifact: {artifact_path}")
    return joblib.load(artifact_path)


def embed_query_vector(text: str) -> list[float]:
    tokenizer, embedding_model = get_embedding_components()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = embedding_model(**inputs)

    last_hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
    embeddings = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
    return embeddings[0].detach().cpu().numpy().astype(float).tolist()


def search_candidates(query: str, top_k: int) -> list[dict[str, Any]]:
    index = get_pinecone_index()
    results = index.query(
        vector=embed_query_vector(query),
        top_k=top_k,
        include_metadata=True,
    )
    matches = _match_get(results, "matches", []) or []

    candidate_refs: list[dict[str, Any]] = []
    for rank_position, match in enumerate(matches):
        metadata = _normalize_metadata(_match_get(match, "metadata", {}))
        pinecone_id = str(_match_get(match, "id", "")).strip()
        pinecone_score = float(_match_get(match, "score", 0.0) or 0.0)

        candidate_refs.append(
            {
                "pinecone_id": pinecone_id,
                "pinecone_score": pinecone_score,
                "rank_position": rank_position,
                "recipe_id": normalize_recipe_id(
                    metadata.get("recipe_id")
                    or metadata.get("recipeId")
                    or metadata.get("id")
                    or pinecone_id
                ),
                "mongo_id": str(
                    metadata.get("mongo_id")
                    or metadata.get("mongoId")
                    or metadata.get("_id")
                    or metadata.get("document_id")
                    or metadata.get("doc_id")
                    or ""
                ).strip(),
                "url": str(metadata.get("url") or metadata.get("source_url") or "").strip(),
                "metadata": metadata,
            }
        )

    return candidate_refs


def get_records_by_index_map(index_list: list[int], sort_field: str = "_id") -> dict[int, dict[str, Any]]:
    if not index_list:
        return {}

    collection = get_mongo_collection()
    index_set = set(index_list)
    results: dict[int, dict[str, Any]] = {}
    cursor = collection.find().sort(sort_field, 1)

    for position, doc in enumerate(cursor):
        if position in index_set:
            results[position] = doc
        if len(results) == len(index_set):
            break

    return results


def _merge_record_with_ref(record: dict[str, Any] | None, ref: dict[str, Any]) -> dict[str, Any]:
    merged = dict(record) if record else {}

    if "_id" in merged:
        merged["_id"] = str(merged["_id"])

    for key, value in ref["metadata"].items():
        if key not in merged or merged[key] in ("", None, [], {}):
            merged[key] = value

    if not merged.get("recipe_id"):
        merged["recipe_id"] = ref["recipe_id"] or normalize_recipe_id(merged.get("_id"))
    else:
        merged["recipe_id"] = normalize_recipe_id(merged["recipe_id"])

    if not merged.get("dish_name") and merged.get("title"):
        merged["dish_name"] = merged["title"]

    merged["pinecone_id"] = ref["pinecone_id"]
    merged["pinecone_score"] = ref["pinecone_score"]
    return merged


def fetch_mongo_candidates(candidate_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidate_refs:
        return []

    collection = get_mongo_collection()
    recipe_id_values = _expand_id_variants(
        [ref["recipe_id"] for ref in candidate_refs if ref["recipe_id"]]
    )
    url_values = [ref["url"] for ref in candidate_refs if ref["url"]]
    mongo_id_values = [ref["mongo_id"] for ref in candidate_refs if ref["mongo_id"]]
    object_ids = [oid for oid in (_to_object_id(value) for value in mongo_id_values) if oid is not None]
    pinecone_index_values = [
        int(ref["pinecone_id"])
        for ref in candidate_refs
        if str(ref["pinecone_id"]).isdigit()
    ]

    docs_by_recipe_id: dict[str, dict[str, Any]] = {}
    if recipe_id_values:
        for doc in collection.find({"recipe_id": {"$in": recipe_id_values}}):
            docs_by_recipe_id[normalize_recipe_id(doc.get("recipe_id"))] = doc

    docs_by_url: dict[str, dict[str, Any]] = {}
    if url_values:
        for doc in collection.find({"url": {"$in": url_values}}):
            url = str(doc.get("url", "")).strip()
            if url:
                docs_by_url[url] = doc

    docs_by_mongo_id: dict[str, dict[str, Any]] = {}
    if mongo_id_values:
        for doc in collection.find({"_id": {"$in": mongo_id_values}}):
            docs_by_mongo_id[str(doc.get("_id"))] = doc

    if object_ids:
        for doc in collection.find({"_id": {"$in": object_ids}}):
            docs_by_mongo_id[str(doc.get("_id"))] = doc

    docs_by_index = get_records_by_index_map(pinecone_index_values)

    ordered_records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for ref in candidate_refs:
        record = None

        if ref["mongo_id"] and ref["mongo_id"] in docs_by_mongo_id:
            record = docs_by_mongo_id[ref["mongo_id"]]
        elif ref["recipe_id"] and ref["recipe_id"] in docs_by_recipe_id:
            record = docs_by_recipe_id[ref["recipe_id"]]
        elif ref["url"] and ref["url"] in docs_by_url:
            record = docs_by_url[ref["url"]]
        elif str(ref["pinecone_id"]).isdigit():
            record = docs_by_index.get(int(ref["pinecone_id"]))

        merged = _merge_record_with_ref(record, ref)
        record_key = (
            merged.get("recipe_id")
            or str(merged.get("_id", "")).strip()
            or str(merged.get("pinecone_id", "")).strip()
        )
        if not record_key or record_key in seen_keys:
            continue

        seen_keys.add(record_key)
        ordered_records.append(merged)

    return ordered_records


def rank_records(query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not records:
        return []

    candidates_df = pd.DataFrame(records).copy()
    if candidates_df.empty:
        return []

    if "recipe_id" not in candidates_df.columns:
        candidates_df["recipe_id"] = [
            normalize_recipe_id(row.get("_id") or row.get("pinecone_id") or idx)
            for idx, row in enumerate(records)
        ]
    else:
        candidates_df["recipe_id"] = candidates_df["recipe_id"].map(normalize_recipe_id)

    candidates_df["query"] = str(query)
    feature_df = build_recipe_features(candidates_df)
    artifact = get_ranker_artifact()
    feature_columns = artifact.get("feature_columns") or []
    if not feature_columns:
        raise RuntimeError("Ranker artifact is missing feature_columns")
    feature_df["score"] = np.asarray(
        artifact["model"].predict(feature_df[feature_columns]),
        dtype=float,
    ).reshape(-1)

    ranked_df = feature_df.sort_values(
        by=["score", "pinecone_score", "view_score", "rating_score"],
        ascending=False,
    ).reset_index(drop=True)

    top_rows = ranked_df.head(int(top_k)).to_dict("records")
    return [_frontend_recipe_payload(row) for row in top_rows]


def search_and_rank_recipes(
    query: str,
    candidate_top_k: int = 30,
    result_top_k: int = 5,
) -> dict[str, Any]:
    candidate_refs = search_candidates(query=query, top_k=candidate_top_k)
    mongo_records = fetch_mongo_candidates(candidate_refs)
    ranked_items = rank_records(query=query, records=mongo_records, top_k=result_top_k)

    return {
        "query": query,
        "candidate_count": len(candidate_refs),
        "mongo_record_count": len(mongo_records),
        "top_k": len(ranked_items),
        "items": ranked_items,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    config_errors = get_config_errors()
    artifact_path = RANKER_ARTIFACT_PATH.expanduser()
    return {
        "status": "ok" if not config_errors and artifact_path.exists() else "degraded",
        "config_errors": config_errors,
        "artifact_path": str(artifact_path),
        "artifact_exists": artifact_path.exists(),
    }


@app.post("/rank")
def rank(payload: RankRequest) -> dict[str, Any]:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        return search_and_rank_recipes(
            query=query,
            candidate_top_k=payload.candidate_top_k,
            result_top_k=payload.result_top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
