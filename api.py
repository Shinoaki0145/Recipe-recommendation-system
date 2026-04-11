from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recipe_ranker import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_RECIPES_PATH,
    RecipeRankerService,
)

app = FastAPI(title="Recipe Ranking API", version="1.0.0")

_service: Optional[RecipeRankerService] = None


class RankRequest(BaseModel):
    needs: list[str] = Field(default_factory=list, description="List of needs/constraints, e.g. ['ca', 'nuong', '4 nguoi']")
    query: Optional[str] = Field(default=None, description="Optional free-text query. If omitted, the API builds one from needs.")
    candidate_recipe_ids: Optional[list[int]] = Field(
        default=None,
        description="Optional list of recipe_id candidates to re-rank. If omitted, the whole dataset is ranked.",
    )
    top_k: int = Field(default=5, ge=1, le=20)


def get_service() -> RecipeRankerService:
    global _service
    if _service is None:
        _service = RecipeRankerService.load_or_train(
            artifact_path=DEFAULT_ARTIFACT_PATH,
            labels_path=DEFAULT_LABELS_PATH,
            recipes_path=DEFAULT_RECIPES_PATH,
            metrics_path=DEFAULT_METRICS_PATH,
        )
    return _service


@app.get("/health")
def health() -> dict:
    service = get_service()
    return service.health()


@app.post("/rank")
def rank_recipes(payload: RankRequest) -> dict:
    service = get_service()
    try:
        return service.rank_from_needs(
            needs=payload.needs,
            candidate_recipe_ids=payload.candidate_recipe_ids,
            top_k=payload.top_k,
            query=payload.query,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
