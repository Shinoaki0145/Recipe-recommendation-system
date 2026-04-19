import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

QUERY_FILE = Path("dataset/recipes_queries.txt")
JSON_PATH = Path("dataset/recipes_processed.json")
OUTPUT_CSV = Path("dataset/recipes_dataset.csv")
DEBUG_CSV = Path("dataset/recipes_dataset_debug.csv")
CHECKPOINT_JSONL = Path("dataset/recipes_dataset_checkpoint.jsonl")

DOTENV_PATH = '.env'
# PINECONE_API_KEY = "xxx"
INDEX_NAME = "recipe-recommendation"
# OPENAI_API_KEY = "xxx"
LLM_MODEL = "gpt-4.1-mini"

TOP_K = 15
RANDOM_K = 0
SEED = 42
PINECONE_OVERFETCH_FACTOR = 4
PINECONE_MAX_RETRIES = 4
LLM_MAX_RETRIES = 4
SAVE_EVERY = 10

# Reuse embedding function giống notebook embed/upload của bạn
from embed_model import embed_text

JSON_FALLBACK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

DIFFICULTY_MAP = {
    1: "Dễ",
    2: "Trung bình",
    3: "Khó",
    "1": "Dễ",
    "2": "Trung bình",
    "3": "Khó",
}

def load_dotenv(dotenv_path):
    dotenv_path = Path(dotenv_path)
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip().removeprefix('export ').strip()
        value = value.strip()
        if value and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def normalize_id(value: Any) -> str:
    text = str(value).strip()
    return text if text else ""


def read_queries(query_file: str) -> List[str]:
    with open(query_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_recipe_df(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("recipes_processed.json phải là list các recipe")

    df = pd.DataFrame(data).fillna("")

    df["recipe_id"] = [str(i) for i in range(len(df))]

    return df


def difficulty_to_text(value: Any) -> str:
    return DIFFICULTY_MAP.get(value, str(value).strip())


def normalize_ingredient_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return value
    return []


def ingredient_to_text(item: Dict[str, Any], include_quantity: bool = True) -> str:
    name = str(item.get("name", "")).strip()
    if not name:
        return ""

    quantity = item.get("quantity", "")
    unit = str(item.get("unit", "")).strip()

    if not include_quantity:
        return name

    quantity_text = ""
    if quantity not in (None, ""):
        quantity_text = str(quantity).strip()
        if quantity_text.endswith(".0"):
            quantity_text = quantity_text[:-2]

    if quantity_text and unit:
        return f"{name} ({quantity_text} {unit})"
    if quantity_text:
        return f"{name} ({quantity_text})"
    if unit:
        return f"{name} ({unit})"
    return name


def row_to_text(row: pd.Series) -> str:
    ingredients = normalize_ingredient_list(row.get("ingredients", []))
    ingredient_lines = [ingredient_to_text(x, include_quantity=True) for x in ingredients]
    ingredient_lines = [x for x in ingredient_lines if x]

    parts = [
        f"ID món: {row.get('recipe_id', '')}",
        f"Tên món: {row.get('dish_name', '')}",
        f"Danh mục: {row.get('category', '')}",
        f"Độ khó: {difficulty_to_text(row.get('difficulty', ''))}",
        f"Thời gian chuẩn bị (phút): {row.get('prep_time_min', '')}",
        f"Thời gian nấu (phút): {row.get('cook_time_min', '')}",
        f"Nhóm khẩu phần: {row.get('servings_bin', '')}",
        f"Độ phổ biến: {row.get('popularity', '')}",
    ]

    if ingredient_lines:
        parts.append("Nguyên liệu: " + "; ".join(ingredient_lines))

    return "\n".join([p for p in parts if not p.endswith(": ")])


def compact_recipe_for_llm(row: pd.Series) -> Dict[str, Any]:
    return {
        "recipe_id": str(row.get("recipe_id", "")),
        "dish_name": row.get("dish_name", ""),
        "difficulty": row.get("difficulty", ""),
        "category": row.get("category", ""),
        "ingredients": row.get("ingredients", ""),
        "prep_time_min": row.get("prep_time_min", ""),
        "cook_time_min": row.get("cook_time_min", ""),
        "servings_bin": row.get("servings_bin", ""),
        "popularity": row.get("popularity", ""),
    }


def embed_query(query: str) -> List[float]:
    emb = embed_text([query]).cpu().numpy()[0]
    return emb.tolist()


def query_pinecone_with_retry(index: Any, query_text: str, top_k: int) -> List[str]:
    vector = embed_query(query_text)
    last_error = None
    for attempt in range(1, PINECONE_MAX_RETRIES + 1):
        try:
            response = index.query(
                vector=vector,
                top_k=top_k * PINECONE_OVERFETCH_FACTOR,
                include_metadata=False,
            )
            matches = response.get("matches", []) if isinstance(response, dict) else response.matches
            ids: List[str] = []
            seen = set()
            for match in matches:
                rid = str(match["id"] if isinstance(match, dict) else match.id)
                if rid not in seen:
                    seen.add(rid)
                    ids.append(rid)
                if len(ids) >= top_k:
                    break
            return ids
        except Exception as exc:
            last_error = exc
            sleep_s = min(2 ** attempt, 10)
            print(f"[WARN] Pinecone lỗi lần {attempt}/{PINECONE_MAX_RETRIES}: {exc}. Sleep {sleep_s}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"Pinecone query thất bại cho query: {query_text}\n{last_error}")


def sample_random_recipe_ids(all_ids: List[str], excluded_ids: List[str], k: int, rng: random.Random) -> List[str]:
    excluded_set = set(excluded_ids)
    pool = [rid for rid in all_ids if rid not in excluded_set]
    if len(pool) < k:
        raise ValueError(f"Không đủ món để random {k} món không trùng. Pool còn {len(pool)} món.")
    return rng.sample(pool, k)


def build_labeling_messages(query_text: str, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_prompt = """
Bạn là chuyên gia gán nhãn relevance cho bài toán gợi ý món ăn.

Mục tiêu:
Với 1 query và danh sách món ăn, hãy gán nhãn 0-4 cho TỪNG món.

Các thuộc tính quan trọng trong dữ liệu:
- dish_name: tên món.
- difficulty: 1 = dễ, 2 = trung bình, 3 = khó
- ingredients: nguyên liệu, có thể có số lượng hoặc không
- category: loại món
- prep_time_min: thời gian chuẩn bị (phút)
- cook_time_min: thời gian nấu (phút)
- servings_bin: nhóm khẩu phần, ví dụ 1 người, 2-3 người, 4-6 người
- popularity: độ phổ biến (có 5 mức độ)

Ý nghĩa nhãn:
- 4 = Rất phù hợp: khớp gần như đầy đủ các điều kiện quan trọng trong query.
- 3 = Khá phù hợp: khớp nhiều điều kiện chính nhưng còn lệch nhẹ hoặc thiếu một vài điều kiện phụ.
- 2 = Phù hợp trung bình: có liên quan rõ ràng nhưng chỉ khớp một phần.
- 1 = Ít phù hợp: chỉ khớp yếu hoặc chỉ dính 1 chi tiết nhỏ.
- 0 = Không phù hợp: hầu như không liên quan hoặc mâu thuẫn với nhu cầu.

Ưu tiên khi chấm:
1. ingredients: nếu query yêu cầu nguyên liệu chính thì đây là ưu tiên rất cao.
2. category: nếu query nói rõ món nướng, món xào, món kho, món cháo...
3. dish_name nếu query nhắc đích danh tên món hoặc biến thể món.
4. difficulty.
5. prep_time_min và cook_time_min.
6. servings_bin.
7. popularity: nếu query nêu độ phổ biến thì ưu tiên xét thuộc tính này.
Luật chấm rất quan trọng:
- Không bịa thêm dữ kiện.
- Nếu thiếu dữ liệu cho một điều kiện thì không được tự suy diễn là có.
- Nếu query nêu nguyên liệu chính mà món không có nguyên liệu đó thì không được label cao.
- Nếu query nhắc tên món rất cụ thể mà dish_name không khớp rõ ràng thì không được label cao chỉ vì chung nguyên liệu.
- Nếu query nêu độ khó rõ ràng mà món lệch hẳn độ khó thì không được label cao.
- Nếu query nêu giới hạn thời gian, hãy so theo từng trường prep_time_min và cook_time_min. Không tự cộng tổng thời gian trừ khi query nói rõ tổng thời gian.
- Nếu query nói "nhanh", "làm gấp", "ít thời gian" thì ưu tiên các món có prep_time_min và cook_time_min thấp hơn, nhưng vẫn phải dựa trên dữ liệu thật.
- Nếu query nêu số lượng nguyên liệu, chỉ xem là khớp khi dữ liệu thật sự có quantity phù hợp.
- Nếu query nhắc khẩu phần, chỉ xét theo servings_bin; không tự nội suy chính xác ngoài dữ liệu đã cho.
- Nếu query nhắc mức độ phổ biến, chỉ ưu tiên popularity; views chỉ là tín hiệu phụ.
- Nếu query không nhắc một thuộc tính thì không phạt món vì thuộc tính đó.
- Chỉ gán 4 khi thực sự rất hợp.
- Không cố cân bằng label; chấm trung thực theo từng món.
- Trả về đúng thứ tự đầu vào.
- reason phải ngắn, rõ, bằng tiếng Việt.
- Mỗi reason nên nêu 1-2 yếu tố chính khiến món phù hợp hoặc không phù hợp.
- Chỉ trả về JSON đúng schema.
""".strip()

    user_payload = {
        "query": query_text,
        "recipes": recipes,
        "output_requirement": "Trả về recipe_id, label, reason cho từng món theo đúng thứ tự đầu vào.",
    }

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def call_llm_for_labels(client: OpenAI, model: str, query_text: str, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages = build_labeling_messages(query_text, recipes)
    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {"type": "string"},
                        "label": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
                        "reason": {"type": "string"},
                    },
                    "required": ["recipe_id", "label", "reason"],
                    "additionalProperties": False,
                },
                "minItems": len(recipes),
                "maxItems": len(recipes),
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }

    response = client.responses.create(
        model=model,
        input=messages,
        text={
            "format": {
                "type": "json_schema",
                "name": "recipe_labels",
                "schema": schema,
                "strict": True,
            }
        },
    )
    data = json.loads(response.output_text)
    labels = data["labels"]
    validate_llm_output(recipes, labels)
    return labels


def safe_call_llm_for_labels(client: OpenAI, model: str, query_text: str, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_error = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return call_llm_for_labels(client, model, query_text, recipes)
        except Exception as exc:
            last_error = exc
            print(f"[WARN] LLM structured output lỗi lần {attempt}/{LLM_MAX_RETRIES}: {exc}")
            time.sleep(min(2 ** attempt, 10))

    messages = build_labeling_messages(query_text, recipes)
    messages.append({
        "role": "user",
        "content": 'Chỉ trả về JSON thuần: {"labels":[{"recipe_id":"...","label":0,"reason":"..."}]}'
    })
    response = client.responses.create(model=model, input=messages)
    text = response.output_text.strip()
    match = JSON_FALLBACK_PATTERN.search(text)
    if not match:
        raise RuntimeError(f"Không parse được JSON từ output. Lỗi trước đó: {last_error}\nOutput: {text[:1000]}")
    data = json.loads(match.group(0))
    labels = data["labels"]
    validate_llm_output(recipes, labels)
    return labels


def validate_llm_output(recipes, labels):
    input_ids = [str(item["recipe_id"]) for item in recipes]
    output_ids = [str(item["recipe_id"]) for item in labels]

    if set(input_ids) != set(output_ids):
        raise ValueError(
            f"LLM trả về thiếu/thừa recipe_id.\nInput: {input_ids}\nOutput: {output_ids}"
        )

    if len(output_ids) != len(set(output_ids)):
        raise ValueError(f"LLM trả về recipe_id bị trùng: {output_ids}")

    for item in labels:
        if int(item["label"]) not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Label không hợp lệ: {item}")


def build_rows(query_text: str, selected_rows: List[pd.Series], labels: List[Dict[str, Any]], source_map: Dict[str, str]):
    label_lookup = {str(x["recipe_id"]): x for x in labels}
    final_rows = []
    debug_rows = []

    for row in selected_rows:
        rid = str(row["recipe_id"])
        llm_item = label_lookup[rid]

        final_rows.append({
            "query": query_text,
            "recipe": row.get("dish_name", ""),
            "label": int(llm_item["label"]),
        })

        debug_rows.append({
            "query": query_text,
            "recipe_id": rid,
            "dish_name": row.get("dish_name", ""),
            "source": source_map.get(rid, ""),
            "label": int(llm_item["label"]),
            "reason": llm_item.get("reason", ""),
            "category": row.get("category", ""),
            "difficulty": row.get("difficulty", ""),
            "prep_time_min": row.get("prep_time_min", ""),
            "cook_time_min": row.get("cook_time_min", ""),
            "servings_bin": row.get("servings_bin", ""),
        })

    return final_rows, debug_rows


def save_outputs(final_rows: List[Dict[str, Any]], debug_rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(final_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(debug_rows).to_csv(DEBUG_CSV, index=False, encoding="utf-8-sig")


def append_checkpoint(record: Dict[str, Any]) -> None:
    with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if TOP_K <= 0 or RANDOM_K < 0:
        raise ValueError("TOP_K và RANDOM_K phải > 0")

    rng = random.Random(SEED)

    print("[INFO] Đọc recipes JSON...")
    df = load_recipe_df(JSON_PATH)
    id_to_row: Dict[str, pd.Series] = {str(row["recipe_id"]): row for _, row in df.iterrows()}
    all_ids = list(id_to_row.keys())

    print("[INFO] Đọc queries...")
    queries = read_queries(QUERY_FILE)
    if not queries:
        raise ValueError("queries.txt không có query nào")
    
    load_dotenv(DOTENV_PATH)
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "").strip()

    if not openai_api_key:
        raise ValueError("Thiếu OPENAI_API_KEY trong .env hoặc biến môi trường")
    if not pinecone_api_key:
        raise ValueError("Thiếu PINECONE_API_KEY trong .env hoặc biến môi trường")

    print("[INFO] Kết nối Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)

    print("[INFO] Kết nối OpenAI...")
    client = OpenAI(api_key=openai_api_key)

    final_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    for i, query_text in enumerate(queries, start=1):
        print(f"\n[INFO] Query {i}/{len(queries)}: {query_text}")

        top_ids = query_pinecone_with_retry(index=index, query_text=query_text, top_k=TOP_K)
        top_ids = [rid for rid in top_ids if rid in id_to_row]
        if len(top_ids) < TOP_K:
            raise ValueError(f"Query '{query_text}' chỉ lấy được {len(top_ids)} món hợp lệ từ Pinecone, cần {TOP_K}.")

        random_ids = sample_random_recipe_ids(all_ids=all_ids, excluded_ids=top_ids, k=RANDOM_K, rng=rng)
        selected_ids = top_ids + random_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"Bị trùng món trong 20 món của query: {query_text}")

        selected_rows = [id_to_row[rid] for rid in selected_ids]
        source_map = {rid: "pinecone" for rid in top_ids}
        source_map.update({rid: "random" for rid in random_ids})

        recipes_for_llm = [compact_recipe_for_llm(row) for row in selected_rows]
        labels = safe_call_llm_for_labels(
            client=client,
            model=LLM_MODEL,
            query_text=query_text,
            recipes=recipes_for_llm,
        )

        batch_final_rows, batch_debug_rows = build_rows(query_text, selected_rows, labels, source_map)
        final_rows.extend(batch_final_rows)
        debug_rows.extend(batch_debug_rows)

        append_checkpoint({
            "query": query_text,
            "selected_ids": selected_ids,
            "labels": labels,
        })

        if i % SAVE_EVERY == 0 or i == len(queries):
            save_outputs(final_rows, debug_rows)
            print(f"[INFO] Đã lưu tạm sau {i} query")

    save_outputs(final_rows, debug_rows)
    print(f"\n[DONE] Saved main dataset to: {OUTPUT_CSV}")
    print(f"[DONE] Saved debug dataset to: {DEBUG_CSV}")
    print(f"[DONE] Saved checkpoint to: {CHECKPOINT_JSONL}")


if __name__ == "__main__":
    main()
