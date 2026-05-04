import os
import requests
from langchain.tools import tool
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
USDA_FOOD_URL   = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

def get_food_retriever():
    from build_faiss_index import get_retriever
    return get_retriever()

# ── Nutrient IDs we care about (USDA standard IDs) ──────────────────────────
# These are stable across all USDA food entries.
NUTRIENT_IDS = {
    "calories": 1008,   # Energy (kcal)
    "protein":  1003,   # Protein (g)
    "carbs":    1005,   # Carbohydrate, by difference (g)
    "fat":      1004,   # Total lipid / fat (g)
}

# ── Unit → grams conversion ──────────────────────────────────────────────────
UNIT_TO_GRAMS = {
    "g":      1,
    "gram":   1,
    "grams":  1,
    "oz":     28.3495,
    "lb":     453.592,
    "kg":     1000,
    "cup":    240,
    "tbsp":   14.787,
    "tsp":    4.929,
    "ml":     1,       # ml ≈ g for water-based ingredients
    "l":      1000,
    "fl oz":  29.574,
    "fl_oz":  29.574,
}

# ── Noise words to strip before searching ───────────────────────────────────
REMOVE_WORDS = [
    "fresh", "chopped", "diced", "sliced", "minced",
    "cooked", "grilled", "roasted", "baked", "steamed",
    "seasoned", "mixture", "sauce", "filling", "topping",
    "raw", "dried", "frozen", "canned", "organic",
]

# ── Manual name overrides for common ambiguous ingredients ───────────────────
NAME_OVERRIDES = {
    "cashew cream":      "cashews",
    "vegan cheese":      "cheddar cheese",
    "plant-based milk":  "oat milk",
    "black bean mixture":"black beans",
    "bean filling":      "black beans",
    "taco filling":      "black beans",
    "pasta":             "spaghetti",
    "noodles":           "egg noodles",
    "stock":             "chicken broth",
}

# ── Search result noise to filter out ───────────────────────────────────────
FILTER_WORDS = [
    "supplement", "protein powder", "shake", "keto",
    "diet bar", "meal replacement", "infant", "formula",
]

def normalize_ingredient(name: str) -> str:
    """Strip descriptors and apply manual overrides so USDA searches are clean."""
    name = name.lower().strip()

    # Apply manual override first
    if name in NAME_OVERRIDES:
        return NAME_OVERRIDES[name]

    # Strip noise words
    for word in REMOVE_WORDS:
        name = name.replace(word, "")

    name = " ".join(name.split())  # collapse extra whitespace

    # Re-check overrides after stripping (e.g. "fresh cashew cream" → "cashew cream")
    return NAME_OVERRIDES.get(name, name)

def to_grams(quantity: float, unit: str) -> float | None:
    """
    Convert a quantity + unit into grams.
    Returns None if the unit is unrecognised so the caller can skip cleanly.
    """
    unit = unit.lower().strip()
    factor = UNIT_TO_GRAMS.get(unit)
    if factor is None:
        return None
    return quantity * factor

def search_usda(query: str) -> dict | None:
    """
    Search USDA FoodData Central for a food by name.
    Prefers Foundation foods (most accurate), then SR Legacy, then Branded.
    Returns the best matching food dict or None.
    """
    params = {
        "query":    query,
        "api_key":  USDA_API_KEY,
        "pageSize": 10,
        # Prefer Foundation > SR Legacy > Branded
        "dataType": ["Foundation", "SR Legacy", "Branded"],
    }

    resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    foods = data.get("foods", [])
    if not foods:
        return None

    # Filter out obvious noise
    filtered = [
        f for f in foods
        if not any(bad in f.get("description", "").lower() for bad in FILTER_WORDS)
    ]
    return (filtered or foods)[0]   # best match after filtering

def get_nutrients_per_100g(fdc_id: int) -> dict:
    """
    Fetch the full food record and extract our four target nutrients per 100g.
    USDA Foundation & SR Legacy always express nutrients per 100g — no scaling needed here.
    Returns a dict with keys: calories, protein, carbs, fat (all floats).
    """
    resp = requests.get(
        USDA_FOOD_URL.format(fdc_id=fdc_id),
        params={"api_key": USDA_API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    food = resp.json()

    nutrients_raw = food.get("foodNutrients", [])

    # Build a lookup: nutrient_id → value
    nutrient_map: dict[int, float] = {}
    for n in nutrients_raw:
        # Structure differs slightly between Foundation and Branded
        nutrient_info = n.get("nutrient") or {}
        nid   = nutrient_info.get("id") or n.get("nutrientId")
        value = n.get("amount") or n.get("value") or 0.0
        if nid:
            nutrient_map[int(nid)] = float(value)

    return {
        key: nutrient_map.get(nid, 0.0)
        for key, nid in NUTRIENT_IDS.items()
    }

@tool("get_ingredient_nutrition")
def get_ingredient_nutrition(ingredients: List[Dict]) -> dict:
    """
    Calculate accurate total nutrition for a list of ingredients using the
    USDA FoodData Central database.

    The USDA stores all nutrients per 100g, so the scaling math is exact:
        nutrient_total = (nutrient_per_100g / 100) * quantity_in_grams

    Input — a JSON list of ingredient objects:
    [
      {"name": "chicken breast", "quantity": 200, "unit": "g"},
      {"name": "brown rice",     "quantity": 1,   "unit": "cup"},
      {"name": "olive oil",      "quantity": 1,   "unit": "tbsp"}
    ]

    Rules:
      - Always prefer weight units (g, oz, lb) for accuracy.
      - Use volume units (cup, tbsp, tsp) only when weight is unknown.
      - Never use "piece" — convert to grams before calling this tool.

    Returns:
      totals            — combined calories/protein/carbs/fat for the whole recipe
      per_ingredient    — individual breakdown for each ingredient
      skipped           — ingredients that could not be found (excluded from totals)
      warning           — present if any ingredients were skipped
    """

    if not USDA_API_KEY:
        return {"error": "USDA_API_KEY is not set. Add it to your .env file."}

    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    per_ingredient   = []
    used_ingredients = []
    skipped          = []

    for item in ingredients:
        raw_name = item.get("name", "").strip()
        if not raw_name:
            continue

        try:
            quantity = float(item.get("quantity", 1))
            unit     = str(item.get("unit", "g"))

            # ── Convert to grams ─────────────────────────────────────────
            quantity_grams = to_grams(quantity, unit)
            if quantity_grams is None:
                print(f"  [{raw_name}] Unknown unit '{unit}' — skipping")
                skipped.append({"ingredient": raw_name, "reason": f"unknown unit '{unit}'"})
                continue
            if quantity_grams <= 0:
                print(f"  [{raw_name}] quantity_grams={quantity_grams} — skipping")
                skipped.append({"ingredient": raw_name, "reason": "quantity is zero or negative"})
                continue

            # ── Normalise name for search ────────────────────────────────
            search_name = normalize_ingredient(raw_name)
            print(f"  Searching USDA: '{search_name}'  ({quantity} {unit} = {quantity_grams:.2f}g)")

            # ── USDA search ──────────────────────────────────────────────
            food = search_usda(search_name)

            # Fallback: try the first word only (e.g. "oat milk" → "oat")
            if food is None and " " in search_name:
                fallback = search_name.split()[0]
                print(f"  Fallback search: '{fallback}'")
                food = search_usda(fallback)

            if food is None:
                print(f"  No USDA results for '{raw_name}'")
                skipped.append({"ingredient": raw_name, "reason": "not found in USDA database"})
                continue

            fdc_id      = food["fdcId"]
            description = food.get("description", raw_name)
            print(f"  Matched: '{description}' (fdcId={fdc_id})")

            # ── Fetch nutrients per 100g ─────────────────────────────────
            per_100g = get_nutrients_per_100g(fdc_id)

            # ── Scale to actual quantity ─────────────────────────────────
            # USDA always reports per 100g → scale = quantity_grams / 100
            scale = quantity_grams / 100.0

            cal  = per_100g["calories"] * scale
            prot = per_100g["protein"]  * scale
            carb = per_100g["carbs"]    * scale
            fat  = per_100g["fat"]      * scale

            print(
                f"  per 100g → cal={per_100g['calories']:.1f}  "
                f"prot={per_100g['protein']:.1f}  "
                f"carb={per_100g['carbs']:.1f}  "
                f"fat={per_100g['fat']:.1f}"
            )
            print(
                f"  scaled ({quantity_grams:.1f}g) → "
                f"cal={cal:.1f}  prot={prot:.1f}  carb={carb:.1f}  fat={fat:.1f}"
            )

            totals["calories"] += cal
            totals["protein"]  += prot
            totals["carbs"]    += carb
            totals["fat"]      += fat
            used_ingredients.append(raw_name)

            per_ingredient.append({
                "ingredient":  raw_name,
                "matched_to":  description,
                "quantity":    f"{quantity} {unit}",
                "grams_used":  round(quantity_grams, 2),
                "calories":    round(cal,  1),
                "protein_g":   round(prot, 1),
                "carbs_g":     round(carb, 1),
                "fat_g":       round(fat,  1),
            })

        except requests.exceptions.RequestException as e:
            print(f"  Network error for '{raw_name}': {e}")
            skipped.append({"ingredient": raw_name, "reason": f"network error: {e}"})

        except Exception as e:
            print(f"  Unexpected error for '{raw_name}': {e}")
            skipped.append({"ingredient": raw_name, "reason": str(e)})

    # ── Build response ───────────────────────────────────────────────────────
    if not used_ingredients:
        return {
            "error": (
                "Could not retrieve nutrition for any ingredient. "
                "Try simpler single-word names (e.g. 'rice' not 'cooked jasmine rice')."
            ),
            "skipped": skipped,
        }

    result = {
        "totals": {
            "calories": round(totals["calories"], 1),
            "protein_g": round(totals["protein"],  1),
            "carbs_g":   round(totals["carbs"],    1),
            "fat_g":     round(totals["fat"],      1),
        },
        "per_ingredient": per_ingredient,
        "used":    used_ingredients,
        "skipped": [s["ingredient"] for s in skipped],
    }

    if skipped:
        result["warning"] = (
            f"Nutrition totals exclude: {', '.join(s['ingredient'] for s in skipped)}. "
            "These items were not found or had errors."
        )

    return result

#Identifies dietary constraints based on user input
@tool("classify_dietary_restrictions")
def classify_dietary_restrictions(user_input: str) -> dict:
    """
    Classify dietary restrictions from user input.

    Example:
    Input: "I don't eat dairy or meat"
    Output:
    {
        "dietary_labels": ["vegan"],
        "restrictions": ["dairy", "meat"]
    }
    """
    text = user_input.lower()

    restrictions = []
    labels = []

    # Detect restrictions
    if "dairy" in text or "lactose" in text:
        restrictions.append("dairy")

    if "meat" in text or "chicken" in text or "beef" in text:
        restrictions.append("meat")

    if "gluten" in text:
        restrictions.append("gluten")

    if "nuts" in text or "peanut" in text:
        restrictions.append("nuts")

    # Infer diet type
    if "dairy" in restrictions and "meat" in restrictions:
        labels.append("vegan")
    elif "meat" in restrictions:
        labels.append("vegetarian")

    return {
        "dietary_labels": labels,
        "restrictions": restrictions
    }

#Retrieves information from RAG documents
@tool("retrieve_food_info")
def retrieve_food_info(query: str) -> dict:
    """
    Retrieve allergy and substitution info.

    Example queries:
    - "milk dairy substitute"
    - "peanut allergy substitute"
    """
    print(f" RAG TOOL CALLED with query: {query}\n")
    retriever = get_food_retriever()
    docs = retriever.invoke(query)

    allergies = []
    substitutions = []

    for doc in docs:
        doc_type = doc.metadata.get("type", "unknown")
        content = doc.page_content

        if doc_type == "allergy":
            allergies.append(content)
        elif doc_type == "substitution":
            substitutions.append(content)

    return {
        "allergies": allergies,
        "substitutions": substitutions
    }
