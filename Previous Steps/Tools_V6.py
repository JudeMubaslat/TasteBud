#Importing libraries
import os
from langchain.tools import tool
from typing import List, Dict
from fatsecret import Fatsecret
from dotenv import load_dotenv

load_dotenv()  # this loads the .env file

consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")

#API Setup
fs = Fatsecret(consumer_key=consumer_key, consumer_secret=consumer_secret)

def get_food_retriever():
    from build_faiss_index import get_retriever
    return get_retriever()

#Normalizing ingredients, so they can be searched properly in the FatSecret API
def normalize_ingredient(name: str) -> str:
    name = name.lower()

    remove_words = [
        "fresh", "chopped", "diced", "sliced",
        "cooked", "grilled", "roasted",
        "seasoned", "mixture", "sauce",
        "filling", "topping"
    ]

    for word in remove_words:
        name = name.replace(word, "")

    name = name.strip()

    replacements = {
        "cashew cream": "cashews",
        "vegan cheese": "cheese",
        "plant-based milk": "milk",
        "black bean mixture": "black beans",
        "bean filling": "beans",
        "taco filling": "beans"
    }

    return replacements.get(name, name)

@tool("get_ingredient_nutrition")
def get_ingredient_nutrition(ingredients: List[Dict]) -> dict:
    """
    Calculate total nutrition for a list of ingredients using the FatSecret API.
    Always call this AFTER substitutions have been applied.

    Input format (valid JSON list):
    [
      {"name": "chicken breast", "quantity": 200, "unit": "g"},
      {"name": "rice", "quantity": 1, "unit": "cup"}
    ]

    Prefer weight units (g, oz) over "piece" for accuracy.
    """

    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    used_ingredients = []
    skipped_ingredients = []

    UNIT_TO_GRAMS = {
        "g": 1, "gram": 1, "grams": 1,
        "oz": 28.35,
        "lb": 453.6,
        "kg": 1000,
        "cup": 240,
        "tbsp": 15,
        "tsp": 5,
        "ml": 1,
        "l": 1000,
        # "piece" is intentionally excluded — the model should use g/oz instead
    }

    filter_words = ["keto", "low carb", "protein bar", "diet", "supplement", "powder"]

    for item in ingredients:
        try:
            raw_name = item.get("name", "")
            name = normalize_ingredient(raw_name)
            quantity = float(item.get("quantity", 1))
            unit = item.get("unit", "g").lower()

            quantity_grams = quantity * UNIT_TO_GRAMS.get(unit, 1)

            print(f"  Searching FatSecret for: '{name}'")

            foods = fs.foods_search(name)

            # Filter out product/supplement noise
            filtered = [
                f for f in foods
                if not any(bad in f["food_name"].lower() for bad in filter_words)
            ]
            foods = filtered if filtered else foods

            # Fallback: try just the first word (e.g. "oat milk" → "oat")
            if not foods:
                fallback = name.split()[0]
                print(f"  Fallback search: '{fallback}'")
                foods = fs.foods_search(fallback)

            if not foods:
                print(f"  No results for: '{raw_name}'")
                skipped_ingredients.append(raw_name)
                continue

            food = fs.food_get(foods[0]["food_id"])
            serving = food["servings"]["serving"]

            if isinstance(serving, list):
                serving = serving[0]

            calories = float(serving.get("calories", 0))
            protein = float(serving.get("protein", 0))
            carbs = float(serving.get("carbohydrate", 0))
            fat = float(serving.get("fat", 0))
            serving_size = float(serving.get("metric_serving_amount", 100))

            scale = quantity_grams / serving_size if serving_size else 1

            totals["calories"] += calories * scale
            totals["protein"] += protein * scale
            totals["carbs"] += carbs * scale
            totals["fat"] += fat * scale
            used_ingredients.append(name)

        except Exception as e:
            print(f"  Error processing '{item}': {e}")
            skipped_ingredients.append(item.get("name", "unknown"))

    if totals["calories"] == 0:
        return {
            "error": "Nutrition lookup failed for all ingredients.",
            "used": used_ingredients,
            "skipped": skipped_ingredients
        }

    return {
        "calories": round(totals["calories"], 2),
        "protein": round(totals["protein"], 2),
        "carbs": round(totals["carbs"], 2),
        "fat": round(totals["fat"], 2),
        "used": used_ingredients,
        "skipped": skipped_ingredients
    }

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
    print(f"\n🔎 RAG TOOL CALLED with query: {query}\n")
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
