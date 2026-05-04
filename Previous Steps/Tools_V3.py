from langchain.tools import tool
from build_faiss_index import get_retriever
retriever = get_retriever()

from fatsecret import Fatsecret
fs = Fatsecret(consumer_key="509c44855cfa4bfd84be0dfcbce42d99",consumer_secret="6f7903de66bf46e680d7a993e8b3db19")

@tool
def get_ingredient_nutrition(ingredients: list[dict]) -> dict:
    """
       Calculate approximate total nutrition for a recipe using ingredient quantities.

       Input:
       - ingredients: list of objects with:
           - name: string (e.g., "chicken breast")
           - quantity: number (e.g., 200)
           - unit: string (e.g., "g", "piece", "cup", "tbsp")

       Example:
       [
         {"name": "chicken breast", "quantity": 200, "unit": "g"},
         {"name": "onion", "quantity": 1, "unit": "piece"},
         {"name": "rice", "quantity": 1, "unit": "cup"}
       ]

       Behavior:
       - Converts ingredient quantities into grams using approximate unit conversions
       - Retrieves nutrition data per serving from the FatSecret API
       - Scales nutrition values based on the ratio of quantity (grams) to serving size
       - Sums total calories, protein, carbohydrates, and fat across all ingredients

       IMPORTANT:
       - Always call this tool AFTER determining ingredient quantities
       - Use realistic cooking quantities and clear units
       - The same ingredient quantities MUST appear in the final recipe output
       - Nutrition values are estimates and may vary based on ingredient matching

       Returns:
       {
         "calories": float,
         "protein": float,
         "carbs": float,
         "fat": float
       }
       """
    totals = {
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0
    }

    UNIT_TO_GRAMS = {
        "g": 1, "gram": 1, "grams": 1,
        "kg": 1000,
        "piece": 50,
        "tortilla": 60,
        "cup": 240,
        "tbsp": 15,
        "tsp": 5
    }

    for item in ingredients:
        try:
            name = item.get("name", "")
            quantity = float(item.get("quantity", 1))
            unit = item.get("unit", "g").lower()

            # 🔹 Convert to grams
            if unit in UNIT_TO_GRAMS:
                quantity_grams = quantity * UNIT_TO_GRAMS[unit]
            else:
                quantity_grams = quantity  # fallback

            foods = fs.foods_search(name)
            if not foods:
                continue

            food = fs.food_get(foods[0]["food_id"])
            serving = food["servings"]["serving"]

            if isinstance(serving, list):
                serving = serving[0]

            calories = float(serving.get("calories", 0))
            protein = float(serving.get("protein", 0))
            carbs = float(serving.get("carbohydrate", 0))
            fat = float(serving.get("fat", 0))

            # 🔹 FIX: scale by serving size
            serving_size = float(serving.get("metric_serving_amount", 100))

            scale = quantity_grams / serving_size if serving_size else 1

            totals["calories"] += calories * scale
            totals["protein"] += protein * scale
            totals["carbs"] += carbs * scale
            totals["fat"] += fat * scale

        except Exception as e:
            print(f"Error with {item}: {e}")

    return totals

@tool
def retrieve_food_info(query: str) -> dict:
    """
    Retrieve allergy and substitution information for ingredients.

    Input:
        query: ingredient or dietary restriction (e.g. "milk dairy substitute")

    Returns:
        {
            "allergies": [ ... ],
            "substitutions": [ ... ]
        }
    """
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

