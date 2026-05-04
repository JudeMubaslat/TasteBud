from langchain.tools import tool

from fatsecret import Fatsecret
fs = Fatsecret(consumer_key="509c44855cfa4bfd84be0dfcbce42d99",consumer_secret="6f7903de66bf46e680d7a993e8b3db19")


ALLERGY_DB = {
    "dairy": ["milk", "butter", "cheese", "cream", "yogurt"],
    "gluten": ["flour", "bread", "pasta", "breadcrumbs"],
    "egg": ["egg", "mayonnaise"],
    "nuts": ["almond", "peanut", "cashew", "walnut"],
    "soy": ["soy sauce", "tofu"],
    "shellfish": ["shrimp", "crab", "lobster"]
}

SUBSTITUTION_DB = {
    "milk": {
        "default": ["evaporated milk", "half-and-half"],
        "vegan": ["almond milk", "oat milk", "soy milk"],
        "dairy": ["almond milk", "oat milk", "coconut milk"]
    },
    "cheese": {
        "default": ["mozzarella", "cheddar"],
        "vegan": ["vegan cheese", "cashew cream"],
        "dairy": ["vegan cheese", "cashew cream"]
    },
    "egg": {
        "default": ["yogurt", "applesauce"],
        "vegan": ["flaxseed + water", "chia egg"],
        "egg": ["flaxseed + water", "silken tofu"]
    },
    "butter": {
        "default": ["margarine", "olive oil"],
        "vegan": ["olive oil", "coconut oil"],
        "dairy": ["margarine", "olive oil"]
    },
    "flour": {
        "default": ["all-purpose flour"],
        "gluten": ["almond flour", "rice flour"]
    }
}

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
def check_allergy(ingredients: list[str], dietary_restrictions: list[str]) -> list:
    """
    Check if any ingredients violate dietary restrictions.

    Inputs:
    - ingredients: list of ingredient names
    - dietary_restrictions: list like ["dairy", "gluten"]

    Returns:
    - list of ingredients that violate restrictions

    Always call this tool before finalizing a recipe if restrictions are provided.
    """

    violations = []

    for ing in ingredients:
        ing_lower = ing.lower()

        for restriction in dietary_restrictions:
            banned_list = ALLERGY_DB.get(restriction.lower(), [])

            for banned in banned_list:
                if banned in ing_lower:
                    violations.append(ing)
                    break

    return list(set(violations))

@tool
def suggest_substitution(
    ingredient: str,
    dietary_restrictions: list[str] = []
) -> list:
    """
    Suggest substitutions for an ingredient.

    Inputs:
    - ingredient: ingredient to replace (e.g., "cheese")
    - dietary_restrictions: list like ["vegan", "dairy"]

    Returns:
    - list of up to 3 substitutes

    Always use this tool if an ingredient violates dietary restrictions.
    """

    ingredient = ingredient.lower()
    dietary_restrictions = [d.lower() for d in dietary_restrictions]

    # Fuzzy match
    matched_key = None
    for key in SUBSTITUTION_DB:
        if key in ingredient or ingredient in key:
            matched_key = key
            break

    if not matched_key:
        return []

    options = SUBSTITUTION_DB[matched_key]

    suggestions = []

    # Try restriction-specific
    for restriction in dietary_restrictions:
        if restriction in options:
            suggestions.extend(options[restriction])

    # Fallback
    if not suggestions:
        suggestions = options.get("default", [])

    return suggestions[:3]
