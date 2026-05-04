from langchain.tools import tool

from fatsecret import Fatsecret
fs = Fatsecret(consumer_key="509c44855cfa4bfd84be0dfcbce42d99",consumer_secret="6f7903de66bf46e680d7a993e8b3db19")

@tool
def get_ingredient_nutrition(ingredients: list[str]) -> dict:
    """
    Get combined nutrition for a list of ingredients.
    """

    totals = {
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0
    }

    for name in ingredients:
        try:
            foods = fs.foods_search(name)
            if not foods:
                continue

            food = fs.food_get(foods[0]["food_id"])
            serving = food["servings"]["serving"]

            if isinstance(serving, list):
                serving = serving[0]

            totals["calories"] += float(serving.get("calories", 0))
            totals["protein"] += float(serving.get("protein", 0))
            totals["carbs"] += float(serving.get("carbohydrate", 0))
            totals["fat"] += float(serving.get("fat", 0))

        except Exception as e:
            print(f"Error with {name}: {e}")

    return totals