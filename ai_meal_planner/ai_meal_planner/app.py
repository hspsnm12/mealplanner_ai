import streamlit as st
import pandas as pd
import numpy as np
import cohere 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
api_key = ''
co = cohere.Client(api_key)  

# Function to generate meal plan using Cohere API
def generate_meal_plan(prompt):
    """Generate a meal plan using Cohere API."""
    try:
        response = co.generate(
            model="command",  # Use an appropriate Cohere model
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
        )
        return response.generations[0].text if response.generations else "Could not generate meal plan. Try again."
    except Exception as e:
        return f"Error generating meal plan: {e}"

# Load cleaned dataset
df_cleaned = pd.read_csv("cleaned_recipes.csv")

df_cleaned["ingredients"].fillna("", inplace=True)
df_cleaned["recipe_name"].fillna("", inplace=True)

# Function to calculate BMR
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:
        return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

def get_daily_calories(bmr, activity_level, goal):
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Super active": 1.9
    }
    daily_calories = bmr * activity_multipliers.get(activity_level, 1.2)
    if goal == "Lose weight":
        daily_calories -= 500
    elif goal == "Gain weight":
        daily_calories += 500
    return daily_calories

# Streamlit UI
st.title("AI Meal Planner")
st.sidebar.header("User Details")
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=154)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=50)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
goal = st.sidebar.selectbox("Goal", ["Maintain weight", "Lose weight", "Gain weight"])
diet_preference = st.sidebar.selectbox("Diet Preference", ["Veg", "Non-Veg", "Vegan"])

bmr = calculate_bmr(weight, height, age, gender)
daily_calories = get_daily_calories(bmr, activity_level, goal)
st.sidebar.write(f"### Recommended Daily Calories: {int(daily_calories)} kcal")

# Macronutrient Preferences
st.sidebar.header("Macronutrient Preferences")
fats = st.sidebar.slider("Fat (g)", 0, 100, 50)
carbs = st.sidebar.slider("Carbs (g)", 0, 300, 150)
protein = st.sidebar.slider("Protein (g)", 0, 200, 75)

# Filter recipes based on dietary preference
if diet_preference == "Veg":
    df_filtered = df_cleaned[~df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False)]
elif diet_preference == "Non-Veg":
    df_filtered = df_cleaned[df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False)]
else:  # Vegan
    df_filtered = df_cleaned[~df_cleaned["ingredients"].str.contains("chicken|fish|egg|mutton|dairy|cheese|milk|butter|tuna|salmon", case=False, na=False)]

features = ["fat_g", "carbs_g", "protein_g", "estimated_calories"]
df_filtered = df_filtered.dropna(subset=features)

if df_filtered.empty:
    st.warning("No recipes found matching your criteria. Please adjust your selections.")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered[features])
    knn_model = KNeighborsRegressor(n_neighbors=10, metric='euclidean')
    knn_model.fit(X_scaled, df_filtered[features])
    user_input = np.array([[fats, carbs, protein, daily_calories]])
    user_scaled = scaler.transform(user_input)
    distances, indices = knn_model.kneighbors(user_scaled)

    st.header("Recommended Recipes")
    for idx in indices[0]:
        recipe = df_filtered.iloc[idx]
        st.subheader(recipe["recipe_name"])
        st.write(f"Calories: {recipe['estimated_calories']} kcal")
        st.write(f"Fats: {recipe['fat_g']} g | Carbs: {recipe['carbs_g']} g | Protein: {recipe['protein_g']} g")
        st.write(f"Servings: {recipe['servings']} | Total Time: {recipe['total_time']} mins")
        st.image(recipe["img_src"], width=300)
        st.write(f"Ingredients: {recipe['ingredients']}")
        st.write(f"Recipe Instructions: {recipe.get('directions', 'Instructions not available')}")
        recipe_txt = f"{recipe['recipe_name']}\nCalories: {recipe['estimated_calories']} kcal\nIngredients: {recipe['ingredients']}\nInstructions: {recipe.get('directions', 'Instructions not available')}"
        st.download_button("Download Recipe", recipe_txt, file_name=f"{recipe['recipe_name']}.txt", key=idx)

# AI Meal Plan Generation
st.header("AI-Generated Meal Plan")
prompt = st.text_area("Enter your meal requirements:", "I am 20 years old, height:154cm, weight:50kg. I want a vegetarian Indian meal.")
if st.button("Generate Meal Plan"):
    meal_plan = generate_meal_plan(prompt)
    st.write(meal_plan)

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors

# # Load cleaned dataset
# df_cleaned = pd.read_csv("cleaned_recipes.csv")


# # Function to calculate BMR using Mifflin-St Jeor Equation
# def calculate_bmr(weight, height, age, gender):
#     if gender == "Male":
#         return 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
#     else:
#         return 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

# # Adjust BMR based on activity level
# def get_daily_calories(bmr, activity_level, goal):
#     activity_multipliers = {
#         "Sedentary": 1.2,
#         "Lightly active": 1.375,
#         "Moderately active": 1.55,
#         "Very active": 1.725,
#         "Super active": 1.9
#     }
    
#     daily_calories = bmr * activity_multipliers.get(activity_level, 1.2)
    
#     if goal == "Lose weight":
#         daily_calories -= 500
#     elif goal == "Gain weight":
#         daily_calories += 500
    
#     return daily_calories

# # Streamlit UI with CSS styling
# st.markdown("""
#     <style>
#         body {
#             background-color: #f4f4f4;
#         }
#         .sidebar .sidebar-content {
#             background-color: #fafafa;
#         }
#         h1, h2, h3 {
#             color: #2c3e50;
#         }
#         .stButton>button {
#             background-color: #4CAF50;
#             color: white;
#             border-radius: 5px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.title("AI Meal Planner")

# # User Inputs
# st.sidebar.header("User Details")
# height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=160)
# weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=60)
# age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
# gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
# activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Super active"])
# goal = st.sidebar.selectbox("Goal", ["Maintain weight", "Lose weight", "Gain weight"])
# diet_preference = st.sidebar.selectbox("Diet Preference", ["Veg", "Non-Veg", "Vegan"])

# # Calculate recommended daily calories
# bmr = calculate_bmr(weight, height, age, gender)
# daily_calories = get_daily_calories(bmr, activity_level, goal)
# st.sidebar.write(f"### Recommended Daily Calories: {int(daily_calories)} kcal")

# # User Macro Preferences
# st.sidebar.header("Macronutrient Preferences")
# fats = st.sidebar.slider("Fat (g)", 0, 100, 50)
# carbs = st.sidebar.slider("Carbs (g)", 0, 300, 150)
# protein = st.sidebar.slider("Protein (g)", 0, 200, 75)

# # Filter recipes based on dietary preference (check both ingredients & name)
# def filter_recipes(df, diet):
#     if diet == "Veg":
#         return df[~df["ingredients"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False) & 
#                  ~df["recipe_name"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False)]
#     elif diet == "Non-Veg":
#         return df[df["ingredients"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False) | 
#                  df["recipe_name"].str.contains("chicken|fish|egg|mutton|tuna|salmon", case=False, na=False)]
#     else:  # Vegan
#         return df[~df["ingredients"].str.contains("chicken|fish|egg|mutton|dairy|cheese|milk|butter|tuna|salmon", case=False, na=False) & 
#                  ~df["recipe_name"].str.contains("chicken|fish|egg|mutton|dairy|cheese|milk|butter|tuna|salmon", case=False, na=False)]

# # Apply filtering
# df_filtered = filter_recipes(df_cleaned, diet_preference)

# # Feature scaling for KNN
# features = ["fat_g", "carbs_g", "protein_g", "estimated_calories"]
# df_filtered = df_filtered.dropna(subset=features)

# if df_filtered.empty:
#     st.warning("No recipes found matching your criteria. Please adjust your selections.")
# else:
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df_filtered[features])
    
#     knn_model = NearestNeighbors(n_neighbors=8, metric='euclidean')
#     knn_model.fit(X_scaled)

#     user_input = np.array([[fats, carbs, protein, daily_calories]])
#     user_scaled = scaler.transform(user_input)
#     distances, indices = knn_model.kneighbors(user_scaled)

#     # Display recommended recipes
#     st.header("Recommended Recipes")
#     for idx, recipe_idx in enumerate(indices[0]):
#         recipe = df_filtered.iloc[recipe_idx]
#         st.subheader(recipe["recipe_name"])
#         st.write(f"**Calories:** {recipe['estimated_calories']} kcal")
#         st.write(f"**Fats:** {recipe['fat_g']} g | **Carbs:** {recipe['carbs_g']} g | **Protein:** {recipe['protein_g']} g")
#         st.write(f"**Servings:** {recipe['servings']}")
#         st.write(f"**Total Time:** {recipe['total_time']} minutes")
#         st.image(recipe["img_src"], width=300)
#         st.write(f"**Ingredients:** {recipe['ingredients']}")
#         st.write(f"**Instructions:** {recipe.get('directions', 'Instructions not available')}")

#         # Button to download recipe as TXT (Fixed issue with unique keys)
#         recipe_text = f"Recipe Name: {recipe['recipe_name']}\nCalories: {recipe['estimated_calories']} kcal\nServings: {recipe['servings']}\nTotal Time: {recipe['total_time']} mins\nIngredients: {recipe['ingredients']}\nInstructions: {recipe.get('directions', 'Not available')}"
#         st.download_button(
#             label="Download Recipe", 
#             data=recipe_text, 
#             file_name=f"{recipe['recipe_name']}.txt", 
#             mime="text/plain",
#             key=f"download_{idx}"  # Unique key to prevent StreamlitDuplicateElementId error
#      )