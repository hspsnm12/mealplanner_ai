# AI MEAL PLANNER

## Overview
AI Meal Planner is a smart recipe recommendation system that suggests meal plans based on user preferences. It leverages **Cohere's NLP API** for text generation and **K-Nearest Neighbors (KNN)** for personalized recipe recommendations. The web application is built using **Streamlit** for an interactive user experience.

## Features
- Generates meal plans using AI (Cohere API)
- Recommends recipes based on user preferences
- Uses **KNN regression** for similarity-based recommendations
- Displays cleaned recipe data for better accuracy
- Interactive and user-friendly interface with **Streamlit**

## Tech Stack
- **Python** (for data processing and model implementation)
- **Streamlit** (for UI)
- **Cohere API** (for text generation)
- **Pandas, NumPy** (for data manipulation)
- **scikit-learn** (for KNN implementation)

## Installation
### Prerequisites
Ensure you have Python installed. You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Application
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd AI_Meal_Planner
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Enter your meal preferences in the provided fields.
- Click "Generate Meal Plan" to receive AI-generated recommendations.
- View recommended recipes based on similarity to previous choices.
- Explore nutritional information and other recipe details.

## Dataset
- The project uses a cleaned dataset (`cleaned_recipes.csv`) containing structured recipe data.
- The dataset has been preprocessed to improve recommendation accuracy.

## API Key Configuration
To use the Cohere API, set up your API key in `app.py`:
```python
api_key = "your_cohere_api_key"
```

## Future Enhancements
- Adding more advanced filtering options (e.g., dietary restrictions)
- Integrating additional NLP models for improved text generation
- Enhancing UI with more interactive elements



