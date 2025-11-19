import pandas as pd
import joblib
from flask import Flask, request, render_template
from datetime import datetime

# --- Configuration ---
app = Flask(__name__)

# Load the trained model and the start date needed for feature engineering
try:
    model = joblib.load('maize_disease_model.joblib')
    start_date = joblib.load('start_date.joblib')
except FileNotFoundError:
    print("Error: Model or start date file not found. Ensure 'maize_disease_model.joblib' and 'start_date.joblib' are in the same directory.")
    exit()

# Define the unique stages for the dropdown menu (adjust if needed, based on your data)
UNIQUE_STAGES = [
    'Germination', 'Seedling', 'Vegetative Growth',
    'Tasseling & Pollination', 'Grain Filling', 'Maturity & Harvest'
]

# --- Feature Engineering Function ---
def create_features(date_str, stage):
    """
    Transforms raw inputs (date and stage) into the features required by the model.
    """
    try:
        # Convert date string (format YYYY-MM-DD from HTML input) to datetime object
        current_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None, "Invalid date format. Please use YYYY-MM-DD."

    # 1. DayOfYear
    day_of_year = current_date.timetuple().tm_yday

    # 2. DaysSinceStart (requires a datetime object for start_date)
    days_since_start = (current_date - start_date).days

    # Create a DataFrame matching the structure the pipeline expects
    data = pd.DataFrame({
        'Stage': [stage],
        'DayOfYear': [day_of_year],
        'DaysSinceStart': [days_since_start]
    })
    
    return data, None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the home page with the prediction form."""
    return render_template('index.html', stages=UNIQUE_STAGES, prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and returns the prediction."""
    
    # Get inputs from the form
    date_input = request.form.get('date_input')
    stage_input = request.form.get('stage_input')
    
    if not date_input or not stage_input:
        return render_template('index.html', stages=UNIQUE_STAGES, 
                               prediction_text="Error: Please provide both date and stage.", 
                               date_val=date_input, stage_val=stage_input)

    # Transform inputs into model features
    X_new, error_message = create_features(date_input, stage_input)
    
    if error_message:
        return render_template('index.html', stages=UNIQUE_STAGES, 
                               prediction_text=f"Error: {error_message}", 
                               date_val=date_input, stage_val=stage_input)

    # Make prediction using the loaded pipeline
    prediction = model.predict(X_new)[0]
    
    result = f"The predicted disease for stage '{stage_input}' on {date_input} is: {prediction}"

    return render_template('index.html', stages=UNIQUE_STAGES, 
                           prediction_text=result, 
                           date_val=date_input, stage_val=stage_input)

# --- Run the App ---
if __name__ == "__main__":
    # Use a secure and non-default port in production, but 5000 is standard for development
    app.run(debug=True)