from flask import Blueprint, render_template, request
import pandas as pd
import joblib
import openai

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    year = int(request.form['year'])
    month = int(request.form['month'])
    avg_temp = float(request.form['avg_temp'])
    country = request.form['country']
    
    model = joblib.load('model/model.pkl')

    # Prepare input data
    data = {'Year': [year], 'Month': [month], 'AverageTemperature': [avg_temp], 'Country': [country]}
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    
    # Align the dataframe with the model's input
    model_columns = joblib.load('model/model_columns.pkl')
    df = df.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df)

    # Get explanation using GPT-3.5
    explanation = get_gpt3_explanation(year, month, avg_temp, country, prediction[0])

    return render_template('result.html', prediction=prediction[0], explanation=explanation)

def get_gpt3_explanation(year, month, avg_temp, country, prediction):
    openai.api_key = 'your_openai_api_key'
    prompt = f"Year: {year}, Month: {month}, Average Temperature: {avg_temp}, Country: {country}, Predicted Uncertainty: {prediction}. Explain why this might be happening."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()
