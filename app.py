from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__, static_url_path='/static')

# Load the saved model
loaded_model = joblib.load('Churn_model.joblib')

# Define a function to preprocess the user-input data
def preprocess_input(features):
    features = np.array(features).reshape(1, -1)  # Reshape to a 2D array
    return pd.DataFrame(features, columns=[
        'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage', 'CustServCalls',
        'DayMins', 'DayCalls', 'MonthlyCharge', 'OverageFee', 'RoamMins'
    ])

# Define a route to handle the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract user input from the form
        features = [
            float(request.form['AccountWeeks']),
            int(request.form['ContractRenewal']),
            int(request.form['DataPlan']),
            float(request.form['DataUsage']),
            int(request.form['CustServCalls']),
            float(request.form['DayMins']),
            int(request.form['DayCalls']),
            float(request.form['MonthlyCharge']),
            float(request.form['OverageFee']),
            float(request.form['RoamMins']),
        ]

        # Preprocess the input
        processed_input = preprocess_input(features)

        # Make predictions using the loaded model
        predictions = loaded_model.predict(processed_input)

        # Assuming binary classification
        predicted_class = "Positive" if predictions[0] > 0.5 else "Negative"

        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
