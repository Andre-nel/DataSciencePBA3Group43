import pandas as pd
import pickle
from dash.dependencies import Input, Output, State
from pathlib import Path

pickledDir = Path(__file__).parent.parent / "models" / "pickled"
with open(pickledDir / "linear_regression_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
with open(pickledDir / "linear_regression_scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)
with open(pickledDir / "linear_regression_features.pkl", "rb") as file:
    loaded_features = pickle.load(file)


def predict_spending_score(age, annual_income, work_experience, family_size, gender, profession):
    # Create a dictionary of default feature values
    default_features = {}
    for feature in loaded_features:
        default_features[feature] = None

    # Update default feature values with the given values
    if age is not None:
        default_features['Age'] = age
    if annual_income is not None:
        default_features['Annual Income ($)'] = annual_income
    if work_experience is not None:
        default_features['Work Experience'] = work_experience
    if family_size is not None:
        default_features['Family Size'] = family_size
    if gender is not None:
        default_features['Gender_Female'] = 1 if gender == "Female" else 0
        default_features['Gender_Male'] = 1 if gender == "Male" else 0

    # Update profession features
    for key in default_features.keys():
        if key.startswith("Profession_"):
            default_features[key] = 1 if key == f"Profession_{profession}" else 0

    # Convert the dictionary to a DataFrame
    input_features = pd.DataFrame(default_features, index=[0])

    # Normalize the input features using the loaded scaler
    for column in ["Age", "Annual Income ($)", "Work Experience", "Family Size"]:
        input_features[column] = loaded_scaler.transform(input_features[column].values.reshape(-1, 1))

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_features)

    return f"Predicted Spending Score: {prediction[0]:.2f}"


def register_callbacks(app):
    @app.callback(
        Output("prediction-result", "children"),
        [Input("btn-predict", "n_clicks")],
        [State("input-age", "value"),
         State("input-annual-income", "value"),
         State("work-experience", "value"),
         State("family-size", "value"),
         State("gender", "value"),
         State("profession", "value")]
    )
    def update_prediction(n_clicks, age, annual_income, work_experience, family_size, gender, profession):
        if not n_clicks:
            return ""

        if not all([age, annual_income, work_experience, family_size, gender, profession]):
            return "Please fill in all the fields."

        try:
            age = int(age)
            annual_income = float(annual_income)
            work_experience = int(work_experience)
            family_size = int(family_size)
        except ValueError:
            return "Please enter valid values for age, annual income, work experience, and family size."

        if gender not in ("Male", "Female"):
            return "Please select a valid gender."

        professions = []
        for feature in loaded_features:
            if feature.startswith("Profession_"):
                professions.append(feature.split("_", 1)[-1])

        if profession not in professions:
            return "Please select a valid profession."

        return predict_spending_score(age, annual_income, work_experience, family_size, gender, profession)
