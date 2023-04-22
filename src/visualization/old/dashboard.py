import pickle
import pandas as pd
import plotly.express
import plotly.graph_objects
import dash
from dash import dcc, html
import dash_bootstrap_components
from dash.dependencies import Input, Output, State
from pathlib import Path


projectRoot = Path(__file__).parent / "../.."
data = pd.read_csv(projectRoot/"data/processed/customers.csv")

# Load the model
pickledDir = Path(__file__).parent.parent / "models" / "pickled"
with open(pickledDir / "linear_regression_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
with open(pickledDir / "linear_regression_scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)
with open(pickledDir / "linear_regression_features.pkl", "rb") as file:
    loaded_features = pickle.load(file)

# Create a dictionary of default feature values
default_features = {}
professions = []
for feature in loaded_features:
    default_features[feature] = None
    if feature.startswith("Profession_"):
        professions.append(feature.split("_", 1)[-1])


# Create the individual plots:

# Demographics Overview
gender_pie = plotly.express.pie(data, names="Gender", title="Gender Distribution")
age_histogram = plotly.express.histogram(data, x="Age", nbins=10, title="Age Distribution")

# Income and Spending Score Distribution
income_histogram = plotly.express.histogram(data, x="Annual Income ($)", nbins=10, title="Annual Income Distribution")
spending_histogram = plotly.express.histogram(
    data, x="Spending Score (1-100)", nbins=10, title="Spending Score Distribution")

# Profession Distribution
profession_bar = plotly.express.bar(data, x="Profession", title="Profession Distribution")

# Work Experience and Family Size
work_experience_histogram = plotly.express.histogram(
    data, x="Work Experience", nbins=10, title="Work Experience Distribution")
family_size_histogram = plotly.express.histogram(data, x="Family Size", nbins=5, title="Family Size Distribution")

# Customer Segmentation
customer_segmentation_scatter = plotly.express.scatter(data, x="Annual Income ($)", y="Spending Score (1-100)",
                                                       color="Profession", title="Customer Segmentation")

# Correlation Matrix
correlation_matrix = plotly.graph_objects.Figure(data=plotly.graph_objects.Heatmap(z=data.corr(
    numeric_only=True), x=data.columns, y=data.columns, colorscale="Viridis", showscale=True))

correlation_matrix.update_layout(title="Correlation Matrix")

# Create the Dash app and layout:

app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])


app.layout = dash_bootstrap_components.Container([
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dcc.Graph(figure=gender_pie), width=6),
        dash_bootstrap_components.Col(dcc.Graph(figure=age_histogram), width=6),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dcc.Graph(figure=income_histogram), width=6),
        dash_bootstrap_components.Col(dcc.Graph(figure=spending_histogram), width=6),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dcc.Graph(figure=profession_bar), width=12),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dcc.Graph(figure=work_experience_histogram), width=6),
        dash_bootstrap_components.Col(dcc.Graph(figure=family_size_histogram), width=6),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dcc.Graph(figure=customer_segmentation_scatter), width=12),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dcc.Graph(figure=correlation_matrix), width=12),
    ]),
    # Add row for the prediction form
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col([
            html.H4("Predict Spending Score"),
            dash_bootstrap_components.Form([
                dash_bootstrap_components.Row([
                    dash_bootstrap_components.Col([
                        dash_bootstrap_components.Label("Age"),
                        dcc.Input(id="input-age", type="number", placeholder="Enter Age", value=25),
                    ]),
                ]),
                dash_bootstrap_components.Row([
                    dash_bootstrap_components.Col([
                        dash_bootstrap_components.Label("Annual Income ($)"),
                        dcc.Input(id="input-annual-income", type="number",
                                  placeholder="Enter Annual Income ($)", value=50000),
                    ]),
                ]),
                dash_bootstrap_components.Row([
                    dash_bootstrap_components.Col([
                        dash_bootstrap_components.Label("Work Experience"),
                        dcc.Input(id="work-experience", type="number",
                                  placeholder="Enter Work Experience (yrs)", value=5),
                    ]),
                ]),
                dash_bootstrap_components.Row([
                    dash_bootstrap_components.Col([
                        dash_bootstrap_components.Label("Family Size"),
                        dcc.Input(id="family-size", type="number",
                                  placeholder="Enter Family Size", value=3),
                    ]),
                ]),
                dash_bootstrap_components.Row([
                    dash_bootstrap_components.Col([
                        dash_bootstrap_components.Label("Gender"),
                        dcc.Dropdown(id="gender",
                                     options=[{"label": "Male", "value": "Male"},
                                              {"label": "Female", "value": "Female"}],
                                     value="Male",
                                     placeholder="Select Gender"),
                    ]),
                ]),
                dash_bootstrap_components.Row([
                    dash_bootstrap_components.Col([
                        dash_bootstrap_components.Label("Profession"),
                        dcc.Dropdown(id="profession",
                                     options=[{"label": prof, "value": prof} for prof in professions],
                                     value="Engineer",
                                     placeholder="Select Profession"),
                    ]),
                ]),
                # Add more rows and columns for other features
                dash_bootstrap_components.Button("Predict", id="btn-predict", color="primary", className="mt-2"),
            ]),
            html.Div(id="prediction-result", className="mt-4"),
        ]),
    ]),
])


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
def predict_spending_score(n_clicks, age, annual_income, work_experience, family_size, gender, profession):
    if (n_clicks is None or age is None or annual_income is None
            or work_experience is None or family_size is None or gender is None or profession is None):
        return ""

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
    for column in ['Age', 'Annual Income ($)']:
        input_features[column] = loaded_scaler.transform(input_features[column].values.reshape(-1, 1))

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_features)

    return f"Predicted Spending Score: {prediction[0]:.2f}"


if __name__ == "__main__":
    app.run_server(debug=True)
