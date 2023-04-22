import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.express as px
import pandas as pd
from pathlib import Path
import numpy as np
import pickle


def layout(app):
    projectRoot = Path(__file__).parent / "../.."
    data = pd.read_csv(projectRoot/"data/processed/customers.csv")
    data = data.drop('CustomerID', axis=1)
    pickledDir = Path(__file__).parent.parent / "models" / "pickled"

    with open(pickledDir / "linear_regression_features.pkl", "rb") as file:
        loaded_features = pickle.load(file)

    # Professions
    professions = []
    for feature in loaded_features:
        if feature.startswith("Profession_"):
            professions.append(feature.split("_", 1)[-1])

    # Demographics Overview
    gender_pie = px.pie(data, names="Gender", title="Gender Distribution")
    age_histogram = px.histogram(data, x="Age", nbins=10, title="Age Distribution")

    # Income and Spending Score Distribution
    income_histogram = px.histogram(data, x="Annual Income ($)", nbins=10, title="Annual Income Distribution")
    spending_histogram = px.histogram(data, x="Spending Score (1-100)", nbins=10, title="Spending Score Distribution")

    # Profession Distribution
    profession_bar = px.bar(data, x="Profession", title="Profession Distribution")

    # Work Experience and Family Size
    work_experience_histogram = px.histogram(data, x="Work Experience", nbins=10, title="Work Experience Distribution")
    family_size_histogram = px.histogram(data, x="Family Size", nbins=5, title="Family Size Distribution")

    # Customer Segmentation
    customer_segmentation_scatter = px.scatter(data, x="Annual Income ($)", y="Spending Score (1-100)",
                                                       color="Profession", title="Customer Segmentation")

    # Correlation Matrix
    numeric_columns = data.select_dtypes(include=np.number).columns
    correlation_matrix = px.imshow(data[numeric_columns].corr(), x=numeric_columns,
                                   y=numeric_columns, color_continuous_scale="Viridis")
    correlation_matrix.update_layout(title="Correlation Matrix")

    prediction_form = dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Age"),
                dbc.Input(type="number", id="input-age",
                          placeholder="Enter Age",
                          value=25),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Annual Income ($)"),
                dbc.Input(type="number", id="input-annual-income",
                          placeholder="Enter Annual Income ($)",
                          value=50000),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Work Experience"),
                dbc.Input(type="number", id="work-experience",
                          placeholder="Enter Work Experience (yrs)",
                          value=5),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Family Size"),
                dbc.Input(type="number", id="family-size",
                          placeholder="Enter Family Size",
                          value=3),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Gender"),
                dcc.Dropdown(id="gender",
                             options=[{"label": "Male", "value": "Male"},
                                      {"label": "Female", "value": "Female"}],
                             value="Male",
                             placeholder="Select Gender",
                             ),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Profession"),
                dcc.Dropdown(id="profession",
                             options=[{"label": prof, "value": prof} for prof in professions],
                             value="Engineer",
                             placeholder="Select Profession",
                             ),
            ])
        ]),
        dbc.Button("Predict", id="btn-predict", color="primary", className="mt-2"),
        html.Div(id="prediction-result", className="mt-4"),
    ])

    return dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=gender_pie), width=6),
            dbc.Col(dcc.Graph(figure=age_histogram), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=income_histogram), width=6),
            dbc.Col(dcc.Graph(figure=spending_histogram), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=profession_bar), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=work_experience_histogram), width=6),
            dbc.Col(dcc.Graph(figure=family_size_histogram), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=customer_segmentation_scatter), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=correlation_matrix), width=12),
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Predict Spending Score"),
                prediction_form,
            ]),
        ]),
    ], fluid=True)
