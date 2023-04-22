import pandas as pd
import plotly.express
import plotly.graph_objects
import dash
import dash_core_components
# import dash_html_components
import dash_bootstrap_components
# from dash.dependencies import Input, Output
from pathlib import Path


projectRoot = Path(__file__).parent / "../.."
data = pd.read_csv(projectRoot/"data/processed/customers.csv")

# Create the individual plots:

# Demographics Overview
gender_pie = plotly.express.pie(data, names='Gender', title='Gender Distribution')
age_histogram = plotly.express.histogram(data, x='Age', nbins=10, title='Age Distribution')

# Income and Spending Score Distribution
income_histogram = plotly.express.histogram(data, x='Annual Income ($)', nbins=10, title='Annual Income Distribution')
spending_histogram = plotly.express.histogram(
    data, x='Spending Score (1-100)', nbins=10, title='Spending Score Distribution')

# Profession Distribution
profession_bar = plotly.express.bar(data, x='Profession', title='Profession Distribution')

# Work Experience and Family Size
work_experience_histogram = plotly.express.histogram(
    data, x='Work Experience', nbins=10, title='Work Experience Distribution')
family_size_histogram = plotly.express.histogram(data, x='Family Size', nbins=5, title='Family Size Distribution')

# Customer Segmentation
customer_segmentation_scatter = plotly.express.scatter(data, x='Annual Income ($)', y='Spending Score (1-100)',
                                                       color='Profession', title='Customer Segmentation')

# Correlation Matrix
correlation_matrix = plotly.graph_objects.Figure(data=plotly.graph_objects.Heatmap(z=data.corr(
    numeric_only=True), x=data.columns, y=data.columns, colorscale='Viridis', showscale=True))

correlation_matrix.update_layout(title='Correlation Matrix')

# Create the Dash app and layout:

app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

app.layout = dash_bootstrap_components.Container([
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=gender_pie), width=6),
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=age_histogram), width=6),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=income_histogram), width=6),
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=spending_histogram), width=6),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=profession_bar), width=12),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=work_experience_histogram), width=6),
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=family_size_histogram), width=6),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=customer_segmentation_scatter), width=12),
    ]),
    dash_bootstrap_components.Row([
        dash_bootstrap_components.Col(dash_core_components.Graph(figure=correlation_matrix), width=12),
    ]),
    # Add more components for the predictive analysis section (Only for Data Science (Eng) 874 students)
])

if __name__ == '__main__':
    app.run_server(debug=True)
