import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path
import seaborn as sb


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def correlations(df: pd.DataFrame, pathToFigures: Path):
    # Get the non-numerical columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    # Encode the non-numerical columns using get_dummies()
    df_encoded = pd.get_dummies(df, columns=non_numeric_cols)
    corr = df_encoded.corr()
    corr.to_csv(pathToFigures / "correlations.csv")


def allInOneEda(data, pathToFigures: Path, name="random"):
    # EDA using ydata_profiling
    profile = ProfileReport(data, explorative=True)
    # profile.to_notebook_iframe()
    # Save the report as an html file
    profile.to_file(pathToFigures/f"{name}.html")


def pairPlot(data: pd.DataFrame, pathToFigures: Path, variable):
    smoker_pairplot = sb.pairplot(data, hue=variable)
    smoker_pairplot.savefig(pathToFigures/f"pp_{variable}.png")


if __name__ == "__main__":
    projectRoot = Path(__file__).parent / "../.."
    pathToFigures = projectRoot/"reports/figures"

    # insurance_data = load_data(projectRoot/"data/raw/insurance.csv")
    customers_data = load_data(projectRoot/"data/raw/customers.csv")

    # allInOneEda(customers_data, projectRoot/"reports/figures", name="CustomersEda")
    # correlations(customers_data)
    pairPlot(customers_data, pathToFigures, variable="Gender")
