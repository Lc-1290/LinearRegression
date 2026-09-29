from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ecommerce.csv"
FIGURES_DIR = BASE_DIR / "figures"
TARGET = "Yearly Amount Spent"
RANDOM_STATE = 42


def load_data():
    """Load the ecommerce dataset."""
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    """Keep numeric features and separate predictors from the target."""
    X = df.select_dtypes(include="number").drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate_model(y_test, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_test, y_pred)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "R2": r2_score(y_test, y_pred),
    }


def save_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(9, 6))
    plt.scatter(y_test, y_pred)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel("Actual Yearly Amount Spent")
    plt.ylabel("Predicted Yearly Amount Spent")
    plt.title("Actual vs Predicted Yearly Amount Spent")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def save_residual_plots(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(9, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Yearly Amount Spent")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Values")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.title("Distribution of Residuals")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residual_distribution.png", dpi=150)
    plt.close()


def save_coefficients_plot(model, feature_names):
    coefficients = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_,
        }
    ).sort_values("Coefficient", ascending=False)

    plt.figure(figsize=(9, 6))
    plt.barh(coefficients["Feature"], coefficients["Coefficient"])
    plt.title("Linear Regression Coefficients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "coefficients.png", dpi=150)
    plt.close()

    return coefficients


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    df = load_data()
    X, y = prepare_data(df)

    print("Dataset summary")
    print(f"Rows: {len(df)}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicated rows: {df.duplicated().sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    print("Test set metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    cv_scores = cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=5,
        scoring="r2",
    )

    print("\n5-fold cross-validation R²")
    print("Scores:", ", ".join(f"{score:.4f}" for score in cv_scores))
    print(f"Mean R²: {cv_scores.mean():.4f}")

    coefficients = save_coefficients_plot(model, X.columns)

    print("\nModel coefficients")
    print(coefficients.to_string(index=False))

    save_actual_vs_predicted(y_test, y_pred)
    save_residual_plots(y_test, y_pred)

    print(f"\nPlots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
