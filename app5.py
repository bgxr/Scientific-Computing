import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statistics import NormalDist
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV



def clean_and_parse_csv(uploaded_file):
    """
    Reads a CSV file, normalizes dates to the first day of the month,
    and cleans numeric columns.
    """
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    # 1. Check delimiter (comma or semicolon)
    try:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";", decimal=",")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=";", decimal=",")

    df.columns = [str(c).strip().strip("\"'") for c in df.columns]

    # 2. Find the date column
    possible_names = ["date", "time", "time period", "datum", "yyyymm", "year"]
    date_col = None
    for c in df.columns:
        if c.lower() in possible_names:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # 3. Clean numeric columns (comma -> dot, remove %)
    numeric_cols = {}
    for c in df.columns:
        if c != date_col:
            cleaned = (
                df[c]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            num_series = pd.to_numeric(cleaned, errors="coerce")
            # Keep only numeric columns that are not completely empty
            if num_series.notna().sum() > 0:
                numeric_cols[c] = num_series

    # 4. Standardize dates (set to the 1st day of the month)
    date_str = df[date_col].astype(str).str.strip().str.strip("\"'")
    parsed = pd.to_datetime(date_str, format="%Y%m", errors="coerce")
    if parsed.isna().sum() > len(parsed) * 0.5:
        parsed = pd.to_datetime(date_str, format="%Y%b", errors="coerce")
    if parsed.isna().sum() > len(parsed) * 0.5:
        parsed = pd.to_datetime(date_str, format="mixed", errors="coerce")

    # Clean DataFrame
    clean_df = pd.DataFrame({"Date": parsed.dt.to_period("M").dt.to_timestamp()})
    for col_name, s in numeric_cols.items():
        clean_df[col_name] = s

    # Remove duplicate dates
    clean_df = (
        clean_df
        .dropna(subset=["Date"])
        .drop_duplicates(subset=["Date"], keep="first")
    )
    return clean_df.sort_values("Date").reset_index(drop=True)


def ensure_stationarity(series, var_name="Macro Variable"):
    """
    Checks whether the series is stationary using the ADF test.
    If p > 0.05, applies percent change (% growth) to make it stationary.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    try:
        p_val = float(adfuller(clean_s, result_object=False)[1])
    except Exception:
        p_val = 1.0

    if p_val > 0.05:
        transformed = series.pct_change() * 100.0
        msg = (
            f"⚠️ **{var_name}** is not stationary in levels "
            f"(ADF p={p_val:.4f} > 0.05). Percent change was taken."
        )
        return transformed, msg, True
    else:
        msg = (
            f"✅ **{var_name}** is already stationary "
            f"(ADF p={p_val:.4f} ≤ 0.05). Used as is."
        )
        return series, msg, False

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 3. SCIKIT-LEARN ELASTIC NET WITH DUAL TIME-SERIES CV (ALPHA & L1_RATIO)
# ==============================================================================
def fit_elastic_net_with_tscv(X, y, n_splits=5):
    """
    Finds optimal Alpha AND L1 Ratio simultaneously using Time-Series CV.
    Prevents zero-coefficient collapse (flat fitted values).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Grid includes smaller alphas to prevent all coefficients from collapsing to 0
    param_grid = {
        "alpha": [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
        "l1_ratio": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] # 0.0 = Ridge, 1.0 = Lasso
    }

    base_model = ElasticNet(max_iter=5000, tol=1e-4)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_
    best_alpha = grid_search.best_params_["alpha"]
    best_l1_ratio = grid_search.best_params_["l1_ratio"]

    # Reconstruct unscaled beta coefficients and intercept
    final_beta = best_model.coef_ / scaler.scale_
    intercept = best_model.intercept_ - np.dot(scaler.mean_ / scaler.scale_, best_model.coef_)

    return intercept, final_beta, best_alpha, best_l1_ratio, best_model, scaler


# ==============================================================================
# 4. FORECASTING MODEL
# ==============================================================================
def train_and_forecast(df, feature_cols, target_col, horizon=1, confidence_level=95):
    """
    Trains Elastic Net with dynamic Alpha and L1 Ratio tuning.
    """
    y_future = df[target_col].shift(-horizon)
    X = df[feature_cols].values
    y = y_future.values

    X_train = X[:-horizon]
    y_train = y[:-horizon]
    latest_X = X[[-1]]

    valid = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
    X_train = X_train[valid]
    y_train = y_train[valid]
    train_dates = df["Date"].iloc[:-horizon].iloc[valid]

    # Model Fitting & Automatic Selection of BOTH Alpha and L1 Ratio
    intercept, beta, best_alpha, best_l1_ratio, best_model, scaler = fit_elastic_net_with_tscv(
        X_train, y_train, n_splits=5
    )

    # Fitted values and point forecast
    X_train_scaled = scaler.transform(X_train)
    fitted_values = best_model.predict(X_train_scaled)

    latest_X_scaled = scaler.transform(latest_X)
    point_forecast = float(best_model.predict(latest_X_scaled)[0])

    # R^2 Score
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_res = np.sum((y_train - fitted_values) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Confidence Interval
    residuals = y_train - fitted_values
    residual_se = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals))

    prob = 1.0 - ((100.0 - confidence_level) / 200.0)
    z = NormalDist().inv_cdf(prob)
    ci_lower = point_forecast - z * residual_se
    ci_upper = point_forecast + z * residual_se

    coef_df = pd.DataFrame({
        "Asset / Sector": feature_cols,
        "Coefficient": beta
    }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)

    return {
        "point_forecast": point_forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r2": r2,
        "fitted_values": fitted_values,
        "y_actual": y_train,
        "train_dates": train_dates,
        "coefficients": coef_df,
        "best_alpha": best_alpha,
        "best_l1_ratio": best_l1_ratio
    }


st.set_page_config(page_title="Macroeconomic Forecasting (Elastic Net)", layout="wide")
st.title("Macroeconomic Forecasting with Elastic Net")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Inputs")
    macro_file = st.file_uploader("Macro Data File (CSV)", type=["csv"], key="macro")
    asset_file = st.file_uploader("Asset Returns File (CSV)", type=["csv"], key="asset")

    if macro_file and asset_file:
        macro_df = clean_and_parse_csv(macro_file)
        asset_df = clean_and_parse_csv(asset_file)

        macro_cols = [c for c in macro_df.columns if c != "Date"]
        target_var = st.selectbox("Target Macro Variable:", macro_cols)
        
        horizon = st.number_input("Forecast Horizon (Months / k):", min_value=1, max_value=12, value=1)
        conf_level = st.slider("Confidence Level (%):", 80, 99, 95)

        run_btn = st.button("Run Forecast", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Forecast Results")

    if macro_file and asset_file and run_btn:
        macro_df[target_var], stat_msg, is_transformed = ensure_stationarity(macro_df[target_var], target_var)
        if is_transformed:
            st.warning(stat_msg)
        else:
            st.success(stat_msg)

        feature_cols = [c for c in asset_df.columns if c != "Date"]
        merged = pd.merge(
            macro_df[["Date", target_var]],
            asset_df[["Date"] + feature_cols],
            on="Date",
            how="inner"
        ).dropna()

        if len(merged) > horizon + 2:
            res = train_and_forecast(
                merged,
                feature_cols,
                target_var,
                horizon=int(horizon),
                confidence_level=conf_level
            )

            # Main Financial Metrics Only (Alpha & L1 Ratio are hidden)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Point Forecast (T+{horizon})", f"{res['point_forecast']:.2f}%")
            c2.metric(f"CI (%{conf_level})", f"[{res['ci_lower']:.2f}% ; {res['ci_upper']:.2f}%]")
            c3.metric("R² Score", f"{res['r2']:.4f}")

            # Plotly chart
            last_date = merged["Date"].iloc[-1]
            future_date = last_date + pd.DateOffset(months=int(horizon))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=res["train_dates"],
                y=res["y_actual"],
                mode="lines",
                name="Realized Macro",
                line=dict(color="black", width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=res["train_dates"],
                y=res["fitted_values"],
                mode="lines",
                name="Model Fit",
                line=dict(color="red", width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=[future_date],
                y=[res["point_forecast"]],
                mode="markers",
                name="Forecast Point",
                marker=dict(color="crimson", size=10, symbol="diamond")
            ))
            fig.add_trace(go.Scatter(
                x=[future_date, future_date],
                y=[res["ci_lower"], res["ci_upper"]],
                mode="lines",
                name="Confidence Interval",
                line=dict(color="crimson", width=2, dash="dash")
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Value (%)",
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                template="plotly_white",
                height=420,
                margin=dict(l=20, r=20, t=30, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Asset Coefficients (Tracking Weights):**")
            st.dataframe(res["coefficients"], use_container_width=True)
        else:
            st.error("Not enough overlapping observations.")
    else:
        st.info("Select the two CSV files on the left and click 'Run Forecast'.")