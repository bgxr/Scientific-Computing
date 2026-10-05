import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from statistics import NormalDist
from statsmodels.tsa.stattools import adfuller


def read_csv_file(uploaded_file):
    """CSV dosyasini oku, tarih ve sayisal sutunlari standartlastir."""
    uploaded_file.seek(0)
    try:
        data = pd.read_csv(uploaded_file)
        if len(data.columns) == 1:
            uploaded_file.seek(0)
            data = pd.read_csv(uploaded_file, sep=";", decimal=",")
    except (pd.errors.ParserError, UnicodeDecodeError):
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, sep=";", decimal=",")

    data.columns = [str(column).strip() for column in data.columns]

    known_date_names = {"date", "time", "time period", "datum", "yyyymm", "year"}
    date_column = next(
        (column for column in data.columns if column.lower() in known_date_names),
        data.columns[0],
    )

    for column in data.columns:
        if column != date_column:
            values = (
                data[column]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            data[column] = pd.to_numeric(values, errors="coerce")

    date_values = data[date_column].astype(str).str.strip()
    parsed_dates = pd.to_datetime(date_values, format="%Y%b", errors="coerce")
    if parsed_dates.isna().mean() > 0.5:
        parsed_dates = pd.to_datetime(date_values, format="%Y%m", errors="coerce")
    if parsed_dates.isna().mean() > 0.5:
        parsed_dates = pd.to_datetime(date_values, format="mixed", errors="coerce")

    data["Date"] = parsed_dates.dt.to_period("M").dt.to_timestamp()
    return data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def make_stationary(series):
    """ADF testine gore seriyi oldugu gibi veya yuzde degisim olarak dondur."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    clean_series = numeric_series.dropna()

    try:
        p_value = adfuller(clean_series, result_object=False)[1]
    except (ValueError, np.linalg.LinAlgError):
        p_value = 1.0

    if p_value > 0.05:
        result = numeric_series.pct_change() * 100
        message = (
            f"ADF p-degeri {p_value:.4f}. Seri durağan degil; "
            "yuzde degisim uygulandi."
        )
        return result, message

    message = f"ADF p-degeri {p_value:.4f}. Seri durağan kabul edildi."
    return numeric_series, message


def soft_threshold(value, penalty):
    """L1 cezasinin coordinate descent icin kullandigi esikleme islemi."""
    return np.sign(value) * max(abs(value) - penalty, 0.0)


def fit_elastic_net(
    x,
    y,
    l1_ratio=0.5,
    penalty=0.1,
    max_iterations=500,
    tolerance=1e-5,
):
    """
    Elastic Net'i coordinate descent ile egitir.

    l1_ratio=1 sadece Lasso, l1_ratio=0 sadece Ridge olur.
    Burada katsayilar standardize edilerek egitilir ve orijinal olcege cevrilir.
    """
    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio 0 ile 1 arasinda olmalidir.")
    if penalty < 0:
        raise ValueError("penalty negatif olamaz.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = x.shape

    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale == 0] = 1.0
    x_scaled = (x - x_mean) / x_scale

    y_mean = y.mean()
    y_centered = y - y_mean
    coefficients_scaled = np.zeros(n_features)

    l1_penalty = penalty * l1_ratio
    l2_penalty = penalty * (1 - l1_ratio)

    for _ in range(max_iterations):
        previous = coefficients_scaled.copy()
        prediction = x_scaled @ coefficients_scaled

        for feature_index in range(n_features):
            residual_without_feature = (
                y_centered
                - prediction
                + x_scaled[:, feature_index] * coefficients_scaled[feature_index]
            )
            correlation = (
                x_scaled[:, feature_index] @ residual_without_feature / n_samples
            )
            updated = soft_threshold(correlation, l1_penalty)
            updated /= 1 + l2_penalty

            prediction += x_scaled[:, feature_index] * (
                updated - coefficients_scaled[feature_index]
            )
            coefficients_scaled[feature_index] = updated

        if np.max(np.abs(coefficients_scaled - previous)) < tolerance:
            break

    coefficients = coefficients_scaled / x_scale
    intercept = y_mean - x_mean @ coefficients
    return intercept, coefficients


def mean_squared_error(actual, predicted):
    return np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)


def choose_penalty(x, y, l1_ratio=0.5, folds=3):
    """Basit bir k-fold validation ile en iyi penalty degerini sec."""
    n_samples = len(y)
    folds = min(folds, n_samples)
    if folds < 2:
        raise ValueError("Lambda secimi icin en az iki egitim gozlemi gerekir.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # fit_elastic_net katsayilari standardize edilmis x ile buldugu icin
    # penalty araligini da ayni olcekte hesapliyoruz.
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale == 0] = 1.0
    x_standardized = (x - x_mean) / x_scale
    maximum_penalty = np.max(
        np.abs(x_standardized.T @ (y - y.mean()))
    ) / (n_samples * l1_ratio)
    if not np.isfinite(maximum_penalty) or maximum_penalty <= 0:
        maximum_penalty = 1.0

    penalties = np.geomspace(maximum_penalty, maximum_penalty * 0.01, 15)
    validation_errors = []
    validation_indices = np.array_split(np.arange(n_samples), folds)

    for penalty in penalties:
        fold_errors = []
        for validation_index in validation_indices:
            training_index = np.setdiff1d(np.arange(n_samples), validation_index)
            intercept, coefficients = fit_elastic_net(
                x[training_index],
                y[training_index],
                l1_ratio=l1_ratio,
                penalty=penalty,
            )
            prediction = intercept + x[validation_index] @ coefficients
            fold_errors.append(mean_squared_error(y[validation_index], prediction))
        validation_errors.append(np.mean(fold_errors))

    return penalties[int(np.argmin(validation_errors))]


def train_and_forecast(data, feature_columns, target_column, horizon, confidence_level):
    """Gecmis verilerle modeli egitip son donem icin tahmin uret."""
    if len(data) <= horizon:
        raise ValueError("Tahmin ufku veri uzunlugundan kisa olmalidir.")

    future_target = data[target_column].shift(-horizon)
    x = data[feature_columns].to_numpy(dtype=float)
    y = future_target.dropna().to_numpy(dtype=float)
    x_train = x[:-horizon]
    latest_x = x[[-1]]

    selected_penalty = choose_penalty(x_train, y)
    intercept, coefficients = fit_elastic_net(
        x_train,
        y,
        penalty=selected_penalty,
    )

    fitted_values = intercept + x_train @ coefficients
    point_forecast = float(intercept + latest_x[0] @ coefficients)
    residuals = y - fitted_values

    alpha = 1 - confidence_level / 100
    z_score = NormalDist().inv_cdf(1 - alpha / 2)
    standard_error = np.std(residuals)

    return {
        "point_forecast": point_forecast,
        "ci_lower": point_forecast - z_score * standard_error,
        "ci_upper": point_forecast + z_score * standard_error,
        "fitted_values": fitted_values,
        "actual_values": y,
        "coefficients": pd.DataFrame(
            {"Feature": feature_columns, "Coefficient": coefficients}
        ),
        "r2": 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2),
        "penalty": selected_penalty,
    }


def make_plot(data, result, horizon):
    """Gerceklesen degerleri, model uyumunu ve tahmin araligini ciz."""
    future_date = data["Date"].iloc[-1] + pd.DateOffset(months=horizon)
    history_dates = data["Date"].iloc[:-horizon]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history_dates,
            y=result["actual_values"],
            mode="lines",
            name="Gerçekleşen değer",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history_dates,
            y=result["fitted_values"],
            mode="lines",
            name="Model uyumu",
            line={"color": "red"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[future_date],
            y=[result["point_forecast"]],
            mode="markers",
            name="Tahmin",
            marker={"color": "crimson", "size": 10, "symbol": "diamond"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[future_date, future_date],
            y=[result["ci_lower"], result["ci_upper"]],
            mode="lines+markers",
            name="Güven aralığı",
            line={"color": "crimson", "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Makroekonomik gösterge tahmini",
        xaxis_title="Tarih",
        yaxis_title="Değer / büyüme oranı (%)",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5,
        },
        template="plotly_white",
        margin={"l": 40, "r": 40, "t": 60, "b": 80},
    )
    return figure


st.set_page_config(page_title="Elastic Net Forecasting", layout="wide")
st.markdown(
    """
    # 📊 Elastic Net ile Makroekonomik Tahmin
    Finansal getirilerden gelecek dönem makro değerini tahmin eder.
    """
)

with st.expander("Kullanım rehberi"):
    st.write(
        "İki CSV yükleyin. Her dosyada tarih sütunu ve sayısal sütunlar "
        "bulunmalıdır. Makro dosyasından bir hedef, finansal dosyadan da "
        "açıklayıcı değişkenler seçilir."
    )

macro_file = st.file_uploader("Makro veri dosyası", type=["csv"])
asset_file = st.file_uploader("Finansal getiri dosyası", type=["csv"])

if macro_file is not None and asset_file is not None:
    macro_data = read_csv_file(macro_file)
    asset_data = read_csv_file(asset_file)

    macro_columns = [column for column in macro_data.columns if column != "Date"]
    feature_columns = [column for column in asset_data.columns if column != "Date"]

    target_column = st.selectbox("Tahmin edilecek makro değişken", macro_columns)
    horizon = st.number_input("Tahmin ufku (ay)", min_value=1, max_value=12, value=1)
    confidence_level = st.slider("Güven seviyesi (%)", 80, 99, 95)

    if st.button("Tahmini çalıştır", use_container_width=True):
        with st.spinner("Veriler birleştiriliyor ve Elastic Net eğitiliyor..."):
            macro_data[target_column], message = make_stationary(
                macro_data[target_column]
            )

            merged_data = pd.merge(
                macro_data[["Date", target_column]],
                asset_data[["Date"] + feature_columns],
                on="Date",
                how="inner",
            ).dropna()

            try:
                result = train_and_forecast(
                    merged_data,
                    feature_columns,
                    target_column,
                    int(horizon),
                    confidence_level,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.info(message)
                st.metric("Nokta tahmini", f"{result['point_forecast']:.2f}%")
                st.metric(
                    f"Güven aralığı (%{confidence_level})",
                    f"[{result['ci_lower']:.2f}% ; {result['ci_upper']:.2f}%]",
                )
                st.plotly_chart(
                    make_plot(merged_data, result, int(horizon)),
                    use_container_width=True,
                )
                st.write(f"R² skoru: `{result['r2']:.4f}`")
                st.write(f"Seçilen penalty: `{result['penalty']:.6g}`")
                st.dataframe(result["coefficients"], use_container_width=True)
