import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statistics import NormalDist


# ==============================================================================
# 1. BASİT CSV OKUMA VE TEMİZLEME (DATA PARSER)
# ==============================================================================
def clean_and_parse_csv(uploaded_file):
    """
    CSV dosyasını okur, tarihleri ayın ilk gününe eşitler ve
    sayısal sütunları temizler.
    """
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    # 1. Ayırıcı (virgül veya noktalı virgül) kontrolü
    try:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";", decimal=",")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=";", decimal=",")

    df.columns = [str(c).strip().strip("\"'") for c in df.columns]

    # 2. Tarih sütununu bulma
    possible_names = ["date", "time", "time period", "datum", "yyyymm", "year"]
    date_col = None
    for c in df.columns:
        if c.lower() in possible_names:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # 3. Sayısal sütunları düzeltme (virgül -> nokta, % silme)
    numeric_cols = {}
    for c in df.columns:
        if c != date_col:
            cleaned = df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False).str.strip()
            num_series = pd.to_numeric(cleaned, errors="coerce")
            # Tamamen boş olmayan sayısal sütunları alalım
            if num_series.notna().sum() > 0:
                numeric_cols[c] = num_series

    # 4. Tarihi standartlaştırma (Ayın 1. gününe çekme)
    date_str = df[date_col].astype(str).str.strip().str.strip("\"'")
    parsed = pd.to_datetime(date_str, format="%Y%m", errors="coerce")
    if parsed.isna().sum() > len(parsed) * 0.5:
        parsed = pd.to_datetime(date_str, format="%Y%b", errors="coerce")
    if parsed.isna().sum() > len(parsed) * 0.5:
        parsed = pd.to_datetime(date_str, format="mixed", errors="coerce")

    # Temiz veri çerçevesi
    clean_df = pd.DataFrame({"Date": parsed.dt.to_period("M").dt.to_timestamp()})
    for col_name, s in numeric_cols.items():
        clean_df[col_name] = s

    # Yinelenen tarihleri temizle (örnek: Kenneth French dosyasındaki alt tablolar)
    clean_df = clean_df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"], keep="first")
    return clean_df.sort_values("Date").reset_index(drop=True)


# ==============================================================================
# 2. DURAĞANLIK KONTROLÜ (ADF TESTİ)
# ==============================================================================
def ensure_stationarity(series, var_name="Makro Değişken"):
    """
    ADF testi ile serinin durağan olup olmadığını kontrol eder.
    p > 0.05 ise yüzde değişim (% growth) uygulayarak durağanlaştırır.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    try:
        p_val = float(adfuller(clean_s, result_object=False)[1])
    except Exception:
        p_val = 1.0

    if p_val > 0.05:
        transformed = series.pct_change() * 100.0
        msg = f"⚠️ **{var_name}** seviye halinde durağan değil (ADF p={p_val:.4f} > 0.05). Yüzde değişim alındı."
        return transformed, msg, True
    else:
        msg = f"✅ **{var_name}** halihazırda durağan (ADF p={p_val:.4f} ≤ 0.05). Olduğu gibi kullanıldı."
        return series, msg, False


# ==============================================================================
# 3. KÜTÜPHANESİZ ELASTIC NET (PURE NUMPY - COORDINATE DESCENT)
# ==============================================================================
def soft_threshold(rho, l1_penalty):
    """Lasso (L1) için Soft-Thresholding operatörü."""
    if rho > l1_penalty:
        return rho - l1_penalty
    elif rho < -l1_penalty:
        return rho + l1_penalty
    else:
        return 0.0


def fit_elastic_net(X, y, l1_ratio=0.5, alpha=0.1, max_iter=300, tol=1e-4):
    """
    Scikit-learn kullanmadan, saf NumPy ile Coordinate Descent Elastic Net.
    Formül: (1 / 2n) * ||y - Xb||^2 + alpha * l1_ratio * |b| + 0.5 * alpha * (1 - l1_ratio) * ||b||^2
    """
    n_samples, n_features = X.shape

    # Veriyi ölçeklendirme (X ortalama 0, varyans 1)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_centered = y - y_mean

    beta = np.zeros(n_features)
    l1_pen = alpha * l1_ratio
    l2_pen = alpha * (1.0 - l1_ratio)

    # Coordinate Descent Döngüsü
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(n_features):
            # j. değişken dışındaki kalan artık (residual)
            residual = y_centered - (X_scaled @ beta) + X_scaled[:, j] * beta[j]
            rho = np.dot(X_scaled[:, j], residual) / n_samples
            
            # Soft-thresholding ve L2 güncellemesi
            beta[j] = soft_threshold(rho, l1_pen) / (1.0 + l2_pen)

        # Değişim toleranstan küçükse dur
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    # Katsayıları orijinal ölçeğe çevirme
    final_beta = beta / X_std
    intercept = y_mean - np.dot(X_mean, final_beta)
    return intercept, final_beta


# ==============================================================================
# 4. TAHMİN MODELİ (TIME-SHIFT & FORECAST)
# ==============================================================================
def train_and_forecast(df, feature_cols, target_col, horizon=1, confidence_level=95, alpha=0.05, l1_ratio=0.5):
    """
    Hedef değişkeni horizon (k) kadar kaydırır (shift(-k)) ve
    Elastic Net ile eğitip geleceği tahmin eder.
    """
    # y_{t+k} eşleşmesi
    y_future = df[target_col].shift(-horizon)
    X = df[feature_cols].values
    y = y_future.values

    # Geçmiş eğitim verisi (son k satır NaN olacağı için hariç tutulur)
    X_train = X[:-horizon]
    y_train = y[:-horizon]
    
    # En son t anındaki finansal getiriler (T+k tahmininde kullanılacak)
    latest_X = X[[-1]]

    # Eksik değerleri temizleme
    valid = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
    X_train = X_train[valid]
    y_train = y_train[valid]
    # Geçmiş eğitim tarihlerini alıp geçerli satırları filtreliyoruz
    train_dates = df["Date"].iloc[:-horizon].iloc[valid]

    # Modeli Eğitme
    intercept, beta = fit_elastic_net(X_train, y_train, l1_ratio=l1_ratio, alpha=alpha)

    # Uyum (fitted) ve Tahmin (point forecast)
    fitted_values = intercept + X_train @ beta
    point_forecast = float(intercept + np.dot(latest_X[0], beta))

    # R^2 Skoru
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    ss_res = np.sum((y_train - fitted_values) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Güven Aralığı Hesaplama (Z-skoru * Hata Standart Sapması)
    residuals = y_train - fitted_values
    residual_se = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals))

    prob = 1.0 - ((100.0 - confidence_level) / 200.0)
    z = NormalDist().inv_cdf(prob)
    ci_lower = point_forecast - z * residual_se
    ci_upper = point_forecast + z * residual_se

    coef_df = pd.DataFrame({
        "Varlık / Sektör": feature_cols,
        "Katsayı": beta
    }).sort_values("Katsayı", key=abs, ascending=False).reset_index(drop=True)

    return {
        "point_forecast": point_forecast,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r2": r2,
        "fitted_values": fitted_values,
        "y_actual": y_train,
        "train_dates": train_dates,
        "coefficients": coef_df
    }


# ==============================================================================
# 5. STREAMLIT ARAYÜZÜ (BASİT & TEMİZ)
# ==============================================================================
st.set_page_config(page_title="Makroekonomik Tahmin (Elastic Net)", layout="wide")
st.title("📊 Elastic Net ile Makroekonomik Tahminleme")
st.caption("Lamont (2001) Takip Portföyleri yaklaşımı - Saf NumPy Coordinate Descent ile.")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1. Girdiler")
    macro_file = st.file_uploader("Makro Veri Dosyası (CSV)", type=["csv"], key="macro")
    asset_file = st.file_uploader("Varlık Getirileri Dosyası (CSV)", type=["csv"], key="asset")

    if macro_file and asset_file:
        macro_df = clean_and_parse_csv(macro_file)
        asset_df = clean_and_parse_csv(asset_file)

        macro_cols = [c for c in macro_df.columns if c != "Date"]
        target_var = st.selectbox("Hedef Makro Değişken:", macro_cols)
        
        horizon = st.number_input("Tahmin Ufku (Ay / k):", min_value=1, max_value=12, value=1)
        conf_level = st.slider("Güven Seviyesi (%):", 80, 99, 95)
        
        # Basit parametreler
        alpha_val = st.number_input("Ceza Katsayısı (Alpha / λ):", min_value=0.001, max_value=5.0, value=0.05, step=0.01)
        l1_ratio = st.slider("L1 Oranı (0: Ridge, 1: Lasso):", 0.0, 1.0, 0.5, step=0.1)

        run_btn = st.button("🚀 Tahmini Hesapla", type="primary", use_container_width=True)

with col_right:
    st.subheader("2. Tahmin Sonuçları")

    if macro_file and asset_file and run_btn:
        # Durağanlık dönüşümü
        macro_df[target_var], stat_msg, is_transformed = ensure_stationarity(macro_df[target_var], target_var)
        if is_transformed:
            st.warning(stat_msg)
        else:
            st.success(stat_msg)

        # Verileri birleştirme
        feature_cols = [c for c in asset_df.columns if c != "Date"]
        merged = pd.merge(macro_df[["Date", target_var]], asset_df[["Date"] + feature_cols], on="Date", how="inner").dropna()

        if len(merged) > horizon + 2:
            res = train_and_forecast(
                merged, feature_cols, target_var,
                horizon=int(horizon),
                confidence_level=conf_level,
                alpha=alpha_val,
                l1_ratio=l1_ratio
            )

            # Metrikler
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Nokta Tahmin (T+{horizon})", f"{res['point_forecast']:.2f}%")
            c2.metric(f"Güven Aralığı (%{conf_level})", f"[{res['ci_lower']:.2f}% ; {res['ci_upper']:.2f}%]")
            c3.metric("R² Skoru", f"{res['r2']:.4f}")

            # Plotly Grafiği
            last_date = merged["Date"].iloc[-1]
            future_date = last_date + pd.DateOffset(months=int(horizon))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["train_dates"], y=res["y_actual"], mode="lines", name="Gerçekleşen Makro", line=dict(color="black", width=1.5)))
            fig.add_trace(go.Scatter(x=res["train_dates"], y=res["fitted_values"], mode="lines", name="Model Uyumu", line=dict(color="red", width=1.5)))
            fig.add_trace(go.Scatter(x=[future_date], y=[res["point_forecast"]], mode="markers", name="Tahmin Noktası", marker=dict(color="crimson", size=10, symbol="diamond")))
            fig.add_trace(go.Scatter(x=[future_date, future_date], y=[res["ci_lower"], res["ci_upper"]], mode="lines", name="Güven Aralığı", line=dict(color="crimson", width=2, dash="dash")))

            fig.update_layout(
                xaxis_title="Tarih", yaxis_title="Değer (%)",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                template="plotly_white", height=420, margin=dict(l=20, r=20, t=30, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Katsayı Tablosu
            st.write("**Varlık Katsayıları (Tracking Weights):**")
            st.dataframe(res["coefficients"], use_container_width=True)
        else:
            st.error("Yeterli ortak gözlem bulunamadı.")
    else:
        st.info("Sol taraftan iki CSV dosyasını seçip 'Tahmini Hesapla' butonuna basın.")


