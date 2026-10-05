import pandas as pd
import numpy as np
from statistics import NormalDist

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score

import streamlit as st

import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller

def clean_and_parse_csv(uploaded_file):
    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)
        
    # 1. Dosya Okuma
    try:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', decimal=',')
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=';', decimal=',')

    df.columns = [str(c).strip() for c in df.columns]

    # 2. Tarih Sütunu Tespiti
    date_col = None
    possible_date_names = ['date', 'time', 'time period', 'datum', 'yyyymm', 'year']
    for col in df.columns:
        if col.lower() in possible_date_names:
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]

    # 3. Sayısal Sütunları Zorunlu Olarak Clean/Float Yapma
    for col in df.columns:
        if col != date_col:
            # String ifadelerdeki virgülleri noktaya çevirip sayı yapıyoruz
            cleaned_col = df[col].astype(str).str.replace(',', '.').str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(cleaned_col, errors='coerce')

    # 4. Esnek Tarih Dönüştürücü ve Ayın 1. Gününe Normalizasyon
    date_series = df[date_col].astype(str).str.strip()
    parsed_dates = pd.to_datetime(date_series, format='%Y%b', errors='coerce')
    
    if parsed_dates.isna().sum() > len(parsed_dates) * 0.5:
        parsed_dates = pd.to_datetime(date_series, format='%Y%m', errors='coerce')
    if parsed_dates.isna().sum() > len(parsed_dates) * 0.5:
        parsed_dates = pd.to_datetime(date_series, format='mixed', errors='coerce')

    df['Date'] = parsed_dates.dt.to_period('M').dt.to_timestamp()
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    return df

def ensure_stationarity(series, var_name="Macro Variable"):
    # Zorunlu olarak sayısal tipe dönüştür
    series = pd.to_numeric(series, errors='coerce')
    clean_series = series.dropna()
    
    try:
        adf_result = adfuller(clean_series)
        p_value = adf_result[1]
    except Exception:
        p_value = 1.0
        
    if p_value > 0.05:
        transformed_series = series.pct_change() * 100
        is_transformed = True
        status_message = (
            f"⚠️ **Durağanlık Uyarısı ({var_name}):** Yüklenen veri seviye halinde durağan değildi "
            f"(ADF p-değeri: {p_value:.4f} > 0.05). Veri otomatik olarak **yüzde değişim (büyüme oranı)** "
            f"alınarak durağanlaştırılmıştır."
        )
    else:
        transformed_series = series
        is_transformed = False
        status_message = (
            f"✅ **Durağanlık Doğrulaması ({var_name}):** Yüklenen veri halihazırda durağandır "
            f"(ADF p-değeri: {p_value:.4f} <= 0.05). Dönüşüm yapılmadan doğrudan kullanılmıştır."
        )
        
    return transformed_series, status_message, is_transformed


def create_interactive_plot(df, fitted_values, y_actual, point_forecast, ci_lower, ci_upper, horizon):
    fig = go.Figure()

    # 1. Gerçekleşen Makro Değişken (Actual)
    fig.add_trace(go.Scatter(
        x=df['Date'].iloc[:-horizon],
        y=y_actual,
        mode='lines',
        name='Gerçekleşen Makro Veri (Actual)',
        line=dict(color='black', width=1.5),
        opacity=0.6
    ))

    # 2. Modelin Geçmiş Uyum Değerleri (Fitted Values)
    fig.add_trace(go.Scatter(
        x=df['Date'].iloc[:-horizon],
        y=fitted_values,
        mode='lines',
        name='Model Uyumu (Fitted Values)',
        line=dict(color='red', width=1.5)
    ))

    # Gelecek Tahmin Tarihi
    last_date = df['Date'].iloc[-1]
    future_date = last_date + pd.DateOffset(months=horizon)

    # 3. Gelecek Tahmin Noktası (Forecasted Point)
    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[point_forecast],
        mode='markers',
        name='Gelecek Tahmini (Point Forecast)',
        marker=dict(color='crimson', size=10, symbol='diamond')
    ))

    # 4. Güven Aralığı Çizgisi (Confidence Interval Bar)
    fig.add_trace(go.Scatter(
        x=[future_date, future_date],
        y=[ci_lower, ci_upper],
        mode='lines+markers',
        name='Güven Aralığı (CI)',
        line=dict(color='crimson', width=3, dash='dash'),
        marker=dict(size=6)
    ))

    # Grafik Düzeni Ayarları
    fig.update_layout(
        title="Makroekonomik Gösterge Tahmin Grafiği",
        xaxis_title="Tarih",
        yaxis_title="Değer / Büyüme Oranı (%)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig



def train_and_forecast(
    df,
    feature_cols,
    target_col='Macro_Var_Stationary',
    horizon=1,
    model_type='ridge',
    confidence_level=95,
):

    # 1. Hedef değişkeni tahmin ufku (horizon k) kadar geriye kaydırıyoruz
    # Yani bugün t anındaysak, y_target t+k anındaki makro değer olur
    y = df[target_col].shift(-horizon)
    X = df[feature_cols]
    
    # Gelecekteki hedefi henüz gerçekleşmediği için en sondaki k kadar satır NaN olur.
    # Model eğitimi için NaN olmayan geçmiş verileri alıyoruz:
    X_train = X.iloc[:-horizon]
    y_train = y.dropna()
    
    # En son t anındaki finansal getiriler (Geleceği tahmin etmek için girdi olarak kullanacağız):
    latest_X = X.iloc[[-1]]

    # 2. Seçilen modele göre Cross-Validation ile en iyi modeli eğitme
    if model_type == 'lasso':
        model = LassoCV(cv=5).fit(X_train, y_train)
    elif model_type == 'elastic':
        model = ElasticNetCV(cv=5).fit(X_train, y_train)
    else:  # Varsayılan olarak Ridge
        model = RidgeCV(cv=5).fit(X_train, y_train)

    # 3. Nokta Tahmini (Point Forecast) ve Geçmiş Veri İçi Uyum (Fitted Values)
    point_forecast = model.predict(latest_X)[0]
    fitted_values = model.predict(X_train)
    
    # Modelin başarı oranı (R2 Skoru)
    r2 = r2_score(y_train, fitted_values)

    # 4. Güven Aralığı (Confidence Interval) Hesaplama
    residuals = y_train - fitted_values
    residual_se = np.std(residuals)

    if not 0 < confidence_level < 100:
        raise ValueError("confidence_level 0 ile 100 arasında olmalıdır.")

    alpha = 1 - confidence_level / 100
    z_score = NormalDist().inv_cdf(1 - alpha / 2)
    ci_lower = point_forecast - z_score * residual_se
    ci_upper = point_forecast + z_score * residual_se
    
    # 5. Regresyon Çıktı Tablosu (Katsayılar)
    results_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    })
    
    return {
        'point_forecast': point_forecast,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'r2': r2,
        'fitted_values': fitted_values,
        'y_actual': y_train,
        'coefficients': results_df
    }

# --- STREAMLIT ARAYÜZ YAPILANDIRMASI ---
st.set_page_config(page_title="General Macro Forecasting Tool", layout="wide")

st.markdown(
    """
    # 📊 Macroeconomic Forecasting Engine
    High-Frequency Asset Returns ile Makro Gösterge Tahminleme Aracı
    """
)

# --- KULLANICI REHBERİ (ANLEITUNG / GUIDELINES) ---
with st.expander("📖 **Dosya Yükleme Rehberi ve Veri Kısıtlamaları (Anleitung)**", expanded=False):
    st.markdown("""
    Sisteminizin sorunsuz çalışması için yükleyeceğiniz CSV dosyalarının aşağıdaki şartlara uyması gerekmektedir:
    
    1. **Tarih Sütunu:** 
       - Dosyanızda en az bir adet tarih sütunu (`Date`, `YYYYMM`, `TIME PERIOD`, `Datum` vb.) bulunmalıdır.
       - Desteklenen formatlar: `YYYYMM` (ör: 199102), `YYYY-MM-DD`, `YYYY-MMM` (ör: 1991Jan), `DD.MM.YYYY`.
    2. **Ondalık ve Separatör Ayırıcılar:**
       - Virgül (`,`) ve nokta (`.`) ayrımı otomatik tespit edilir.
    3. **Makro Veri Seti:**
       - Tahmin etmek istediğiniz ana değişkeni (ör: TÜFE, Sanayi Üretim Endeksi, GDP) barındırmalıdır.
       - Veri seviye halinde durağan değilse sistem otomatik olarak **yüzde değişime (growth rate)** dönüştürür.
    4. **Varlık Getirileri (Asset Returns) Seti:**
       - Sektör portföyleri veya finansal varlıkların getirilerini (yüzde veya oran cinsinden) barındırmalıdır.
    """)

# Ekranı Sol (INPUT) ve Sağ (OUTPUT) Olarak İkiye Bölüyoruz
col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("1. Girdiler (Inputs)")
    
    macro_file = st.file_uploader("Makro Değişken Dosyası (CSV)", type=["csv"], key="macro")
    asset_file = st.file_uploader("Finansal Getiriler Dosyası (CSV)", type=["csv"], key="asset")
    
    # Dosyalar yüklendiyse dinamik sütun seçimlerini göster
    if macro_file and asset_file:
        macro_raw = clean_and_parse_csv(macro_file)
        asset_raw = clean_and_parse_csv(asset_file)
        
        # Kullanıcının hedef makro değişken sütununu seçmesi
        possible_macro_cols = [c for c in macro_raw.columns if c != 'Date']
        target_col = st.selectbox("Tahmin Edilecek Makro Değişkeni Seçin:", possible_macro_cols)
        
        horizon = st.number_input("Tahmin Ufku (Ay / k):", min_value=1, max_value=12, value=1)
        confidence_level = st.slider("Güven Seviyesi (%):", min_value=80, max_value=99, value=95)
        model_type = st.selectbox("Regresyon Yöntemi:", ["ridge", "lasso", "elastic"])
        
        run_btn = st.button("🚀 Tahmini Çalıştır", use_container_width=True)

with col_output:
    st.header("2. Çıktılar (Outputs)")
    
    if macro_file and asset_file and run_btn:
        # 1. Sol tarafta okunan verilerin kopyasını alalım (Tekrar read_csv yapmaya gerek yok)
        macro_df = macro_raw.copy()
        asset_df = asset_raw.copy()
        
        # Durağanlık Kontrolü ve Dönüşüm
        macro_df[target_col], stat_msg, is_transformed = ensure_stationarity(macro_df[target_col], var_name=target_col)
        
        # Arayüzde Durağanlık Bilgilendirmesi
        if is_transformed:
            st.warning(stat_msg)
        else:
            st.success(stat_msg)
            
        # Finansal Sektör Sütunları (Date dışındaki tüm sütunlar)
        feature_cols = [c for c in asset_df.columns if c != 'Date']
        
        # İç Birleştirme (Inner Join)
        merged_df = pd.merge(
            macro_df[['Date', target_col]], 
            asset_df[['Date'] + feature_cols], 
            on='Date', 
            how='inner'
        ).dropna().reset_index(drop=True)
        
        if len(merged_df) > 0:
            # 2. Modeli Eğit ve Tahmin Et
            res = train_and_forecast(
                merged_df, 
                feature_cols, 
                target_col=target_col, 
                horizon=horizon, 
                model_type=model_type,
                confidence_level=confidence_level
            )
            
            # Metrik Kartları
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Nokta Tahmini (Point Forecast)", f"{res['point_forecast']:.2f}%")
            m_col2.metric(f"Güven Aralığı (%{confidence_level})", f"[{res['ci_lower']:.2f}% ; {res['ci_upper']:.2f}%]")
            
            # 3. Etkileşimli Plotly Grafiği
            fig = create_interactive_plot(
                merged_df, 
                res['fitted_values'], 
                res['y_actual'], 
                res['point_forecast'], 
                res['ci_lower'], 
                res['ci_upper'], 
                horizon
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. Regresyon Çıktıı Tablosu ve R2
            st.subheader("📋 Regresyon Çıktısı ve Katsayılar")
            st.write(f"**Model Açıklayıcılık Oranı (R² Skoru):** `{res['r2']:.4f}`")
            st.dataframe(res['coefficients'], use_container_width=True)
        else:
            st.error("Tarih eşleşmesi sağlanamadı! Lütfen iki dosyadaki tarih aralıklarını kontrol edin.")



# --- DEBUG / MONITORING PANELS ---
if macro_file and asset_file:
    with st.expander("🛠️ HATA AYIKLAMA (DEBUG) PANELS - Tıkla ve İncele"):
        st.write("--- 1. MAKRO VERİ SETİ ---")
        m_df = clean_and_parse_csv(macro_file)
        st.write(f"Satır Sayısı: {len(m_df)}")
        st.write("Tarih Kolonu Tipi:", m_df['Date'].dtype if 'Date' in m_df else "YOK")
        st.write("İlk 3 Satır Tarih Örneği:", m_df['Date'].head(3).tolist() if 'Date' in m_df else "Yok")
        st.dataframe(m_df.head(3))

        st.write("--- 2. FINANSAL GETIRI SETİ ---")
        a_df = clean_and_parse_csv(asset_file)
        st.write(f"Satır Sayısı: {len(a_df)}")
        st.write("Tarih Kolonu Tipi:", a_df['Date'].dtype if 'Date' in a_df else "YOK")
        st.write("İlk 3 Satır Tarih Örneği:", a_df['Date'].head(3).tolist() if 'Date' in a_df else "Yok")
        st.dataframe(a_df.head(3))

        st.write("--- 3. EŞLEŞEN TARİHLER (INNER JOIN) ---")
        if 'Date' in m_df and 'Date' in a_df:
            intersection = pd.merge(m_df[['Date']], a_df[['Date']], on='Date', how='inner')
            st.write(f"Kesişen Ortak Satır Sayısı: {len(intersection)}")