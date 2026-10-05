import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score

import streamlit as st
import matplotlib.pyplot as plt

def prepare_data(french_file_path, macro_file_path):

    # 1. Kenneth French 10 Industry verisini oku
    french_df = pd.read_csv(french_file_path, skiprows=11, nrows=1202)
    french_df.columns = [c.strip() for c in french_df.columns]

    # Tarih kolonunun adını standartlaştıralım ve metne çevirelim
    first_col = french_df.columns[0]
    french_df.rename(columns={first_col: 'YYYYMM'}, inplace=True)
    french_df['YYYYMM'] = french_df['YYYYMM'].astype(str).str.strip()
    
    # 192607 gibi metinleri gerçek tarih objesine çevirelim (Örn: 1926-07-01)
    french_df['Date'] = pd.to_datetime(french_df['YYYYMM'], format='%Y%m')
    
    # Sektör getirilerini sayısal (float) türe zorlayalım
    industry_cols = [c for c in french_df.columns if c not in ['YYYYMM', 'Date']]
    for col in industry_cols:
        french_df[col] = pd.to_numeric(french_df[col], errors='coerce')

    # 2. ECB Makro verisini oku
    macro_df = pd.read_csv(macro_file_path)
    
    # Tarih kolonunu düzenle (Örn: "1991Jan" metnini tarihe çevir)
    macro_df['Date'] = pd.to_datetime(macro_df['TIME PERIOD'], format='%Y%b')
    
    # Makro değişkenin değer sütununu al ve sayıya çevir
    val_col = macro_df.columns[2]
    macro_df['Macro_Var'] = pd.to_numeric(macro_df[val_col], errors='coerce')
    
    # Durağanlık dönüşümü: Yüzde değişim / büyüme oranı
    macro_df['Macro_Var_Stationary'] = macro_df['Macro_Var'].pct_change() * 100

    # 3. İki veriyi ortak tarihlere göre birleştir (Inner Join)
    merged_df = pd.merge(
        macro_df[['Date', 'Macro_Var_Stationary']], 
        french_df[['Date'] + industry_cols], 
        on='Date', 
        how='inner'
    ).dropna()
    
    return merged_df, industry_cols



def train_and_forecast(df, feature_cols, target_col='Macro_Var_Stationary', horizon=1, model_type='ridge'):

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
    
    # %95 Güven seviyesi için Z-skoru (~1.96)
    z_score = 1.96
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

# Streamlit Sayfa Yapısı
st.set_page_config(page_title="Macro Forecasting Tool", layout="wide")
st.title("Forecasting Macros using Financial Data")

# Ekranı Sol (Input) ve Sağ (Output) olarak ikiye bölüyoruz
col_input, col_output = st.columns([1, 2])

with col_input:
    st.header("INPUT")
    
    # 1. Dosya Yükleyiciler
    macro_file = st.file_uploader("Macro Variable File (CSV)", type=["csv"])
    asset_file = st.file_uploader("Asset Returns File (CSV)", type=["csv"])
    
    # 2. Parametre Seçimleri
    horizon = st.number_input("Forecast Horizon (months/k)", min_value=1, max_value=12, value=1)
    confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95)
    model_type = st.selectbox("Method", ["ridge", "lasso", "elastic"])
    
    # 3. Çalıştır Butonu
    run_btn = st.button("Run Forecast")

with col_output:
    st.header("OUTPUT")
    
    # Kullanıcı her iki dosyayı da yükleyip "Run Forecast" butonuna bastığında çalışacak
    if run_btn:
        if macro_file is not None and asset_file is not None:
            # 1. Veriyi İşle ve Hizala
            df, feature_cols = prepare_data(asset_file, macro_file)
            
            # 2. Modeli Eğit ve Tahmin Et
            res = train_and_forecast(
                df, 
                feature_cols, 
                horizon=horizon, 
                model_type=model_type
            )
            
            # 3. Nokta Tahmin ve Güven Aralığı Kartları
            st.metric(
                label="Point Forecast", 
                value=f"{res['point_forecast']:.2f}%"
            )
            st.write(
                f"**Confidence Interval ({confidence_level}%):** "
                f"[{res['ci_lower']:.2f}%; {res['ci_upper']:.2f}%]"
            )
            
            # 4. Grafik Çizimi (Matplotlib)
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Gerçek geçmiş değerler ve modelin geçmiş uyumu
            ax.plot(df['Date'].iloc[:-horizon], res['y_actual'], label="Macro Variable (Actual)", color="black", alpha=0.5)
            ax.plot(df['Date'].iloc[:-horizon], res['fitted_values'], label="Fitted Values", color="red")
            
            # Gelecek tahmini ve güven aralığı
            last_date = df['Date'].iloc[-1]
            future_date = last_date + pd.DateOffset(months=horizon)
            
            ax.scatter(future_date, res['point_forecast'], color="red", s=100, zorder=5, label="Forecasted Point")
            ax.errorbar(
                future_date, 
                res['point_forecast'], 
                yerr=1.96 * np.std(res['y_actual'] - res['fitted_values']), 
                fmt='o', 
                color='red', 
                capsize=5, 
                label="Confidence Interval"
            )
            
            ax.set_title("Macro Variable Forecasting Plot")
            ax.legend()
            st.pyplot(fig)
            
            # 5. Regresyon Çıktı Tablosu ve R2 Skoru
            st.subheader("Table: Regression Output")
            st.write(f"**R² Score:** {res['r2']:.4f}")
            st.dataframe(res['coefficients'])
            
        else:
            st.warning("Lütfen hem Makro Veri hem de Varlık Getirileri dosyasını yükleyin!")