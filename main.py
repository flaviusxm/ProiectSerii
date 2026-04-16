import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings

# ==========================================
# CONFIGURARE ȘI ESTETICĂ
# ==========================================
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
sns.set_palette("tab10")
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = 'rezultate_analiza'

def print_header(text):
    print(f"\n{'='*60}\n{text}\n{'='*60}")

# ==========================================
# 1. ÎNCĂRCARE ȘI PREGĂTIRE DATE (FRED)
# ==========================================

def load_macro_data():
    print("Încărcare date FRED (CPI, Interest Rate, Oil Price)...")
    
    try:
        df_cpi = pd.read_csv('CPIAUCSL.csv')
        df_fed = pd.read_csv('FEDFUNDS.csv')
        df_oil = pd.read_csv('POILWTIUSDM.csv')
        
        # Redenumire coloane pentru consistență
        df_cpi.rename(columns={'observation_date': 'date', 'CPIAUCSL': 'CPI'}, inplace=True)
        df_fed.rename(columns={'observation_date': 'date', 'FEDFUNDS': 'InterestRate'}, inplace=True)
        df_oil.rename(columns={'observation_date': 'date', 'POILWTIUSDM': 'OilPrice'}, inplace=True)
        
        # Merge
        df = pd.merge(df_cpi, df_fed, on='date', how='inner')
        df = pd.merge(df, df_oil, on='date', how='inner')
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # TRANSFORMĂRI ECONOMETRICE
        # 1. Inflație (Year-on-Year % Change)
        df['Inflation'] = df['CPI'].pct_change(12) * 100
        
        # 2. Log Petrol (pentru stabilizarea varianței)
        df['LogOil'] = np.log(df['OilPrice'])
        
        # Curățare NaNs de la pct_change
        df.dropna(inplace=True)
        
        print(f"Date încărcate cu succes ({len(df)} observații lunare).")
        return df[['Inflation', 'InterestRate', 'LogOil']]
        
    except Exception as e:
        print(f"Eroare la încărcarea datelor: {e}")
        return pd.DataFrame()

# ==========================================
# 2. ANALIZĂ UNIVARIATĂ (Inflație)
# ==========================================

def check_stationarity(series, name):
    print(f"\n--- Test Staționaritate: {name} ---")
    adf_res = adfuller(series)
    print(f"ADF p-value: {adf_res[1]:.4f}")
    kpss_res = kpss(series, regression='c')
    print(f"KPSS p-value: {kpss_res[1]:.4f}")
    return adf_res[1] < 0.05

def run_univariate_inflation(df):
    print_header("ANALIZĂ UNIVARIATĂ: INFLAȚIE")
    series = df['Inflation']
    
    check_stationarity(series, 'Inflation')
    
    # Split
    horizon = 12 # 1 an
    train, test = series[:-horizon], series[-horizon:]
    
    print(f"Prognoză pentru {horizon} luni...")
    
    try:
        # HW
        model_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
        pred_hw = model_hw.forecast(horizon)
        
        # SARIMA (Auto)
        model_sarima = auto_arima(train, seasonal=True, m=12, suppress_warnings=True)
        pred_sarima, conf_int = model_sarima.predict(n_periods=horizon, return_conf_int=True)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(train.index[-60:], train[-60:], label='Istoric Recent')
        plt.plot(test.index, test, 'k--', label='Actual')
        plt.plot(test.index, pred_hw, 'r', label='Holt-Winters')
        plt.plot(test.index, pred_sarima, 'g', label='SARIMA')
        plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='g', alpha=0.1)
        plt.title("Prognoza Ratei Inflației (YoY %)")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, '01_inflation_forecast.png'))
        plt.close()
        
        print(f"RMSE SARIMA: {np.sqrt(mean_squared_error(test, pred_sarima)):.4f}")
    except Exception as e:
        print(f"Eroare forecast univariat: {e}")

# ==========================================
# 3. ANALIZĂ MULTIVARIATĂ (Inflație, Petrol, Dobândă)
# ==========================================

def run_multivariate_analysis(df):
    print_header("ANALIZĂ MULTIVARIATĂ: VECM / VAR")
    
    # Standardizare pentru comparabilitate în IRF
    df_std = (df - df.mean()) / df.std()
    
    # A. Cointegrare (Johansen)
    try:
        coint = coint_johansen(df, det_order=0, k_ar_diff=1)
        # Verificăm dacă există cel puțin o relație de cointegrare (Trace Statistic > Critical Value)
        if coint.lr1[0] > coint.cvt[0, 1]:
            print("Sistemul este COINTEGRAT. Estimăm modelul VECM...")
            model_vecm = VECM(df_std, k_ar_diff=2, coint_rank=1).fit()
        else:
            print("Sistemul NU este cointegrat. Estimăm modelul VAR...")
            model_vecm = None
            
        # Pentru IRF și FEVD folosim întotdeauna VAR pe date diferențiate
        model = VAR(df_std.diff().dropna()).fit(maxlags=2)
        
        # Dacă sistemul e cointegrat, afișăm rezultatele VECM
        if model_vecm is not None:
            print("Rezultate VECM:")
            print(model_vecm.summary())
            
        # IRF (Impulse Response Functions)
        print("Generare Impulse Response (Efectul prețului petrolului asupra inflației)...")
        irf = model.irf(24)
        irf.plot(orth=True, impulse='LogOil', response='Inflation')
        plt.savefig(os.path.join(OUTPUT_DIR, '02_irf_oil_inflation.png'))
        plt.close()
        
        # Toate IRF-urile
        irf.plot(orth=True)
        plt.savefig(os.path.join(OUTPUT_DIR, '03_irf_all.png'))
        plt.close()
        
        # FEVD (Variance Decomposition)
        fevd = model.fevd(12)
        fevd.plot()
        plt.savefig(os.path.join(OUTPUT_DIR, '04_fevd.png'))
        plt.close()
        
        print("Analiză multivariată finalizată cu succes.")
        
    except Exception as e:
        print(f"Eroare în analiza multivariată: {e}")

# ==========================================
# EXECUȚIE
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    data = load_macro_data()
    
    if not data.empty:
        run_univariate_inflation(data)
        run_multivariate_analysis(data)
        
    print_header("PROIECT FINALIZAT")
    print(f"Rezultatele se află în: {OUTPUT_DIR}")