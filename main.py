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
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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
# 2. PUNCTUL 1: ANALIZA TRENDULUI (DETERMINIST VS STOCHASTIC)
# ==========================================

def analyze_trends(df):
    print_header("CERINȚA 1: TREND DETERMINIST VS STOCHASTIC")
    series = df['Inflation']
    t = np.arange(len(series)).reshape(-1, 1)
    
    # 1. Trend Determinist (Liniar)
    model_det = LinearRegression().fit(t, series)
    trend_det = model_det.predict(t)
    
    # 2. Trend Stochastic (Random Walk cu Drift - aproximat prin diferențiere)
    # Dacă seria e staționară după diferențiere, trendul e stochastic
    
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label='Inflație Reală', alpha=0.5)
    plt.plot(series.index, trend_det, 'r--', label='Trend Determinist (Liniar)')
    plt.title("Analiza Trendului în Rata Inflației")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, '00_trend_analysis.png'))
    plt.close()
    
    print("Interpretare: Dacă seria prezintă fluctuații persistente și nu revine rapid la o medie fixă,")
    print("vorbim de un trend stochastic. Testele ADF/KPSS vor confirma acest lucru mai jos.")

# ==========================================
# 3. ANALIZĂ UNIVARIATĂ (Inflație)
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
        
        # DIAGNOSTIC SARIMA
        print("\n--- Diagnostic Model SARIMA ---")
        residuals = model_sarima.resid()
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        print(f"Ljung-Box p-value (lag 10): {lb_test['lb_pvalue'].values[0]:.4f}")
        if lb_test['lb_pvalue'].values[0] > 0.05:
            print("Reziduuri albe (zgomot alb) - Model VALID.")
        else:
            print("Reziduuri autocorelate - Modelul ar putea fi îmbunătățit.")

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
        
        rmse_hw = np.sqrt(mean_squared_error(test, pred_hw))
        rmse_sarima = np.sqrt(mean_squared_error(test, pred_sarima))
        print(f"RMSE Holt-Winters: {rmse_hw:.4f}")
        print(f"RMSE SARIMA: {rmse_sarima:.4f}")
        print(f"Concluzie: Modelul {'SARIMA' if rmse_sarima < rmse_hw else 'HW'} este mai precis.")
        
    except Exception as e:
        print(f"Eroare forecast univariat: {e}")

# ==========================================
# 3. ANALIZĂ MULTIVARIATĂ (Inflație, Petrol, Dobândă)
# ==========================================

def run_multivariate_analysis(df):
    print_header("ANALIZĂ MULTIVARIATĂ: VECM / VAR / GRANGER")
    
    # Standardizare pentru comparabilitate în IRF
    df_std = (df - df.mean()) / df.std()
    
    # A. GRANGER CAUSALITY
    print("\n--- Teste de Cauzalitate Granger (Lag 4) ---")
    for col in ['InterestRate', 'LogOil']:
        res = grangercausalitytests(df[['Inflation', col]], maxlag=4, verbose=False)
        p_val = res[4][0]['ssr_ftest'][1]
        print(f"Cauzalitate {col} -> Inflation: p-value = {p_val:.4f}")

    # B. Cointegrare (Johansen)
    try:
        coint = coint_johansen(df, det_order=0, k_ar_diff=1)
        if coint.lr1[0] > coint.cvt[0, 1]:
            print("\nSistemul este COINTEGRAT (Trace Test > Critical Value).")
            print("Există o relație stabilă pe termen lung între Inflație, Dobândă și Petrol.")
            model_vecm = VECM(df_std, k_ar_diff=2, coint_rank=1).fit()
            print("\nRezultate VECM (Coeficienți de ajustare):")
            print(model_vecm.alpha) # Viteza de ajustare
        else:
            print("\nSistemul NU este cointegrat. Estimăm modelul VAR...")
            model_vecm = None
            
        # Pentru IRF și FEVD folosim VAR pe date diferențiate (staționare)
        model = VAR(df_std.diff().dropna()).fit(maxlags=4)
        
        # IRF (Impulse Response Functions)
        print("\nGenerare Impulse Response...")
        irf = model.irf(24)
        irf.plot(orth=True, impulse='LogOil', response='Inflation')
        plt.title("Răspunsul Inflației la un șoc în Prețul Petrolului")
        plt.savefig(os.path.join(OUTPUT_DIR, '02_irf_oil_inflation.png'))
        plt.close()
        
        # FEVD (Variance Decomposition)
        fevd = model.fevd(12)
        fevd.plot()
        plt.savefig(os.path.join(OUTPUT_DIR, '04_fevd.png'))
        plt.close()
        
        print("Analiză multivariată finalizată.")
        
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
        analyze_trends(data)
        run_univariate_inflation(data)
        run_multivariate_analysis(data)
        
    print_header("PROIECT FINALIZAT")
    print(f"Rezultatele se află în: {OUTPUT_DIR}")