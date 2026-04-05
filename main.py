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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# ==========================================
# CONFIGURARE ȘI ESTETICĂ
# ==========================================
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
sns.set_palette("tab10")
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# LISTA DE ȚĂRI PENTRU ANALIZĂ (Poți adăuga/șterge coduri Eurostat aici)
TARGET_COUNTRIES = ['RO', 'DE', 'HU']
OUTPUT_DIR = 'rezultate_analiza'

def print_header(text):
    print(f"\n{'='*60}\n{text}\n{'='*60}")

# ==========================================
# 1. CURĂȚARE ȘI UNIFICARE DATE
# ==========================================

def get_isced_data(df, isced, name, geo):
    subset = df[(df['geo'] == geo) &
                (df['isced11'] == isced) & 
                (df['age'] == 'Y25-64') & 
                (df['sex'] == 'T') & 
                (df['unit'] == 'PC')].copy()
    
    if subset.empty:
        return pd.DataFrame()
        
    subset['TIME_PERIOD'] = pd.to_datetime(subset['TIME_PERIOD'], format='%Y')
    subset = subset.set_index('TIME_PERIOD').sort_index()
    subset = subset[['OBS_VALUE']].rename(columns={'OBS_VALUE': name})
    return subset[~subset.index.duplicated(keep='first')]

def prepare_dataset(geo, full_df):
    print(f"Pregătire date pentru: {geo}...")
    
    # 1. Tertiary: University Graduates (ED5-8)
    # 2. Secondary: Upper Secondary (ED3_4)
    # 3. Basic: Lower Secondary and below (ED0-2)
    
    df_tertiary = get_isced_data(full_df, 'ED5-8', 'Tertiary', geo)
    df_secondary = get_isced_data(full_df, 'ED3_4', 'Secondary', geo)
    df_basic = get_isced_data(full_df, 'ED0-2', 'Basic', geo)

    if df_tertiary.empty or df_secondary.empty:
        print(f"Eroare: Date insuficiente pentru {geo}.")
        return pd.DataFrame()

    # Unim datele
    df_final = df_tertiary.join([df_secondary, df_basic], how='inner')
    
    # Tratare valori lipsă
    if df_final.isnull().values.any():
        df_final = df_final.interpolate(method='linear').ffill().bfill()

    return df_final

# ==========================================
# 2. ANALIZĂ UNIVARIATĂ
# ==========================================

def run_univariate(df, geo, target_col='Tertiary'):
    print(f"--- Analiză Univariată: {geo} ({target_col}) ---")
    series = df[target_col]
    
    # A. Trend Visual
    plt.figure(figsize=(10, 5))
    plt.plot(series, marker='s', label=f'Observed ({geo})')
    plt.title(f"Evoluția absolvenților ({target_col}) în {geo}")
    plt.ylabel("% din populație")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{geo}_01_trend.png'))
    plt.close()
    
    # B. Split
    horizon = 4
    if len(series) < 10: return # Insuficient
    train, test = series[:-horizon], series[-horizon:]

    # C. Modele
    try:
        model_hw = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
        pred_hw = model_hw.forecast(horizon)
        
        model_sarima = auto_arima(train, seasonal=False, error_action='ignore', suppress_warnings=True)
        pred_sarima, conf_int = model_sarima.predict(n_periods=horizon, return_conf_int=True)

        # Calcul erori
        rmse_hw = np.sqrt(mean_squared_error(test, pred_hw))
        rmse_sarima = np.sqrt(mean_squared_error(test, pred_sarima))

        # Diagnostic Reziduuri (Ljung-Box)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(model_sarima.resid(), lags=[10], return_df=True)
        lb_p = lb_test['lb_pvalue'].values[0]
        print(f"RMSE {geo}: HW={rmse_hw:.2f}, SARIMA={rmse_sarima:.2f}, Ljung-Box p-val={lb_p:.4f}")

        # Plot Comparativ cu Intervale de Încredere
        plt.figure(figsize=(11, 5))
        plt.plot(train, label='Istoric (Train)')
        plt.plot(test, label='Observed (Test)', color='black', linewidth=1.5)
        plt.plot(test.index, pred_hw, 'r--', label='HW Forecast')
        plt.plot(test.index, pred_sarima, 'g-', label='SARIMA Forecast')
        
        # Interval de Încredere 95%
        plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='g', alpha=0.1, label='95% Conf. Interval')
        
        plt.title(f"Analiză Univariată {geo} - {target_col} (cu Interval de Încredere)")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{geo}_02_forecast.png'))
        plt.close()
    except Exception as e:
        print(f"Eroare analiză univariată {geo}: {e}")

# ==========================================
# 3. ANALIZĂ MULTIVARIATĂ
# ==========================================

def run_multivariate(df, geo):
    print(f"--- Analiză Multivariată: {geo} ---")
    data_mv = df[['Tertiary', 'Secondary']].copy()
    if len(data_mv) < 15:
        print(f"Date insuficiente (N={len(data_mv)}) pentru VAR în {geo}.")
        return

    # A. Test Cointegrare
    try:
        coint_res = coint_johansen(data_mv, det_order=0, k_ar_diff=1)
        trace = coint_res.lr1[0]
        crit = coint_res.cvt[0, 1]
        
        if trace > crit:
            model = VECM(data_mv, k_ar_diff=1, coint_rank=1).fit()
        else:
            model = VAR(data_mv.diff().dropna()).fit(maxlags=1)

        # IRF
        if hasattr(model, 'irf'):
            irf = model.irf(periods=8)
            irf.plot(orth=True)
            plt.savefig(os.path.join(OUTPUT_DIR, f'{geo}_03_irf.png'))
            plt.close()
            
        print(f"Analiză multivariată completă pentru {geo}.")
    except Exception as e:
        print(f"Eroare VAR/VECM {geo}: {e}")

# ==========================================
# EXECUȚIE
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print_header("MENIU ANALIZĂ MODULARĂ (SERII DE TIMP)")
    
    file_path = 'edat_lfse_03_linear_2_0.csv'
    if not os.path.exists(file_path):
        print(f"Eroare: Lipsă fișier date {file_path}")
        sys.exit(1)
        
    print("Încărcare bază de date Eurostat (poate dura câteva secunde)...")
    full_dataset = pd.read_csv(file_path)

    # Mini Meniu Interactiv
    print("\nOpțiuni valabile:")
    print("1. Introduceți codul unei țări (ex: RO, DE, FR, IT, HU)")
    print("2. Introduceți 'EU' pentru Uniunea Europeană (EU27_2020)")
    print("3. Introduceți 'ALL' pentru toate țările disponibile (Atenție: durează mult!)")
    print("4. Introduceți mai multe coduri separate prin virgulă (ex: RO, DE)")
    print("5. Apăsați ENTER pentru selecția implicită (RO, DE, HU)")
    
    user_input = input("\nAlegerile tale: ").strip().upper()
    
    if user_input == 'EU':
        COUNTRIES_TO_PROCESS = ['EU27_2020']
    elif user_input == 'ALL':
        COUNTRIES_TO_PROCESS = sorted(full_dataset['geo'].unique().tolist())
    elif ',' in user_input:
        COUNTRIES_TO_PROCESS = [c.strip() for c in user_input.split(',')]
    elif user_input != '':
        COUNTRIES_TO_PROCESS = [user_input]
    else:
        COUNTRIES_TO_PROCESS = ['RO', 'DE', 'HU'] # Default set

    for country in COUNTRIES_TO_PROCESS:
        print_header(f"PROCESARE: {country}")
        df_country = prepare_dataset(country, full_dataset)
        
        if not df_country.empty and len(df_country) >= 5:
            run_univariate(df_country, country)
            run_multivariate(df_country, country)
        else:
            print(f"Sărim peste {country} din cauza lipsei de date sau cod incorect.")

    print_header("TOATE ANALIZELE AU FOST FINALIZATE")
    print(f"Rezultatele se află în folderul: '{OUTPUT_DIR}'")