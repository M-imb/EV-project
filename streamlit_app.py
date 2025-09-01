import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="📊 Аналитика электромобилей", layout="wide")
st.title("📊 EV Project - Анализ и прогноз")
st.write("## Расширенный анализ и моделирование данных")

# --- Загрузка данных ---
st.set_page_config(page_title="📊 Аналитика электромобилей", layout="wide")
st.title("📊 EV Project - Анализ и прогноз")
st.write("## Расширенный анализ и моделирование данных")

# --- Загрузка данных ---
df = pd.read_parquet("Electric_Vehicle_Population_Data.parquet", engine="pyarrow")

# --- Предварительная обработка и фильтрация ---
df.rename(columns={
    'Model Year': 'year',
    'Make': 'manufacturer',
    'Model': 'model',
    'Electric Vehicle Type': 'ev_type',
    'Electric Range': 'ev_range',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'cafv_eligible',
    'Postal Code': 'postal_code',
    'City': 'city',
    'State': 'state',
    'County': 'county',
    'Electric Utility': 'utility'
}, inplace=True)

df.drop(columns=[
    'VIN (1-10)',
    'Base MSRP',
    'Legislative District',
    'DOL Vehicle ID',
    'Vehicle Location',
    '2020 Census Tract'
], inplace=True, errors='ignore')

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df.dropna(subset=['year'], inplace=True)
df['year'] = df['year'].astype(int)

if df.empty:
    st.error("Нет данных для отображения")
    st.stop()
    

# --- Временной ряд ---
ts_df = df.groupby('year').size().reset_index(name='vehicle_count')
ts_df['year'] = pd.to_datetime(ts_df['year'], format='%Y')
ts_df.set_index('year', inplace=True)
ts_df = ts_df.asfreq('YS').fillna(0)

# --- Вкладки ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Общий обзор",
    "🏭 Производители",
    "🚗 Модели",
    "⏳ Прогноз",
    "📊 Временной анализ"
])

# --- Вкладка 1 ---
with tab1:
    st.subheader("Размерность данных после ETL")
    st.write(df.shape)
    st.subheader("Первые 5 строк")
    st.dataframe(df.head())
    st.subheader("Динамика регистраций по годам")
    fig, ax = plt.subplots()
    ts_df['vehicle_count'].plot(ax=ax, marker='o')
    ax.set_ylabel("Количество EV")
    ax.set_xlabel("Год")
    st.pyplot(fig)

# --- Вкладка 2 ---
with tab2:
    st.subheader("Топ‑5 производителей")
    st.bar_chart(df['manufacturer'].value_counts().head(5))
    st.subheader("Распределение по производителям")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(y='manufacturer', data=df, order=df['manufacturer'].value_counts().index[:10], ax=ax)
    st.pyplot(fig)

# --- Вкладка 3 ---
with tab3:
    st.subheader("Топ‑5 моделей")
    st.bar_chart(df['model'].value_counts().head(5))
    st.subheader("Топ‑5 комбинаций 'Марка‑Модель'")
    st.write(df.groupby(['manufacturer', 'model']).size().nlargest(5))

# --- Вкладка 4 ---
with tab4:
    st.subheader("Прогноз количества EV с помощью Prophet")
    prophet_df = ts_df.reset_index().rename(columns={'year': 'ds', 'vehicle_count': 'y'})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)
    st.pyplot(model.plot(forecast))
    st.pyplot(model.plot_components(forecast))

# --- Вкладка 5: Временной анализ ---
with tab5:
    st.subheader("Декомпозиция временного ряда")
    decomposition = seasonal_decompose(ts_df['vehicle_count'], model='additive', period=1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    decomposition.observed.plot(ax=axes[0], title='Наблюдаемый ряд')
    decomposition.trend.plot(ax=axes[1], title='Тренд')
    decomposition.seasonal.plot(ax=axes[2], title='Сезонность')
    decomposition.resid.plot(ax=axes[3], title='Остатки')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Тест Дики–Фуллера")
    result = adfuller(ts_df['vehicle_count'])
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    st.write("Critical Values:", result[4])
    if result[1] <= 0.05:
        st.success("Ряд является стационарным (p-value <= 0.05)")
    else:
        st.warning("Ряд не является стационарным (p-value > 0.05)")

    st.subheader("ACF и PACF")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(ts_df['vehicle_count'], ax=axes[0])
    axes[0].set_title('Автокорреляция (ACF)')
    plot_pacf(ts_df['vehicle_count'], ax=axes[1])
    axes[1].set_title('Частичная автокорреляция (PACF)')
    plt.tight_layout()
    st.pyplot(fig)

    st.info("ACF-график медленно затухает → наличие тренда. PACF показывает сильные корреляции на первых лагах.")

    st.subheader("Оценка Prophet (train/test split)")
    prophet_df = ts_df.reset_index().rename(columns={'year': 'ds', 'vehicle_count': 'y'})
    train_size = int(len(prophet_df) * 0.7)
    prophet_train = prophet_df.iloc[:train_size]
    prophet_test = prophet_df.iloc[train_size:]
    m = Prophet()
    m.fit(prophet_train)
    future = m.make_future_dataframe(periods=len(prophet_test), freq='YS')
    forecast = m.predict(future)
    prophet_pred = forecast['yhat'].iloc[train_size:]
    rmse = np.sqrt(mean_squared_error(prophet_test['y'], prophet_pred))
    mae = mean_absolute_error(prophet_test['y'], prophet_pred)
    r2 = r2_score(prophet_test['y'], prophet_pred)
    st.write(pd.DataFrame([[ 'Prophet', rmse, mae, r2 ]], columns=['Model','RMSE','MAE','R2']))
    st.pyplot(m.plot(forecast))
    st.pyplot(m.plot_components(forecast))

# --- Вкладка 6: Сравнение моделей ---
with tab6:
    st.subheader("Сравнение моделей прогнозирования")

    # --- Разделение данных ---
    train_size = int(len(ts_df) * 0.7)
    ts_train = ts_df.iloc[:train_size]
    ts_test = ts_df.iloc[train_size:]

    metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R2'])

    def evaluate_model(model_name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return pd.Series([model_name, rmse, mae, r2], index=metrics_df.columns)

    # --- 1. Регрессии ---
    X = ts_df.index.year.values.reshape(-1, 1)
    y = ts_df['vehicle_count'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(random_state=42)
    cb_model = CatBoostRegressor(random_state=42, verbose=0)

    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    cb_pred = cb_model.predict(X_test)

    metrics_df = pd.concat([
        metrics_df,
        evaluate_model('Linear Regression', y_test, lr_pred).to_frame().T,
        evaluate_model('Random Forest', y_test, rf_pred).to_frame().T,
        evaluate_model('CatBoost', y_test, cb_pred).to_frame().T
    ], ignore_index=True)

    # --- 2. ARIMA ---
    arima_model = ARIMA(ts_train['vehicle_count'], order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(len(ts_test))
    metrics_df = pd.concat([metrics_df, evaluate_model('ARIMA', ts_test['vehicle_count'], arima_pred).to_frame().T], ignore_index=True)

    # --- 3. LSTM ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ts_df)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 3
    X_train_lstm, y_train_lstm = create_sequences(train_data, seq_length)
    X_test_lstm, y_test_lstm = create_sequences(test_data, seq_length)

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=1, verbose=0)

    test_predictions_lstm = model_lstm.predict(X_test_lstm)
    test_predictions_lstm = scaler.inverse_transform(test_predictions_lstm)
    y_test_lstm = scaler.inverse_transform(y_test_lstm)

    metrics_df = pd.concat([metrics_df, evaluate_model('LSTM', y_test_lstm, test_predictions_lstm).to_frame().T], ignore_index=True)

    # --- 4. Prophet ---
    prophet_df = ts_df.reset_index().rename(columns={'year': 'ds', 'vehicle_count': 'y'})
    prophet_train = prophet_df.iloc[:train_size]
    prophet_test = prophet_df.iloc[train_size:]

    m = Prophet()
    m.fit(prophet_train)

    future = m.make_future_dataframe(periods=len(prophet_test), freq='YS')
    forecast = m.predict(future)
    prophet_pred = forecast['yhat'].iloc[train_size:]
    metrics_df = pd.concat([metrics_df, evaluate_model('Prophet', prophet_test['y'], prophet_pred).to_frame().T], ignore_index=True)

    # --- Вывод метрик ---
    st.dataframe(metrics_df.round(2))

    # --- График ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts_train.index, ts_train['vehicle_count'], label='Train', color='black')
    ax.plot(ts_test.index, ts_test['vehicle_count'], label='Test', color='gray')
    ax.plot(ts_test.index, lr_pred, label='Linear Regression', color='blue')
    ax.plot(ts_test.index, rf_pred, label='Random Forest', color='green')
    ax.plot(ts_test.index, cb_pred, label='CatBoost', color='purple')
    ax.plot(ts_test.index, arima_pred, label='ARIMA', color='red')
    ax.plot(ts_test.index[seq_length:], test_predictions_lstm.flatten(), label='LSTM', color='cyan')
    ax.plot(prophet_test['ds'], prophet_pred, label='Prophet', color='orange')
    ax.set_title('Сравнение прогнозов всех моделей')
    ax.set_xlabel('Год')
    ax.set_ylabel('Количество EV')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Выводы ---
    st.markdown("""
    **Выводы:**
    1. Ряд имеет сильный восходящий тренд — простые модели (Linear Regression) его улавливают.
    2. CatBoost и Random Forest лучше справляются с нелинейностями.
    3. ARIMA учитывает тренд и автокорреляцию.
    4. LSTM требует больше данных, но тоже даёт неплохой результат.
    5. Prophet автоматически находит тренды и сезонность, показывая высокую точность.
    """)
