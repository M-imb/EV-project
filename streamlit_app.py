import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pyarrow
import warnings
warnings.filterwarnings('ignore')

# --- Настройка страницы ---
st.set_page_config(page_title="📊 Аналитика электромобилей", layout="wide")
st.title("📊 EV Project - Анализ и прогноз")
st.write("## Расширенный анализ и моделирование данных")

# --- Загрузка данных ---
df = pd.read_parquet("Electric_Vehicle_Population_Data.parquet")

df.rename(columns={
    'Model Year': 'year',
    'Make': 'manufacturer',
    'Model': 'model',
    'Electric Vehicle Type': 'ev_type',
    'Electric Range': 'ev_range',
    'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'cafv_eligible',
    'City': 'city',
    'State': 'state',
    'County': 'county',
    'Electric Utility': 'utility'
}, inplace=True)

df.drop(columns=[
    'VIN (1-10)','Base MSRP','Legislative District','DOL Vehicle ID', 'Postal Code',
    'Vehicle Location','2020 Census Tract'
], inplace=True, errors='ignore')

df['year'] = pd.to_numeric(df['year'], errors='coerce')
df.dropna(subset=['year'], inplace=True)
df['year'] = df['year'].astype(int)

# --- Агрегация ---
ts_df = df.groupby('year').size().reset_index(name='vehicle_count')
ts_df['year'] = pd.to_datetime(ts_df['year'], format='%Y')
ts_df.set_index('year', inplace=True)
ts_df = ts_df.asfreq('YS').fillna(0)

# --- Вкладки ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Общий обзор","🏭 Производители","🚗 Модели",
    "⏳ Прогноз Prophet","📊 Временной анализ","🧪 Сравнение моделей"
])

# --- Вкладка 1 ---
with tab1:
    st.subheader("Размерность данных после ETL")
    st.write(df.shape)
    st.subheader("Первые 5 строк")
    st.dataframe(df.head())

    st.subheader("Динамика регистраций по годам")
    fig, ax = plt.subplots(figsize=(8,4))
    ts_df['vehicle_count'].plot(ax=ax)
    ax.set_ylabel("Количество EV")
    ax.set_xlabel("Год")
    st.pyplot(fig, clear_figure=True)

# --- Вкладка 2 ---
with tab2:
    st.subheader("Топ-10 производителей")
    fig, ax = plt.subplots(figsize=(8,4))
    manufacturer_counts = df['manufacturer'].value_counts().head(10)
    ax.barh(manufacturer_counts.index, manufacturer_counts.values)
    ax.set_xlabel("Количество")
    ax.set_ylabel("Производитель")
    st.pyplot(fig, clear_figure=True)

# --- Вкладка 3 ---
with tab3:
    st.subheader("Топ-10 моделей")
    st.bar_chart(df['model'].value_counts().head(10))
    st.subheader("Топ-5 комбинаций 'Марка-Модель'")
    st.write(df.groupby(['manufacturer','model']).size().nlargest(5))

# --- Вкладка 4 ---
with tab4:
    st.subheader("Прогноз количества EV (Prophet)")
    if st.button("Запустить Prophet"):
        prophet_df = ts_df.reset_index().rename(columns={'year': 'ds','vehicle_count':'y'})
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=5, freq='Y')
        forecast = model.predict(future)
        st.pyplot(model.plot(forecast), clear_figure=True)
        st.pyplot(model.plot_components(forecast), clear_figure=True)

# --- Вкладка 5 ---
with tab5:
    st.subheader("Декомпозиция временного ряда")
    decomposition = seasonal_decompose(ts_df['vehicle_count'], model='additive', period=2)
    fig, axes = plt.subplots(4,1,figsize=(8,6),sharex=True)
    decomposition.observed.plot(ax=axes[0],title='Наблюдаемый ряд')
    decomposition.trend.plot(ax=axes[1],title='Тренд')
    decomposition.seasonal.plot(ax=axes[2],title='Сезонность')
    decomposition.resid.plot(ax=axes[3],title='Остатки')
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Тест Дики–Фуллера")
    result = adfuller(ts_df['vehicle_count'])
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        st.success("Ряд стационарный")
    else:
        st.warning("Ряд не стационарный")

    st.subheader("ACF и PACF")
    fig, axes = plt.subplots(2,1,figsize=(8,4))
    plot_acf(ts_df['vehicle_count'], ax=axes[0])
    plot_pacf(ts_df['vehicle_count'], ax=axes[1])
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# --- Вкладка 6 ---
with tab6:
    st.subheader("Сравнение моделей (быстрые)")
    train_size = int(len(ts_df)*0.8)
    ts_train, ts_test = ts_df.iloc[:train_size], ts_df.iloc[train_size:]

    metrics_df = pd.DataFrame(columns=['Model','RMSE','MAE','R2'])

    def evaluate(model, y_true, y_pred):
        return pd.Series([
            model,
            np.sqrt(mean_squared_error(y_true,y_pred)),
            mean_absolute_error(y_true,y_pred),
            r2_score(y_true,y_pred)
        ], index=metrics_df.columns)

    # Линейная, RF, CatBoost
    X = ts_df.index.year.values.reshape(-1,1)
    y = ts_df['vehicle_count'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

    lr = LinearRegression().fit(X_train,y_train)
    rf = RandomForestRegressor(random_state=42).fit(X_train,y_train)
    cb = CatBoostRegressor(random_state=42,verbose=0).fit(X_train,y_train)

    preds = {
        "Linear Regression": lr.predict(X_test),
        "Random Forest": rf.predict(X_test),
        "CatBoost": cb.predict(X_test),
        "ARIMA": ARIMA(ts_train['vehicle_count'],order=(1,1,1)).fit().forecast(len(ts_test)),
        "Holt-Winters": ExponentialSmoothing(ts_train['vehicle_count'],trend='add').fit().forecast(len(ts_test))
    }

    for name,pred in preds.items():
        metrics_df = pd.concat([metrics_df, evaluate(name, ts_test['vehicle_count'], pred).to_frame().T])

    st.dataframe(metrics_df.round(2).sort_values(by="R2",ascending=False))

    st.subheader("График прогнозов")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(ts_df.index, ts_df['vehicle_count'], label="Фактические", color="black")
    for name,pred in preds.items():
        ax.plot(ts_test.index, pred, linestyle="--", label=name)
    ax.legend()
    ax.set_title("Сравнение моделей")
    st.pyplot(fig, clear_figure=True)

    st.subheader("LSTM (по кнопке)")
    if st.button("Запустить LSTM"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(ts_df[['vehicle_count']])
        train, test = scaled[:train_size], scaled[train_size:]

        def seq(data, n=3):
            X,y = [],[]
            for i in range(len(data)-n):
                X.append(data[i:i+n]); y.append(data[i+n])
            return np.array(X), np.array(y)

        X_train_lstm,y_train_lstm = seq(train,3)
        X_test_lstm,y_test_lstm = seq(test,3)

        model_lstm = Sequential([
            LSTM(50,activation='relu',input_shape=(3,1)),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam',loss='mse')
        model_lstm.fit(X_train_lstm,y_train_lstm,epochs=50,batch_size=1,verbose=0)

        pred_lstm = scaler.inverse_transform(model_lstm.predict(X_test_lstm))

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(ts_test.index[3:], scaler.inverse_transform(y_test_lstm), label="Истинные")
        ax.plot(ts_test.index[3:], pred_lstm, label="LSTM", linestyle="--")
        ax.legend()
        st.pyplot(fig, clear_figure=True)
