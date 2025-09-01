import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import plotly.express as px
import plotly.graph_objects as go
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
@st.cache_data
def load_data():
url = "https://raw.githubusercontent.com/M-imb/EV_final-project/master/Electric_Vehicle_Population_Data.parquet"
try:
df = pd.read_parquet(url, engine="pyarrow")
return df
except Exception as e:
st.error(f"Ошибка загрузки данных: {e}")
return pd.DataFrame()

df = load_data()

if df.empty:
st.stop()

# --- Предварительная обработка и фильтрация ---
df.columns = df.columns.str.replace(' ', '_')
df['Base_MSRP'] = pd.to_numeric(df['Base_MSRP'], errors='coerce')
df = df.dropna(subset=["Electric_Vehicle_Type", "Model_Year", "Make", "Electric_Range", "Base_MSRP", "Vehicle_Location"])

# Добавление новых признаков на основе цены
df['MSRP_Category'] = pd.cut(df['Base_MSRP'], bins=[0, 30000, 50000, 80000, np.inf], labels=['Low', 'Mid', 'High', 'Premium'])
df['EV_Type_Simplified'] = df['Electric_Vehicle_Type'].apply(lambda x: 'BEV' if 'Battery' in x else 'PHEV')

st.sidebar.header("Фильтры данных")
year_filter = st.sidebar.slider("Фильтр по году выпуска", int(df["Model_Year"].min()), int(df["Model_Year"].max()), (2010, 2022))

df_filtered = df[(df["Model_Year"] >= year_filter[0]) & (df["Model_Year"] <= year_filter[1])]

if df_filtered.empty:
st.warning("Нет данных после фильтрации по году. Попробуйте изменить диапазон лет.")
st.stop()

# --- Создание вкладок ---
tab1, tab2, tab3 = st.tabs(["📊 Анализ и визуализация", "📉 Прогноз (Регрессия)", "📈 Прогноз (Временные ряды)"])

# --- ВКЛАДКА 1: АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ---
with tab1:
st.header("Обзор и визуализация данных")
st.subheader("🔍 Случайные 10 строк отфильтрованных данных")
st.dataframe(df_filtered.sample(10), use_container_width=True)

st.subheader("📈 Основные графики")
col1, col2 = st.columns(2)
with col1:
fig_sales_by_year = px.histogram(df_filtered, x='Model_Year', title='Продажи электромобилей по годам')
st.plotly_chart(fig_sales_by_year, use_container_width=True)
with col2:
top_makes = df_filtered['Make'].value_counts().nlargest(10).index
df_top_makes = df_filtered[df_filtered['Make'].isin(top_makes)]
fig_make_sales = px.histogram(df_top_makes, x='Make', color='EV_Type_Simplified',
title='Продажи по производителям и типам EV', barmode='group')
st.plotly_chart(fig_make_sales, use_container_width=True)

st.subheader("🗺️ Географическая карта продаж")
try:
location_counts = df_filtered['Vehicle_Location'].value_counts().reset_index()
location_counts.columns = ['location', 'count']
if 'State' in df_filtered.columns:
sales_by_state = df_filtered['State'].value_counts().reset_index()
sales_by_state.columns = ['State', 'Count']
fig_geo = px.choropleth(sales_by_state, locations='State', locationmode="USA-states", color='Count',
scope="usa", color_continuous_scale="Viridis",
title='Продажи автомобилей по штатам США')
st.plotly_chart(fig_geo, use_container_width=True)
else:
st.warning("Для построения геокарты нужны данные о штатах. В исходном наборе их нет.")
except Exception as e:
st.error(f"Ошибка при построении геокарты: {e}")

st.subheader("🤝 Матрица корреляций")
correlation_df = df_filtered[['Model_Year', 'Electric_Range', 'Base_MSRP']].corr()
fig_corr = px.imshow(correlation_df, text_auto=True, color_continuous_scale='RdBu_r',
title='Матрица корреляций')
st.plotly_chart(fig_corr, use_container_width=True)

# --- Подготовка данных для регрессии и временных рядов ---
df_regression = df_filtered.copy()
df_regression['Make_encoded'] = ce.TargetEncoder(cols=['Make']).fit_transform(df_regression['Make'], df_regression['Base_MSRP'])
features_reg = ['Model_Year', 'Electric_Range', 'Make_encoded']
target_reg = 'Base_MSRP'
X_reg = df_regression[features_reg]
y_reg = df_regression[target_reg]

# Разделение данных для регрессии
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Масштабирование
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# --- ВКЛАДКА 2: ПРОГНОЗ (РЕГРЕССИЯ) ---
with tab2:
st.header("Прогнозирование цены (Регрессия)")
st.subheader("🛠️ Обучение моделей")
models_reg = {
'Linear Regression': LinearRegression(),
'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
'Random Forest': RandomForestRegressor(random_state=42)
}
metrics_reg = []
predictions_reg = {}
for name, model in models_reg.items():
model.fit(X_train_reg_scaled, y_train_reg)
y_pred = model.predict(X_test_reg_scaled)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
mae = mean_absolute_error(y_test_reg, y_pred)
r2 = r2_score(y_test_reg, y_pred)
metrics_reg.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})
predictions_reg[name] = y_pred

st.subheader("📊 Визуализация прогнозов")
fig_reg = go.Figure()
fig_reg.add_trace(go.Scatter(x=y_test_reg.index, y=y_test_reg, mode='markers', name='Истинные значения'))
for name, pred in predictions_reg.items():
fig_reg.add_trace(go.Scatter(x=y_test_reg.index, y=pred, mode='lines', name=f'Прогноз {name}'))
fig_reg.update_layout(title='Сравнение прогнозов моделей регрессии', xaxis_title='Индекс', yaxis_title='Base MSRP')
st.plotly_chart(fig_reg, use_container_width=True)

st.subheader("📋 Таблица метрик")
metrics_df_reg = pd.DataFrame(metrics_reg)
st.table(metrics_df_reg)

st.subheader("✨ Выводы")
best_model_r2 = metrics_df_reg.loc[metrics_df_reg['R²'].idxmax()]
st.write(f"Лучшая модель по метрике R²: **{best_model_r2['Model']}** с R² = {best_model_r2['R²']:.2f}")
st.write("CatBoost и Random Forest, как правило, показывают лучшие результаты, чем линейная регрессия, так как они способны улавливать нелинейные зависимости в данных.")

# --- Подготовка данных для временных рядов ---
df_ts = df_filtered.groupby('Model_Year')['Base_MSRP'].mean().reset_index()
df_ts['Model_Year'] = pd.to_datetime(df_ts['Model_Year'], format='%Y')
df_ts.set_index('Model_Year', inplace=True)
ts = df_ts['Base_MSRP']

# Разделение данных
train_ts = ts[:'2020']
test_ts = ts['2021':]

# --- ВКЛАДКА 3: ПРОГНОЗ (ВРЕМЕННЫЕ РЯДЫ) ---
with tab3:
st.header("Прогнозирование цены (Временные ряды)")
st.subheader("🔍 Декомпозиция временного ряда")
try:
decomposition = seasonal_decompose(ts, model='additive', period=1)
fig_decomp = go.Figure()
fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Тренд'))
fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Сезонность'))
fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Остатки'))
fig_decomp.update_layout(title='Декомпозиция временного ряда', xaxis_title='Год', yaxis_title='Base MSRP')
st.plotly_chart(fig_decomp, use_container_width=True)

st.subheader("🧐 Тест Дики-Фуллера")
adf_test = adfuller(ts)
st.write("Статистика теста:", adf_test[0])
st.write("p-value:", adf_test[1])
st.write("Критические значения:", adf_test[4])
if adf_test[1] < 0.05:
st.success("Ряд **стационарен** (p-value < 0.05).")
else:
st.warning("Ряд **не стационарен** (p-value >= 0.05). Рассмотрим дифференцирование.")

st.subheader("📈 Модель ARIMA")
try:
st.write("Графики ACF и PACF для определения параметров ARIMA:")
fig_acf = go.Figure(data=go.Scatter(x=np.arange(len(acf(ts, nlags=5))), y=acf(ts, nlags=5)))
fig_acf.update_layout(title='ACF Plot', xaxis_title='Lag', yaxis_title='ACF')
st.plotly_chart(fig_acf, use_container_width=True)
fig_pacf = go.Figure(data=go.Scatter(x=np.arange(len(pacf(ts, nlags=5))), y=pacf(ts, nlags=5)))
fig_pacf.update_layout(title='PACF Plot', xaxis_title='Lag', yaxis_title='PACF')
st.plotly_chart(fig_pacf, use_container_width=True)

p = st.number_input("Выберите значение p (AR):", min_value=0, max_value=5, value=1, step=1)
d = st.number_input("Выберите значение d (I):", min_value=0, max_value=2, value=0, step=1)
q = st.number_input("Выберите значение q (MA):", min_value=0, max_value=5, value=0, step=1)

model_arima = ARIMA(train_ts, order=(p, d, q))
model_arima_fit = model_arima.fit()
predictions_arima = model_arima_fit.forecast(steps=len(test_ts))

fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(x=train_ts.index, y=train_ts, mode='lines', name='Обучающая выборка'))
fig_arima.add_trace(go.Scatter(x=test_ts.index, y=test_ts, mode='lines', name='Тестовая выборка'))
fig_arima.add_trace(go.Scatter(x=test_ts.index, y=predictions_arima, mode='lines', name='Прогноз ARIMA'))
fig_arima.update_layout(title=f'Прогноз ARIMA (p={p}, d={d}, q={q})', xaxis_title='Год', yaxis_title='Base MSRP')
st.plotly_chart(fig_arima, use_container_width=True)

rmse_arima = np.sqrt(mean_squared_error(test_ts, predictions_arima))
st.write(f'RMSE для ARIMA: {rmse_arima:.2f}')

except Exception as e:
st.error(f"Ошибка при построении модели ARIMA: {e}")

st.subheader("🔮 Модель Prophet")
try:
df_prophet = df_ts.copy()
df_prophet = df_prophet.reset_index()
df_prophet.columns = ['ds', 'y']
model_prophet = Prophet()
model_prophet.fit(df_prophet[:-len(test_ts)])
future = model_prophet.make_future_dataframe(periods=len(test_ts), freq='YS')
forecast = model_prophet.predict(future)

fig_prophet = go.Figure()
fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'][:-len(test_ts)], y=df_prophet['y'][:-len(test_ts)], mode='lines', name='Обучающая выборка'))
fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'][-len(test_ts):], y=df_prophet['y'][-len(test_ts):], mode='lines', name='Тестовая выборка'))
fig_prophet.add_trace(go.Scatter(x=forecast['ds'][-len(test_ts):], y=forecast['yhat'][-len(test_ts):], mode='lines', name='Прогноз Prophet'))
fig_prophet.update_layout(title='Прогноз Prophet', xaxis_title='Год', yaxis_title='Base MSRP')
st.plotly_chart(fig_prophet, use_container_width=True)

predictions_prophet = forecast['yhat'][-len(test_ts):].values
rmse_prophet = np.sqrt(mean_squared_error(test_ts, predictions_prophet))
st.write(f'RMSE для Prophet: {rmse_prophet:.2f}')
except Exception as e:
st.error(f"Ошибка при построении модели Prophet: {e}")

except Exception as e:
st.error(f"Общая ошибка во вкладке 'Временные ряды': {e}")
