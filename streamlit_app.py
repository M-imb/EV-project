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
df = pd.read_parquet("Electric_Vehicle_Population_Data.parquet", engine="pyarrow")

# # --- Загрузка данных ---
# @st.cache_data
# def load_data():
#     url = "https://raw.githubusercontent.com/M-imb/EV-project/master/Electric_Vehicle_Population_Data.parquet"
#     try:
#         df = pd.read_parquet(url, engine="pyarrow")
#         return df
#     except Exception as e:
#         st.error(f"Ошибка загрузки данных: {e}")
#         return pd.DataFrame()


# df = load_data()
# if df.empty:
#     st.stop()


# --- Предварительная обработка и фильтрация ---
df.columns = df.columns.str.replace(' ', '_')
df['Base_MSRP'] = pd.to_numeric(df['Base_MSRP'], errors='coerce')
df = df.dropna(subset=["Electric_Vehicle_Type", "Model_Year", "Make", "Electric_Range", "Base_MSRP", "Vehicle_Location"])

# Добавление новых признаков на основе цены
df['MSRP_Category'] = pd.cut(df['Base_MSRP'],
                             bins=[0, 30000, 50000, 80000, np.inf],
                             labels=['Low', 'Mid', 'High', 'Premium'])
df['EV_Type_Simplified'] = df['Electric_Vehicle_Type'].apply(lambda x: 'BEV' if 'Battery' in x else 'PHEV')

st.sidebar.header("Фильтры данных")
year_filter = st.sidebar.slider(
    "Фильтр по году выпуска",
    int(df["Model_Year"].min()),
    int(df["Model_Year"].max()),
    (2010, 2022),
)
df_filtered = df[(df["Model_Year"] >= year_filter[0]) & (df["Model_Year"] <= year_filter[1])]

if df_filtered.empty:
    st.warning("Нет данных после фильтрации по году. Попробуйте изменить диапазон лет.")
    st.stop()


# --- Создание вкладок ---
tab1, tab2, tab3 = st.tabs([
    "📊 Анализ и визуализация",
    "📉 Прогноз (Регрессия)",
    "📈 Прогноз (Временные ряды)"
])


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
        fig_make_sales = px.histogram(
            df_top_makes,
            x='Make',
            color='EV_Type_Simplified',
            title='Продажи по производителям и типам EV',
            barmode='group'
        )
        st.plotly_chart(fig_make_sales, use_container_width=True)

    st.subheader("🗺️ Географическая карта продаж")
    try:
        location_counts = df_filtered['Vehicle_Location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']

        if 'State' in df_filtered.columns:
            sales_by_state = df_filtered['State'].value_counts().reset_index()
            sales_by_state.columns = ['State', 'Count']
            fig_geo = px.choropleth(
                sales_by_state,
                locations='State',
                locationmode="USA-states",
                color='Count',
                scope="usa",
                color_continuous_scale="Viridis",
                title='Продажи автомобилей по штатам США'
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.warning("Для построения геокарты нужны данные о штатах. В исходном наборе их нет.")
    except Exception as e:
        st.error(f"Ошибка при построении геокарты: {e}")

    st.subheader("🤝 Матрица корреляций")
    correlation_df = df_filtered[['Model_Year', 'Electric_Range', 'Base_MSRP']].corr()
    fig_corr = px.imshow(
        correlation_df,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Матрица корреляций'
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# --- Подготовка данных для регрессии и временных рядов ---
df_regression = df_filtered.copy()
df_regression['Make_encoded'] = ce.TargetEncoder(cols=['Make']).fit_transform(
    df_regression['Make'], df_regression['Base_MSRP']
)

features_reg = ['Model_Year', 'Electric_Range', 'Make_encoded']
target_reg = 'Base_MSRP'
X_reg = df_regression[features_reg]
y_reg = df_regression[target_reg]

# Разделение данных для регрессии
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Масштабирование
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)


# --- ВКЛАДКА 2: ПРОГНОЗ (СРАВНЕНИЕ ПО ГОДАМ ТОП-3 А/М) ---
with tab2:
    st.header("Прогнозирование продаж по годам (топ‑3 авто)")

    # Определяем топ‑3 авто по суммарным продажам в исторических данных
    top3_models = (
        df.groupby('Model')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    st.write(f"Топ‑3 модели по продажам: {', '.join(top3_models)}")

    # Фильтруем данные только по этим моделям
    df_top3 = df[df['Model'].isin(top3_models)]

    # Обучаем модели
    st.subheader("🛠️ Обучение моделей")
    models_reg = {
        'Linear Regression': LinearRegression(),
        'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }

    # Здесь предполагается, что X_train_reg_scaled, y_train_reg и т.д. уже подготовлены
    predictions_yearly = []

    for name, model in models_reg.items():
        model.fit(X_train_reg_scaled, y_train_reg)
        y_pred = model.predict(X_test_reg_scaled)

        # Формируем DataFrame с прогнозом и годом
        pred_df = pd.DataFrame({
            'Year': X_test_reg['Year'],  # предполагается, что в X_test_reg есть колонка Year
            'ModelName': X_test_reg['Model'],  # и колонка Model
            'PredictedSales': y_pred
        })

        # Оставляем только топ‑3 авто
        pred_df = pred_df[pred_df['ModelName'].isin(top3_models)]

        # Группируем по году и модели авто
        yearly_sales = (
            pred_df.groupby(['Year', 'ModelName'])['PredictedSales']
            .sum()
            .reset_index()
        )
        yearly_sales['ML_Model'] = name
        predictions_yearly.append(yearly_sales)

    # Объединяем прогнозы всех моделей
    predictions_yearly_df = pd.concat(predictions_yearly, ignore_index=True)

    # 📊 Визуализация: группированный столбчатый график
    st.subheader("📊 Сравнение прогнозов по годам для топ‑3 авто")
    fig_yearly = px.bar(
        predictions_yearly_df,
        x='Year',
        y='PredictedSales',
        color='ML_Model',
        barmode='group',
        facet_row='ModelName',
        title='Прогноз годовых продаж (топ‑3 авто)',
        labels={'PredictedSales': 'Продажи', 'Year': 'Год'}
    )
    st.plotly_chart(fig_yearly, use_container_width=True)

    # 📋 Таблица с прогнозами
    st.subheader("📋 Таблица прогнозов")
    st.dataframe(predictions_yearly_df.sort_values(['ModelName', 'Year', 'ML_Model']))

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
    except Exception as e:
        st.error(f"Ошибка декомпозиции: {e}")

    st.subheader("🧐 Тест Дики-Фуллера")
    adf_test = adfuller(ts)
    st.write("Статистика теста:", adf_test[0])
    st.write("p-value:", adf_test[1])
    st.write("Критические значения:", adf_test[4])
    if adf_test[1] < 0.05:
        st.success("Ряд стационарен (p-value < 0.05).")
    else:
        st.warning("Ряд не стационарен (p-value ≥ 0.05). Рассмотрим дифференцирование.")

    st.subheader("📈 Модель ARIMA")
    try:
        st.write("Графики ACF и PACF для подбора параметров ARIMA:")
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
        df_prophet = df_ts.reset_index().rename(columns={'Model_Year': 'ds', 'Base_MSRP': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(df_prophet[:-len(test_ts)])

        future = model_prophet.make_future_dataframe(periods=len(test_ts), freq='YS')
        forecast = model_prophet.predict(future)

        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'][:-len(test_ts)], y=df_prophet['y'][:-len(test_ts)],
                                         mode='lines', name='Обучающая выборка'))
        fig_prophet.add_trace(go.Scatter(x=df_prophet['ds'][-len(test_ts):], y=df_prophet['y'][-len(test_ts):],
                                         mode='lines', name='Тестовая выборка'))
        fig_prophet.add_trace(go.Scatter(x=forecast['ds'][-len(test_ts):], y=forecast['yhat'][-len(test_ts):],
                                         mode='lines', name='Прогноз Prophet'))
        fig_prophet.update_layout(title='Прогноз Prophet', xaxis_title='Год', yaxis_title='Base MSRP')
        st.plotly_chart(fig_prophet, use_container_width=True)

        predictions_prophet = forecast['yhat'][-len(test_ts):].values
        rmse_prophet = np.sqrt(mean_squared_error(test_ts, predictions_prophet))
        st.write(f'RMSE для Prophet: {rmse_prophet:.2f}')
    except Exception as e:
        st.error(f"Ошибка при построении модели Prophet: {e}")
