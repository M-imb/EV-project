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
st.write("## Расширенный анализ и моделирование данных .........доработать")

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
    st.subheader("Данные после ETL")
    st.write(df.shape)

    # Топ-5 марок по продажам вместо head()
    st.subheader("Топ-5 марок по продажам")
    top_manufacturers = df['manufacturer'].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(5,2))  
    ax.bar(top_manufacturers.index, top_manufacturers.values, color="skyblue")
    ax.set_ylabel("Количество а/м")
    ax.set_xlabel("Марка")
    st.pyplot(fig, clear_figure=True)

    # Динамика регистраций по годам 
    st.subheader("Динамика регистраций по годам")
    fig, ax = plt.subplots(figsize=(5,2))
    ts_df['vehicle_count'].plot(ax=ax, color="green")
    ax.set_ylabel("Количество EV")
    ax.set_xlabel("Год")
    st.pyplot(fig, clear_figure=True)

    # Сравнение по типу EV (ev_type)
    st.subheader("Сравнение по типу электромобилей (EV Type)")
    ev_type_counts = df['ev_type'].value_counts()
    fig, ax = plt.subplots(figsize=(5,2))
    ax.bar(ev_type_counts.index, ev_type_counts.values, color="orange")
    ax.set_ylabel("Количество EV")
    ax.set_xlabel("Тип EV")
    st.pyplot(fig, clear_figure=True)

# --- Вкладка 2 ---
with tab2:
    st.subheader("Топ-5 производителей")
    fig, ax = plt.subplots(figsize=(5,2))
    manufacturer_counts = df['manufacturer'].value_counts().head(5)
    ax.barh(manufacturer_counts.index, manufacturer_counts.values)
    ax.set_xlabel("Количество")
    ax.set_ylabel("Производитель")
    st.pyplot(fig, clear_figure=True)

    # --- Топ-3 марки по годам ---
    st.subheader("Динамика продаж топ-3 марок")
    top3_brands = df['manufacturer'].value_counts().head(3).index.tolist()
    brand_data = (
        df[df['manufacturer'].isin(top3_brands)]
        .groupby(['year','manufacturer'])
        .size()
        .reset_index(name="sales")
    )
    fig, ax = plt.subplots(figsize=(6,3))
    for brand in top3_brands:
        subset = brand_data[brand_data['manufacturer']==brand]
        ax.plot(subset['year'], subset['sales'], label=brand)
    ax.set_title("Исторические продажи топ-3 марок")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # --- Прогноз по топ-3 ---
    st.subheader("Прогноз продаж топ-3 марок")
    model_choice = st.selectbox("Выбери модель:", ["Prophet","Linear Regression","Random Forest"], key="brand_model")
    horizon = st.slider("Горизонт прогноза (лет)", 1, 5, 3, 1, key="brand_horizon")

    for brand in top3_brands:
        st.markdown(f"### {brand}")
        subset = brand_data[brand_data["manufacturer"]==brand][["year","sales"]].copy()
        subset = subset.rename(columns={"year":"ds","sales":"y"})

        if model_choice=="Prophet":
            m = Prophet(yearly_seasonality=True)
            m.fit(subset)
            future = m.make_future_dataframe(periods=horizon, freq="Y")
            forecast = m.predict(future)
            fig1 = m.plot(forecast, xlabel="Год", ylabel="Продажи")
            st.pyplot(fig1, clear_figure=True)

        elif model_choice=="Linear Regression":
            X = np.array(subset["ds"].dt.year).reshape(-1,1)
            y = subset["y"].values
            lr = LinearRegression().fit(X,y)
            future_years = np.arange(subset["ds"].dt.year.max()+1, subset["ds"].dt.year.max()+horizon+1)
            future_pred = lr.predict(future_years.reshape(-1,1))

            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.plot(subset["ds"].dt.year, y, label="Факт")
            ax2.plot(future_years, future_pred, linestyle="--", label="Прогноз")
            ax2.legend(); ax2.set_title(f"Прогноз продаж {brand}")
            st.pyplot(fig2, clear_figure=True)

            # Метрики качества на последних 20% данных
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
            y_pred = lr.predict(X_test)
            st.caption(f"MAE={mean_absolute_error(y_test,y_pred):.2f}, RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):.2f}, R²={r2_score(y_test,y_pred):.2f}")

        elif model_choice=="Random Forest":
            X = np.array(subset["ds"].dt.year).reshape(-1,1)
            y = subset["y"].values
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X,y)
            future_years = np.arange(subset["ds"].dt.year.max()+1, subset["ds"].dt.year.max()+horizon+1)
            future_pred = rf.predict(future_years.reshape(-1,1))

            fig3, ax3 = plt.subplots(figsize=(6,3))
            ax3.plot(subset["ds"].dt.year, y, label="Факт")
            ax3.plot(future_years, future_pred, linestyle="--", label="Прогноз")
            ax3.legend(); ax3.set_title(f"Прогноз продаж {brand}")
            st.pyplot(fig3, clear_figure=True)

            # Метрики качества на последних 20% данных
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
            y_pred = rf.predict(X_test)
            st.caption(f"MAE={mean_absolute_error(y_test,y_pred):.2f}, RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):.2f}, R²={r2_score(y_test,y_pred):.2f}")

# --- Вкладка 3 ---
with tab3:
    st.header("📊 Топ-3 модели по продажам")
    st.write("Колонки в df:", df.columns.tolist())
    top_models = df['model'].value_counts().head(3).index
    st.write("Топ-3 модели:", ", ".join(top_models))

    df_top_models = df[df['model'].isin(top_models)]
    model_trends = df_top_models.groupby(['year', 'model']).size().reset_index(name='count')

    fig3 = px.line(model_trends, x='year', y='count', color='model', title="Динамика продаж топ-3 моделей")
    st.plotly_chart(fig3, use_container_width=True)

    # Выбор модели для прогноза
    selected_model = st.selectbox("Выберите модель для прогноза:", top_models)
    df_model = df_top_models[df_top_models['model'] == selected_model].groupby('year').size().reset_index(name='count')

    horizon = st.slider("Горизонт прогноза (лет)", 1, 10, 5)

    # Prophet
    from prophet import Prophet
    df_model_prophet = df_model.rename(columns={'year': 'ds', 'count': 'y'})
    df_model_prophet['ds'] = pd.to_datetime(df_model_prophet['ds'], format='%Y')
    m = Prophet()
    m.fit(df_model_prophet)
    future = m.make_future_dataframe(periods=horizon, freq='Y')
    forecast = m.predict(future)

    fig_forecast = px.line(forecast, x='ds', y='yhat', title=f"Прогноз Prophet для {selected_model}")
    fig_forecast.add_scatter(x=df_model_prophet['ds'], y=df_model_prophet['y'], mode='lines+markers', name='Исторические данные')
    st.plotly_chart(fig_forecast, use_container_width=True)



# --- Вкладка 4 ---
with tab4:
    st.header("🔮 Прогноз Prophet (по типу EV)")

    top_ev_types = df['Electric Vehicle Type'].value_counts().index
    ev_choice = st.selectbox("Выберите тип EV для прогноза:", top_ev_types)

    df_ev = df[df['Electric Vehicle Type'] == ev_choice].groupby('year').size().reset_index(name='count')
    horizon_ev = st.slider("Горизонт прогноза (лет)", 1, 15, 7)

    if len(df_ev) > 2:
        df_ev_prophet = df_ev.rename(columns={'year': 'ds', 'count': 'y'})
        df_ev_prophet['ds'] = pd.to_datetime(df_ev_prophet['ds'], format='%Y')

        m_ev = Prophet()
        m_ev.fit(df_ev_prophet)
        future_ev = m_ev.make_future_dataframe(periods=horizon_ev, freq='Y')
        forecast_ev = m_ev.predict(future_ev)

        fig_ev = px.line(forecast_ev, x='ds', y='yhat', title=f"Прогноз для {ev_choice}")
        fig_ev.add_scatter(x=df_ev_prophet['ds'], y=df_ev_prophet['y'], mode='lines', name='История')
        st.plotly_chart(fig_ev, use_container_width=True)
    else:
        st.warning("Недостаточно данных для прогноза.")

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
    fig, axes = plt.subplots(2,1,figsize=(5,2))
    plot_acf(ts_df['vehicle_count'], ax=axes[0])
    plot_pacf(ts_df['vehicle_count'], ax=axes[1])
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Прогнозы ARIMA и Holt-Winters")

    hw_model = ExponentialSmoothing(ts_df['vehicle_count'], trend='add').fit()
    hw_forecast = hw_model.forecast(5)

    arima_model = ARIMA(ts_df['vehicle_count'], order=(1,1,1)).fit()
    arima_forecast = arima_model.forecast(steps=5)

    fig, ax = plt.subplots(figsize=(6,3))
    ts_df['vehicle_count'].plot(ax=ax, label="История")
    hw_forecast.plot(ax=ax, label="Holt-Winters")
    arima_forecast.plot(ax=ax, label="ARIMA")
    ax.legend()
    st.pyplot(fig)

    # Остатки ARIMA
    st.subheader("Остатки ARIMA")
    residuals = arima_model.resid
    fig_res, ax_res = plt.subplots(figsize=(6,2))
    residuals.plot(ax=ax_res)
    st.pyplot(fig_res)

# --- Вкладка 6 ---
with tab6:
    st.subheader("Сравнение моделей (LR, RF, CatBoost, ARIMA, Holt-Winters)")

    # Разделение данных
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Sales"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    results = []

    # ----- Linear Regression -----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results.append({
        "model": "Linear Regression",
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "R²": r2_score(y_test, y_pred_lr)
    })
    
    # ----- Random Forest -----
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append({
        "model": "Random Forest",
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "R²": r2_score(y_test, y_pred_rf)
    })

    # ----- CatBoost -----
    try:
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(iterations=200, depth=6, silent=True, random_state=42)
        cb.fit(X_train, y_train)
        y_pred_cb = cb.predict(X_test)
        results["CatBoost"] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_cb)),
            "MAE": mean_absolute_error(y_test, y_pred_cb),
            "R²": r2_score(y_test, y_pred_cb),
            "y_pred": y_pred_cb,
        }
    except:
        st.warning("CatBoost не установлен")

    # ----- ARIMA -----
    from statsmodels.tsa.arima.model import ARIMA
    try:
        arima_model = ARIMA(y_train, order=(2,1,2))
        arima_res = arima_model.fit()
        y_pred_arima = arima_res.forecast(len(y_test))
        results["ARIMA"] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_arima)),
            "MAE": mean_absolute_error(y_test, y_pred_arima),
            "R²": r2_score(y_test, y_pred_arima),
            "y_pred": y_pred_arima,
        }
    except:
        st.warning("ARIMA не смогла обучиться")

    # ----- Holt-Winters -----
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw = ExponentialSmoothing(y_train, trend="add", seasonal=None)
        hw_fit = hw.fit()
        y_pred_hw = hw_fit.forecast(len(y_test))
        results["Holt-Winters"] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_hw)),
            "MAE": mean_absolute_error(y_test, y_pred_hw),
            "R²": r2_score(y_test, y_pred_hw),
            "y_pred": y_pred_hw,
        }
    except:
        st.warning("Holt-Winters не смог обучиться")

    # В таблицу
    metrics_df = pd.DataFrame(results)
    st.dataframe(metrics_df.style.highlight_min(axis=0, color="lightgreen"))
    
    # Топ-3 по R²
    top_models = metrics_df.sort_values(by="R²", ascending=False).head(3)
    st.write("🔥 Топ-3 модели:", ", ".join(top_models["model"].tolist()))

    # ===== ВИЗУАЛ =====
    st.write("📊 Метрики моделей:")
    col1, col2, col3 = st.columns(3)
    best_model = min(results, key=lambda m: results[m]["RMSE"])
    col1.metric("📉 Лучшая модель", best_model, "")
    col2.metric("RMSE (↓)", f"{results[best_model]['RMSE']:.2f}")
    col3.metric("MAE (↓)", f"{results[best_model]['MAE']:.2f}")

    # Таблица с результатами
    metrics_df = pd.DataFrame({m: {"RMSE": r["RMSE"], "MAE": r["MAE"], "R²": r["R²"]} for m, r in results.items()}).T
    st.dataframe(metrics_df.style.highlight_min(axis=0, color="lightgreen"))

    # График прогнозов
    fig = px.line(title="Сравнение прогнозов моделей")
    fig.add_scatter(x=list(range(len(y))), y=y, mode="lines", name="Фактические", line=dict(color="black"))
    
    for model, r in zip(metrics_df["Model"], results):
        fig.add_scatter(
            x=list(range(len(y_train), len(y))),
            y=r["y_pred"],
            mode="lines+markers",
            name=model
        )
    
    st.plotly_chart(fig, use_container_width=True)
