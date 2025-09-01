# 📦 Electro vehicle project
```
⬆️ (Replace above with your app's name)
```

Dataset contains the following columns, each representing different aspects of the electric vehicle (EV) population in the United States:

•	VIN (1-10): Partial Vehicle Identification Number.

•	County: The county in which the vehicle is registered.

•	City: The city in which the vehicle is registered.

•	State: The state in which the vehicle is registered. It appears that this dataset may be focused on Washington (WA) state.

•	Postal Code: The postal code where the vehicle is registered.

•	Model Year: The year of the vehicle model.

•	Make: The manufacturer of the vehicle.

•	Model: The model of the vehicle.

•	Electric Vehicle Type: The type of electric vehicle, e.g., Battery Electric Vehicle (BEV).

•	Clean Alternative Fuel Vehicle (CAFV) Eligibility: Eligibility status for clean alternative fuel vehicle programs.

•	Electric Range: The maximum range of the vehicle on a single charge (in miles). -Base MSRP: The Manufacturer’s Suggested Retail Price.

•	Legislative District: The legislative district where the vehicle is registered.

•	DOL Vehicle ID: Department of Licensing Vehicle Identification.

•	Vehicle Location: Geographic coordinates of the vehicle location.

•	Electric Utility: The electric utility service provider for the vehicle’s location.

•	2020 Census Tract: The census tract for the vehicle’s location

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-starter-kit.streamlit.app/)

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Section Heading

Учитывая специфику датасета — короткий временной ряд и сильный восходящий тренд — лучшими моделями для прогнозирования будут:
1.	Facebook Prophet: Эта модель идеально подходит для таких данных. Она специально разработана для временных рядов с выраженным трендом и без ярко выраженной сезонности. Она автоматически определяет точки изменения тренда, что очень важно для экспоненциального роста, который мы видим.
2.	Holt-Winters: Эта модель также хорошо работает с трендами. Она менее сложна, чем Prophet, но очень эффективна для прогнозирования на короткие периоды. Она даст   понятное представление о прогнозе.


## Further Reading

.
