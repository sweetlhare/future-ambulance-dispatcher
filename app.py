import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static

from colour import Color
import holidays
from datetime import datetime
from haversine import haversine
import datetime as dt

from catboost import CatBoostClassifier


@st.cache(ttl=36000, max_entries=1000)
def load_models():   
    return CatBoostClassifier().load_model(Path('models/call')), \
        CatBoostClassifier().load_model(Path('models/heaviness')), \
            CatBoostClassifier().load_model(Path('models/result'))

call_model, heavy_model, result_model = load_models()


@st.cache(ttl=36000, max_entries=1000)
def load_data():

    substations = pd.read_excel('demonstration_data/substations.xlsx').rename(columns={'name':'substation'})
    substations = substations[substations.substation.isin(['ПСМП №5', 'ПСМП №7', 'ПСМП №1', 'ПСМП №3', 'ПСМП №8', 'ПСМП №4', 'ПСМП №2', 'ПСМП №9', 'ПСМП №6'])]
    substations['latitude_x'] = substations.coords.apply(lambda x: float(x.split(',')[0]))
    substations['longitude_x'] = substations.coords.apply(lambda x: float(x.split(',')[1]))
    substations.coords = substations[['latitude_x', 'longitude_x']].apply(lambda x: tuple([x[0], x[1]]), axis=1)

    weather = pd.read_csv('demonstration_data/weather_2022.csv', encoding='utf-8', skiprows=6, sep=';', index_col=0)
    weather_columns = weather.columns[1:]
    weather = weather.drop(weather.columns[-1], axis=1)
    weather.columns = weather_columns
    weather = weather.reset_index().rename(columns={'index':'datetime'})
    weather['datetime'] = pd.to_datetime(weather['datetime'])
    weather = weather.sort_values('datetime').reset_index(drop=True)
    weather = weather[['datetime', 'T', 'Tg', 'Po', 'Pa', 'U', 'ff3']]
    weather.datetime = pd.to_datetime(weather.datetime)

    return substations, weather

substations, weather = load_data()


def make_test():
    squares = []

    for i in range(0, 50):
        for j in range(0, 50):
            squares.append((56.194032 + i * (56.395527 - 56.194032) / 50,
                            43.787622 + j * (44.227140 - 43.787622) / 50))

    test = pd.DataFrame()

    test['coords'] = squares
    test['latitude_y'] = test.coords.apply(lambda x: x[0])
    test['longitude_y'] = test.coords.apply(lambda x: x[1])

    temp = pd.DataFrame()
    for i in range(len(substations)):
        temp[substations.substation.iloc[i]] = test.coords.apply(lambda x: haversine(x, substations.coords.iloc[i]))
        
    temp = temp.T

    nearest_subs = []
    for i in range(temp.shape[1]):
        nearest_subs.append(temp[i].sort_values().index[0])
        
    test['substation'] = nearest_subs
    test = pd.merge(test, substations, on='substation', how='left')
    test['dist'] = test[['coords_x', 'coords_y']].apply(lambda x: haversine(x[0], x[1]), axis=1)
    test = test.drop('coords_y', axis=1).rename(columns={'coords_x': 'coords'})


    test['datetime'] = dt.datetime.now()
    test['hour'] = test.datetime.apply(lambda x: x.hour)
    test['weekday'] = test.datetime.apply(lambda x: x.weekday())
    test['day'] = test.datetime.apply(lambda x: x.day)
    test['week'] = test.datetime.apply(lambda x: x.week)
    test['month'] = test.datetime.apply(lambda x: x.month)
    test['holiday'] = test.datetime.apply(lambda x: holidays.RUS().get(x))
    test['holiday'] = test['holiday'].fillna('')

    for col in weather.columns[1:]:
        test[col] = weather.loc[weather.datetime <= test['datetime'].iloc[0], col].iloc[-1]

    return test

test = make_test()


# <------------------------------------------------------------------------->

# st.set_page_config(layout="wide")

st.title('Скорая Медицинская Помощь')

st.header('Карта потенциальной загруженности')

# detection_model = load_model()
    
# <------------------------------------------------------------------------->   
def update_test(test, dttm):
    test['datetime'] = dttm
    test['hour'] = test.datetime.apply(lambda x: x.hour)
    test['weekday'] = test.datetime.apply(lambda x: x.weekday())
    test['day'] = test.datetime.apply(lambda x: x.day)
    test['week'] = test.datetime.apply(lambda x: x.week)
    test['month'] = test.datetime.apply(lambda x: x.month)
    test['holiday'] = test.datetime.apply(lambda x: holidays.RUS().get(x))
    test['holiday'] = test['holiday'].fillna('')
    for col in weather.columns[1:]:
        test[col] = weather.loc[weather.datetime <= test['datetime'].iloc[0], col].iloc[-1]
    return test

def predict_test(test, model):
    test_res = pd.DataFrame()
    test_res['substation'] = test.substation
    test_res['latitude'] = test.latitude_y
    test_res['longitude'] = test.longitude_y
    test_res['call_target'] = model.predict_proba(test[model.feature_names_])[:, -1]
    return test_res


temp = pd.read_csv('demonstration_data/data_for_demo.csv')
cat_features = ['substation', 'holiday', 'who', 'calling', 'type', 'occasion']
temp[cat_features] = temp[cat_features].fillna('')
temp['Прогнозируемый Результат'] = result_model.predict(temp[result_model.feature_names_])[0][0]
temp['Прогнозируемая Тяжесть'] = round(heavy_model.predict_proba(temp[heavy_model.feature_names_])[0, -1], 2)
temp = temp.rename(columns={'occasion': 'Повод', 'calling': 'Вызов', 'type': 'Вид', 'substation': 'Подстанция', 'adress':'Адрес'})


dttm = st.date_input(
     "Выберите дату",
     dt.date(2022, 6, 5))
dttm = pd.to_datetime(str(dttm)+' '+str(dt.datetime.now().time()))
test = update_test(test, dttm)

indexes = np.random.randint(0, temp.shape[0], np.random.randint(0, 5))
if st.button('Обновить'):
    test = update_test(test, dttm)
    indexes = np.random.randint(0, temp.shape[0], np.random.randint(0, 5))

df = predict_test(test, call_model)
start_color = Color('#FFC699')
colors = list(start_color.range_to(Color('#000000'), 20))
def heat_col(x):
    i = max(0, int((((x-0.5)/0.5)**10)*20))
    return colors[i].get_hex()
df['color'] = df['call_target'].apply(lambda x:heat_col(x) )
map = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=11)
        
# ВЕРОЯТНОСТИ ВЫЗОВА СКОРОЙ
df[df.call_target > 0.5].apply(lambda x:folium.Circle(location=[x['latitude'], x['longitude']], 
radius=200, fill=True, opacity=0.2, fill_opacity=0.5, color=x['color'], popup=x['call_target']).add_to(map), axis=1)

# ПОДСТАНЦИИ
substations.apply(lambda x:folium.Marker(location=[x['latitude_x'], x['longitude_x']], icon=folium.Icon(color='lightred'), popup=x['substation']).add_to(map), axis=1)

# ДОСТУПНЫЕ СКОРЫЕ
folium.Marker(location=[56.297904, 44.027806], icon=folium.Icon(color='white'), popup='Скорая').add_to(map)
folium.Marker(location=[56.275912, 44.038082], icon=folium.Icon(color='white'), popup='Скорая').add_to(map)

# ВЫЗОВ
for idx in indexes:
    folium.Marker(location=[temp.latitude_y.iloc[idx], temp.longitude_y.iloc[idx]], icon=folium.Icon(color='black'), popup='Вызов\n{}'.format(temp['Адрес'].iloc[idx])).add_to(map)

# ВЫВОД КАРТЫ
folium_static(map)

# АКТУАЛЬНЫЕ ВЫЗОВЫ
st.header('АКТУАЛЬНЫЕ ВЫЗОВЫ')

st.table(temp[['Адрес', 'Повод', 'Вызов', 'Вид', 'Подстанция', 'Прогнозируемая Тяжесть', 'Прогнозируемый Результат']].iloc[indexes].set_index('Адрес'))