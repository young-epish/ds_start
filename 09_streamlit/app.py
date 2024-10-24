import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import yaml
import time
from PIL import Image

# Загружаем модель LGBMClassifier
with open('models/LGB_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Загружаем config's
with open('configs/metrics.yaml', 'r') as f:
    metrics = yaml.safe_load(f)

with open('configs/feature_constraints.yaml', 'r') as f:
    borders = yaml.safe_load(f)

# Загружаем данные, которые использовались для обучения
train_data = pd.read_csv('data/train.csv', index_col=0)

train_data['Age'] = train_data['Age'].astype(int)
train_data['NumOfProducts'] = train_data['NumOfProducts'].astype(int)
train_data['IsActiveMember'] = train_data['IsActiveMember'].astype(int)
train_data['HasCrCard'] = train_data['HasCrCard'].astype(int)

bank_products = ['Кредитная карта', 'Дебетовая карта', 'Сберегательный счет', 'Инвестиционный счет']


# Заголовок
st.title('Прогноз оттока клиентов')
# Показываем откуда взяты данные
st.subheader('Подробную информацию о соревновании можно найти [здесь](https://www.kaggle.com/competitions/playground-series-s4e1/overview)')

st.image(Image.open('data/customer_churn.jpeg'))

# Показываем исходные данные
st.header('Исходные данные:')
if st.toggle('Показать описание переменных'):
    st.markdown("""
    \r- `Customer ID`: Уникальный идентификатор для каждого клиента
    \r- `Surname`: Фамилия клиента
    \r- `Credit` Score: Числовое значение, представляющее кредитный рейтинг клиента
    \r- `Geography`: Страна, в которой проживает клиент (Франция, Испания или Германия)
    \r- `Gender`: Пол клиента (Мужчина или Женщина)
    \r- `Age`: Возраст клиента
    \r- `Tenure`: Количество лет, в течение которых клиент обслуживается в банке
    \r- `Balance`: Баланс на счете клиента
    \r- `NumOfProducts`: Количество банковских продуктов, которыми пользуется клиент (например, сберегательный счет, кредитная карта)
    \r- `HasCrCard`: Есть ли у клиента кредитная карта (1 = да, 0 = нет)
    \r- `IsActiveMember`: Является ли клиент активным членом (1 = да, 0 = нет)
    \r- `EstimatedSalary`: Предполагаемая зарплата клиента
    \r- `Exited`: Покинул ли клиент банк (1 = да, 0 = нет)                
""")

if st.toggle('Вывести полный DataFrame'):
    st.dataframe(train_data)
else:
    st.dataframe(train_data.head(5))

# Информируем пользователя о модели
st.markdown(f"""## Используемая модель:
    \r- `RandomForest` - для выбора наиболее важных признаков 
    \r- `LightGBM` - для окончательного результата
    \nЦелевая метрика на отложенной выборке: **{list(metrics)[0]}** = {metrics[list(metrics)[0]]:.2f}"""
)

# Форма для ввода данных пользователем
st.header('Введите параметры для предсказания')

# Поля для ввода пользователем данных
age = st.slider('Возраст', 
                min_value=borders['Age']['min'], max_value=borders['Age']['max'], 
                value=30, step=1)

balance = st.number_input('Баланс на счете', 
                          min_value=borders['Balance']['min'], max_value=borders['Balance']['max'], 
                          value=50000.0)

products = st.multiselect('Используемые продукты банка', options=bank_products, default=bank_products[0])
num_of_products = len(products)
# Отображение выбранных пользователем продуктов
if num_of_products  == 0:
    st.error('Пожалуйста, выберите хотя бы один продукт.')
st.write(f'Вы выбрали {num_of_products} продукт(а)')

is_active_member = st.checkbox('Активный клиент')

geography = st.selectbox('География (страна проживания)', borders['Geography'].keys())

gender = st.radio('Пол', borders['Gender'].keys())

# Преобразование введенных данных в формат, подходящий для модели
input_data = pd.DataFrame({
    'Age': [age],
    'NumOfProducts': [num_of_products],
    'IsActiveMember': [int(is_active_member)],
    'Balance': [balance],
    'Geography': borders['Geography'][geography],
    'Gender': borders['Gender'][gender]
})

# Кнопка для предсказания
if st.button('Предсказать вероятность оттока'):
    # Рассчет вероятности оттока
    prediction = model.predict_proba(input_data)[0][1]
    
    bar_p = st.progress(0)

    for percentage_complete in range(100):
        time.sleep(0.005)
        bar_p.progress(percentage_complete + 1)
    
    st.subheader(f'Вероятность оттока: {prediction * 100:.2f}%')