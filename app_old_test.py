import streamlit as st
import lightgbm as lgb
import numpy as np

# Заголовок приложения
st.title("ML Model Prediction App")

# Загрузка модели LightGBM
model = lgb.Booster(model_file='model/model.txt')

# Функция для предсказания
def predict(input_data):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    return prediction[0]

# Создание формы для ввода данных
st.write("Введите значения для предсказания модели:")

# Пример признаков, которые нужно ввести (замени на свои)
battery_power = st.number_input('Battery Power (mAh)', min_value=500, max_value=2000)
clock_speed = st.number_input('Clock Speed (GHz)', min_value=0.5, max_value=3.0)
ram = st.number_input('RAM (MB)', min_value=256, max_value=4000)

# Можно добавить больше признаков по мере необходимости

# Когда пользователь нажимает кнопку "Предсказать"
if st.button('Предсказать'):
    input_data = [battery_power, clock_speed, ram]
    
    # Вызываем функцию предсказания
    result = predict(input_data)
    
    # Отображаем результат
    st.success(f"Предсказание модели: {result}")