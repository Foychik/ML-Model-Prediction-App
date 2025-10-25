import streamlit as st
import lightgbm as lgb
import numpy as np
from collections import OrderedDict

# Заголовок приложения
st.title("ML Model Prediction App")

# Загрузка модели LightGBM
model = lgb.Booster(model_file='model/model.txt')

# Функция предсказания, предоставленная твоим тиммейтом
def predict(data: OrderedDict):
    '''
    Args:
      data - словарь, с ключами в таком порядке:
        'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep',
        'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time',
        'three_g', 'touch_screen', 'wifi'
    Returns:
      Значение предсказанного класса (от 0 до 3)
    '''
    # Преобразование словаря значений в numpy массив
    input_array = np.array(list(data.values())).reshape(1, -1)
    
    # Предсказание с помощью модели
    prediction = model.predict(input_array)
    
    # Возвращаем индекс класса с наибольшей вероятностью
    predicted_class = np.argmax(prediction, axis=1)
    
    return predicted_class[0]

# Форма для ввода данных
st.write("Введите значения для предсказания модели:")

battery_power = st.number_input('Battery Power (mAh)', min_value=500, max_value=2000)
blue = st.selectbox('Has Bluetooth?', [0, 1])
clock_speed = st.number_input('Clock Speed (GHz)', min_value=0.5, max_value=3.0)
dual_sim = st.selectbox('Has Dual SIM?', [0, 1])
fc = st.number_input('Front Camera Resolution (MP)', min_value=0, max_value=19)
four_g = st.selectbox('Has 4G?', [0, 1])
int_memory = st.number_input('Internal Memory (GB)', min_value=2, max_value=64)
m_dep = st.number_input('Mobile Depth (cm)', min_value=0.1, max_value=1.0)
mobile_wt = st.number_input('Mobile Weight (g)', min_value=80, max_value=200)
n_cores = st.number_input('Number of Cores', min_value=1, max_value=8)
pc = st.number_input('Primary Camera Resolution (MP)', min_value=0, max_value=20)
px_height = st.number_input('Pixel Height', min_value=0, max_value=1960)
px_width = st.number_input('Pixel Width', min_value=500, max_value=1998)
ram = st.number_input('RAM (MB)', min_value=256, max_value=4000)
sc_h = st.number_input('Screen Height (cm)', min_value=5, max_value=19)
sc_w = st.number_input('Screen Width (cm)', min_value=0, max_value=18)
talk_time = st.number_input('Talk Time (hours)', min_value=2, max_value=20)
three_g = st.selectbox('Has 3G?', [0, 1])
touch_screen = st.selectbox('Has Touch Screen?', [0, 1])
wifi = st.selectbox('Has WiFi?', [0, 1])

# Когда пользователь нажимает кнопку "Предсказать"
if st.button('Предсказать'):
    # Создаем словарь с введёнными данными
    input_data = OrderedDict([
        ('battery_power', battery_power),
        ('blue', blue),
        ('clock_speed', clock_speed),
        ('dual_sim', dual_sim),
        ('fc', fc),
        ('four_g', four_g),
        ('int_memory', int_memory),
        ('m_dep', m_dep),
        ('mobile_wt', mobile_wt),
        ('n_cores', n_cores),
        ('pc', pc),
        ('px_height', px_height),
        ('px_width', px_width),
        ('ram', ram),
        ('sc_h', sc_h),
        ('sc_w', sc_w),
        ('talk_time', talk_time),
        ('three_g', three_g),
        ('touch_screen', touch_screen),
        ('wifi', wifi)
    ])
    
    # Вызываем функцию предсказания
    result = predict(input_data)
    
    # Отображаем результат
    st.success(f"Предсказанный класс: {result}")
