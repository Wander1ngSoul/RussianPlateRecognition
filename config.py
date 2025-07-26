from easyocr import Reader
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

try:
    print("Инициализация EasyOCR...")
    reader = Reader(['ru'], gpu=False, verbose=False)
    print("EasyOCR успешно инициализирован")
except Exception as e:
    print(f"Ошибка инициализации EasyOCR: {e}")
    exit()

try:
    cascade_path = os.getenv('CASCADE')
    print(f"Загрузка каскада из {cascade_path}...")
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    if plate_cascade.empty():
        raise ValueError("Не удалось загрузить каскадный классификатор")
    print("Каскад успешно загружен")
except Exception as e:
    print(f"Ошибка загрузки каскада: {e}")
    exit()