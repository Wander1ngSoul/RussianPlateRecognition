import cv2
import re


def preprocess_plate(plate_img):
    try:
        plate_img = cv2.resize(plate_img, (0, 0), fx=2, fy=2)

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return binary
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return None


def validate_plate_text(text):
    try:
        text = re.sub(r'[^АВЕКМНОРСТУХ0-9]', '', text.upper())

        if len(text) not in [6, 7, 8]:
            return None

        if re.match(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2}$', text):
            return f"{text[:1]}{text[1:4]}{text[4:6]} {text[6:]}"
        return None
    except Exception as e:
        print(f"Ошибка валидации: {e}")
        return None