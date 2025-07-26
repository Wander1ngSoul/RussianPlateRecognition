from datetime import datetime
import cv2
import os

def save_results(image, plate_text, output_dir="results"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{output_dir}/plate_{plate_text.replace(' ', '')}_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"Сохранено: {filename}")
    except Exception as e:
        print(f"Ошибка сохранения: {e}")