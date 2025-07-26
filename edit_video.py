import cv2
from collections import defaultdict
from config import reader, plate_cascade
from utils import save_results
from edit_plate import preprocess_plate, validate_plate_text

def process_video(video_path):
    print(f"\nНачало обработки видео: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видеофайл")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Всего кадров: {total_frames}")

        frame_skip = 5
        plate_counter = defaultdict(int)
        found_plates = set()

        for frame_num in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 20))

            if len(plates) == 0:
                continue

            print(f"\nКадр {frame_num}: найдено {len(plates)} номеров")

            for i, (x, y, w, h) in enumerate(plates, 1):
                plate_img = frame[y:y + h, x:x + w]
                processed = preprocess_plate(plate_img)

                if processed is None:
                    continue

                print(f"  Номер {i}: обработка...")
                results = reader.readtext(processed, detail=1, batch_size=1,
                                          allowlist='АВЕКМНОРСТУХ0123456789')

                if not results:
                    print("  Ничего не распознано")
                    continue

                for res in results:
                    text = res[1]
                    confidence = res[2]
                    print(f"  Распознано: '{text}' (точность: {confidence:.2f})")

                    valid_text = validate_plate_text(text)
                    if valid_text:
                        print(f"  Валидный номер: {valid_text}")
                        plate_counter[valid_text] += 1

                        if plate_counter[valid_text] >= 2 and valid_text not in found_plates:
                            print(f"\n=== НАЙДЕН НОМЕР: {valid_text} ===\n")
                            found_plates.add(valid_text)
                            marked_frame = frame.copy()
                            cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(marked_frame, f"{valid_text} ({confidence:.2f})",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            save_results(marked_frame, valid_text)

    except Exception as e:
        print(f"Ошибка обработки видео: {e}")
    finally:
        cap.release()
        print("\nОбработка завершена")
        print("Найденные номера:", ", ".join(found_plates) if found_plates else "не найдено")