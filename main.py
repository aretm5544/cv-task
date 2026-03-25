import cv2
import pandas as pd
import argparse
from ultralytics import YOLO

def main():
    # 1. Парсинг аргументов в терминале через argparse (как требует ТЗ)
    parser = argparse.ArgumentParser(description="Dodo Cleaning Detection Prototype")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    # 2. Инициализация: берем быструю модель YOLOv8 Nano
    
    model = YOLO('yolov8n.pt') 

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {args.video}")
        return

    # Получаем параметры видео для правильной настройки записи и логики
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3. Выбор зоны (ROI) для одного столика (ручной выбор мышкой) 
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось прочитать первый кадр.")
        return

    print("Выберите столик мышкой и нажмите ENTER. Для отмены нажмите 'c'.")
    roi = cv2.selectROI("Выберите зону столика", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Выберите зону столика")
    rx, ry, rw, rh = roi

    # Защита: если зона не выделена (rw и rh равны 0)
    if rw == 0 or rh == 0:
        print("Зона столика не была выбрана. Завершение работы скрипта.")
        return

    # 4. Настройка записи итогового видео
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    # 5. Переменные логики и защиты от дребезга (очень важная часть для модели нано, как выяснилось в ходе проекта)
    events = []
    is_occupied = False 
    state_counter = 0
    CONFIRM_FRAMES = int(fps * 4) # Ровно 4 секунды задержки для подтверждения события

    print("Начало обработки видео... (нажмите 'q' на английской раскладке для досрочного завершения)")
    frame_count = 0

    # 6. Основной цикл покадровой обработки
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
            
        frame_count += 1
        current_time_sec = frame_count / fps

        # Трекинг: порог conf=0.15 (помогает видеть спины/сливающиеся цвета)
        # (Менял увереноость и пришёл к тому что 0.15 - оптимальный баланс между пропусками и ложными срабатываниями для модели нано в нашем случае)
        results = model.track(frame, persist=True, verbose=False, classes=[0], conf=0.15)
        
        person_in_zone = False
        
        # Проверяем, есть ли люди в кадре
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Точка для проверки - ЦЕНТР рамки человека (идеально для сидящих)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Если хотя бы один человек попал центром в зону стола
                if rx < cx < rx + rw and ry < cy < ry + rh:
                    person_in_zone = True
                    break

        # Логика смены состояний (Гистерезис)
        if person_in_zone != is_occupied:
            state_counter += 1
            if state_counter >= CONFIRM_FRAMES:
                is_occupied = person_in_zone # Официально меняем статус стола
                state_counter = 0 # Обнуляем счетчик "сомнений"
                
                event_type = 'approach' if is_occupied else 'departure'
                events.append({'event': event_type, 'timestamp': round(current_time_sec, 2)})
                print(f"[{current_time_sec:.2f}s] Зафиксировано событие: {event_type}")
        else:
            state_counter = 0 # Состояние подтверждается, сбрасываем счетчик

        # 7. Визуализация на кадре
        color = (0, 0, 255) if is_occupied else (0, 255, 0) # Красный/Зеленый
        status_text = "Occupied" if is_occupied else "Empty"
        
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.putText(frame, status_text, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        out.write(frame) # Записываем кадр в output.mp4
        
        # Показываем видео
        cv2.imshow("Processing Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Обработка прервана пользователем. Завершаем...")
            break

    # 8. Корректное закрытие файлов (чтобы видео не "билось")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nВидео успешно сохранено как output.mp4")

    # 9. Базовая аналитика (Pandas)
    if events:
        df = pd.DataFrame(events)
        print("\n--- Итоговый лог событий ---")
        print(df.to_string())
        
        delays = []
        last_dep = None
        
        # Ищем циклы: когда стол освободился -> когда подошел следующий
        for _, row in df.iterrows():
            if row['event'] == 'departure':
                last_dep = row['timestamp']
            elif row['event'] == 'approach' and last_dep is not None:
                delays.append(row['timestamp'] - last_dep)
                last_dep = None # Сброс после нахождения пары
        
        if delays:
            avg_delay = sum(delays) / len(delays)
            print("-" * 40)
            print(f"СРЕДНЕЕ ВРЕМЯ ЗАДЕРЖКИ (ОЖИДАНИЯ): {avg_delay:.2f} сек.")
            print("-" * 40)
        else:
            print("Недостаточно полных циклов (уход -> подход) для расчета среднего времени.")
        
        df.to_csv("results.csv", index=False)
        print("Таблица событий сохранена в файл results.csv")
    else:
        print("За время видео событий не обнаружено.")

if __name__ == "__main__":
    main()