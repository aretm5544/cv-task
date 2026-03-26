import cv2
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Dodo Table Detection (Traditional CV)")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Ошибка открытия видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # выбор зоны столика
    ret, first_frame = cap.read()
    if not ret: return
    roi = cv2.selectROI("Select Table Zone", first_frame, fromCenter=False)
    cv2.destroyWindow("Select Table Zone")
    rx, ry, rw, rh = roi

    # вычитаю фон (MOG2)
    # history=500 означает, что он медленно "забывает" старый фон
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    out = cv2.VideoWriter('output_traditional.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    events = []
    is_occupied = False
    state_counter = 0
    CONFIRM_FRAMES = int(fps * 3) # задержка 3 секунды для надежности (как и в первом варинте с моделью)

    print("Обработка запущена...") #для красоты

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # применяем маску фона
        fgmask = fgbg.apply(frame)

        # фильтрация шума (убираем тени и мелкие точки)
        # оставляем только чисто белые объекты (значение 255)
        _, thresh = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        
        # размываем и расширяем, чтобы объединить части тела человека в один контур
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # поиск контуров только внутри ROI для оптимизации
        roi_mask = np.zeros_like(thresh)
        roi_mask[ry:ry+rh, rx:rx+rw] = thresh[ry:ry+rh, rx:rx+rw]
        
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # проверяем, есть ли значимое движение в зоне столика
        motion_in_zone = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 1500: # минимальный размер "пятна" (вроде бы самый оптимальный)
                motion_in_zone = True
                break

        # логика подтверждения состояния (Гистерезис)
        if motion_in_zone != is_occupied:
            state_counter += 1
            if state_counter >= CONFIRM_FRAMES:
                is_occupied = motion_in_zone
                state_counter = 0
                event_type = 'approach' if is_occupied else 'departure'
                events.append({'event': event_type, 'timestamp': round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 2)})
        else:
            state_counter = 0

        # визуализация
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 2)
        cv2.putText(frame, "Occupied" if is_occupied else "Empty", (rx, ry-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # также вывел саму маску с бинаризацией чтобы увидеть сами силуэты
        cv2.imshow('Mask', thresh) 
        
        out.write(frame)
        cv2.imshow('Dodo Traditional Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # аналитика 
    if events:
        df = pd.DataFrame(events)
        df.to_csv("results_traditional.csv", index=False)
        print("\nРезультаты сохранены. Среднее время ожидания рассчитано в CSV.")
    else:
        print("Событий не зафиксировано.")

if __name__ == "__main__":
    main()