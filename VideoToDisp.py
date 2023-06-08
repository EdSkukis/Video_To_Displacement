import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка видеофайла
video_path = 'Video\\Beam_3.mp4'
cap = cv2.VideoCapture(video_path)

# Чтение первого кадра
ret, frame = cap.read()
if not ret:
    print("Не удалось прочитать видео")
    exit()

# Выбор области интереса (ROI) для отслеживания
# Установите координаты ROI, соответствующие балке на первом кадре
roi = cv2.selectROI(frame)
# print rectangle points of selected roi
print(roi)
# Crop selected roi from raw image
# roi_cropped = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
roi_x = roi[0]  # X-координата верхнего левого угла ROI
roi_y = roi[1]  # Y-координата верхнего левого угла ROI
roi_width = roi[2]   # Ширина ROI
roi_height = roi[3]   # Высота ROI

# Создание трекера для отслеживания объекта в ROI
# tracker = cv2.TrackerKCF_create()

# Создание трекера для отслеживания объекта в ROI

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
# dont have 0, 3
tracker_type = tracker_types[7]
if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

# tracker = cv2.TrackerKCF_create()
bbox = (roi_x, roi_y, roi_width, roi_height)
tracker.init(frame, bbox)

prev_center = None  # Предыдущий центр ограничивающего прямоугольника

displacement_x,  displacement_y = 0, 0
displ_arrey = np.array([[], []])

while True:
    # Чтение следующего кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Обновление трекера
    success, bbox = tracker.update(frame)

    if success:
        # Отрисовка прямоугольника вокруг отслеживаемого объекта
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + 2, y + 2), (0, 255, 0), 2)
        cv2.putText(frame, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Вычисление центра ограничивающего прямоугольника
        center = (int(x + w / 2), int(y + h / 2))

        if prev_center is not None:
            # Вычисление смещения между текущим и предыдущим кадром
            displacement = (center[0] - prev_center[0], center[1] - prev_center[1])
            displacement_X = displacement[0] + displacement_x
            displacement_y = displacement[1] + displacement_y
            displ_arrey = np.append(displ_arrey, [[displacement_X], [displacement_y]], axis=1)
            # print(displacement_X)
            # print("Смещение (dx, dy):", displacement)

        prev_center = center

    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Вывод текущего кадра с отрисованным объектом
    cv2.imshow("Video", frame)
    cv2.waitKey(1)

    # # Ожидание нажатия клавиши 'q' для выхода
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()


plt.plot(displ_arrey[0])
plt.plot(displ_arrey[1])
plt.ylabel('Displacement')
plt.show()

