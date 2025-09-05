import cv2
cap = cv2.VideoCapture(0)             # 레거시 V4L2 장치
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ok, frame = cap.read()
    if not ok:
        print("캡처 실패"); break
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
