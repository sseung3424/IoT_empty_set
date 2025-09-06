import cv2
from picamera2 import Picamera2

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()

# Set resolution = 640x480, format = RGB888
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)

# Start camera streaming
picam2.start()

while True:
    # Capture one frame as numpy array
    frame = picam2.capture_array()

    # Show the frame in OpenCV window
    cv2.imshow("rpicam", frame)

    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Stop the camera and close all windows
picam2.stop()
cv2.destroyAllWindows()

