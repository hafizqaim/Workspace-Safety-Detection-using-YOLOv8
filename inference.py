import cv2
from ultralytics import YOLO

# Load your custom-trained model
model = YOLO('best.pt')

# To use a video file instead of a webcam, uncomment the line below
# and replace 'path/to/your/video.mp4' with your video's path.
cap = cv2.VideoCapture('test2.mp4')

# To use the webcam
#cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    # We only care about detecting helmets (class 12) and vests (class 16)
    results = model(frame, classes=[12, 16])

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Workplace Safety Monitoring', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()