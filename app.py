import cv2
import uuid
import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

# --- Initialization ---
app = FastAPI()

# Load the pre-trained YOLOv8 model
# Ensure 'best.pt' is in the same directory as this script
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model can't be loaded, as the app is useless without it.
    exit()

# A dictionary to store the paths of uploaded videos, using a session ID as the key
video_sessions = {}

# Create a directory to store temporary video uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Frontend Endpoint ---

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main HTML page."""
    try:
        with open("templates/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found</h1><p>Please make sure the 'templates' folder and 'index.html' file exist.</p>"


# --- Webcam Streaming ---

def generate_webcam_frames():
    """Generates annotated frames from the webcam."""
    camera = cv2.VideoCapture(0)  # 0 is the default webcam
    if not camera.isOpened():
        print("Error: Could not start webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run YOLOv8 inference on the frame, filtering for helmets (12) and vests (16)
            results = model(frame, classes=[12, 16], verbose=False)
            annotated_frame = results[0].plot()

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

@app.get("/video_feed/webcam")
def video_feed_webcam():
    """Streams video from the webcam."""
    # Use StreamingResponse for generator functions
    return StreamingResponse(generate_webcam_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# --- Video Upload and Processing ---

@app.post("/upload_video")
async def upload_video(video_file: UploadFile = File(...)):
    """Handles video file uploads."""
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{video_file.filename}")

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
    
    # Store the path for the streaming endpoint
    video_sessions[session_id] = file_path
    
    return JSONResponse(content={"session_id": session_id})

def generate_upload_frames(session_id: str):
    """Generates annotated frames from an uploaded video."""
    file_path = video_sessions.get(session_id)
    if not file_path or not os.path.exists(file_path):
        print(f"Error: Video for session {session_id} not found.")
        return

    video = cv2.VideoCapture(file_path)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        else:
            # Run YOLOv8 inference
            results = model(frame, classes=[12, 16], verbose=False)
            annotated_frame = results[0].plot()
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    video.release()
    # Clean up: remove the video file and session entry after streaming
    if os.path.exists(file_path):
        os.remove(file_path)
    if session_id in video_sessions:
        del video_sessions[session_id]

@app.get("/video_feed/upload")
def video_feed_upload(session_id: str):
    """Streams video from an uploaded file based on the session ID."""
    # Use StreamingResponse for generator functions
    return StreamingResponse(generate_upload_frames(session_id), media_type="multipart/x-mixed-replace; boundary=frame")
