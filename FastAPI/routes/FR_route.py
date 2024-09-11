import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import File, UploadFile
from fastapi import APIRouter
from schemas.FR_schema import FRResponse
from config.FR_cfg import ModelConfig
from models.FR_predictor import Predictor
import cv2
from fastapi.responses import StreamingResponse

router = APIRouter()

# Initialize the predictor as before
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight=ModelConfig.MODEL_WEIGHT,
    device=ModelConfig.DEVICE
)

# Existing file upload prediction endpoint
@router.post("/predict")
async def predict(file_upload: UploadFile = File(...)):
    response = await predictor.predict(file_upload.file)
    return FRResponse(**response)

# New camera stream endpoint
@router.get("/camera")
async def open_camera():
    # Open a connection to the webcam (device 0 is default)
    camera = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            success, frame = camera.read()  # Read a frame from the webcam
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")