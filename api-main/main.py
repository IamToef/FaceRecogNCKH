from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from app.face_recognition import process_image
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

@app.get("/", response_class=RedirectResponse)
async def root():
    return "/docs"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    result_image, info = process_image(np.array(img))

    result_image_pil = Image.fromarray(result_image)
    result_image_pil.save("app/static/result_image.jpg")

    # Return HTML with image and information
    return HTMLResponse(content=f"""
    <html>
        <body>
            <h2>Information:</h2>
            <p>{info}</p>
            <img src="/static/result_image.jpg" />
        </body>
    </html>
    """)

# Helper function to generate frames from the webcam
def gen_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()  # Capture frame-by-frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Frame as multipart

    cap.release()  # Release the webcam when done

@app.get("/camera_feed/")
async def camera_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# HTML page to show the camera feed
@app.get("/camera/")
async def camera_page():
    return HTMLResponse(content="""
    <html>
        <body>
            <h1>Webcam Stream</h1>
            <img src="/camera_feed" width="640" height="480">
        </body>
    </html>
    """)
