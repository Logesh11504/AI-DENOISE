from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import io
import base64
import random
from PIL import Image
import requests
from pathlib import Path
from contextlib import asynccontextmanager
import gc
# Use tflite_runtime instead of full tensorflow to save ~400MB RAM
import tflite_runtime.interpreter as tflite

# --- PATHS & CONFIG ---
ROOT_DIR = Path(__file__).parent.parent
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model.tflite")
DRIVE_ID = "179XRJXXPYi72f2Za8rO74n_mqIkgRCYE" # <--- UPDATE THIS

# Global TFLite variables
interpreter = None
input_details = None
output_details = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_details, output_details
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading TFLite model...")
        download_large_file_from_drive(DRIVE_ID, MODEL_PATH)
    
    # Initialize TFLite Interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    gc.collect() 
    print("TFLite Model loaded successfully!")
    yield

app = FastAPI(lifespan=lifespan)

def download_large_file_from_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk: f.write(chunk)

def add_custom_noise_and_missing_thick_grains(image):
    noisy_image = image.copy().squeeze() * 255.0 
    noise = np.random.normal(0, 25, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255)

    # Thick grains
    for _ in range(2):
        x, y = random.randint(0, 200), random.randint(0, 200)
        w, h = random.randint(20, 50), random.randint(20, 50)
        noisy_image[y:y+h, x:x+w] = 255
        x, y = random.randint(0, 200), random.randint(0, 200)
        noisy_image[y:y+h, x:x+w] = 0

    # Dots
    for _ in range(random.randint(4, 6)):
        dot_color = 255 if random.random() > 0.5 else 0
        size = 3 if dot_color == 255 else random.randint(4, 6)
        x, y = random.randint(0, 250), random.randint(0, 250)
        noisy_image[y:y+size, x:x+size] = dot_color

    return (noisy_image / 255.0)[..., np.newaxis].astype(np.float32)

def encode_image_to_base64(image_np):
    img_uint8 = (image_np.squeeze() * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img_uint8)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def match_fingerprints(img1_np, img2_np):
    def prep(img):
        i = (img.squeeze() * 255).astype(np.uint8)
        return cv2.threshold(i, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    i1_bin, i2_bin = prep(img1_np), prep(img2_np)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(i1_bin, None)
    kp2, des2 = orb.detectAndCompute(i2_bin, None)

    if des1 is None or des2 is None: return 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    return len([m for m in matches if m.distance < 50])

@app.post("/api/predict")
async def predict(file: UploadFile = File(None)):
    try:
        if file is None:
            default_path = os.path.join(CURRENT_DIR, "noise.png")
            img = Image.open(default_path).convert('L').resize((256, 256))
        else:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('L').resize((256, 256))

        original_np = np.array(img).astype(np.float32)[..., np.newaxis] / 255.0
        noisy_np = add_custom_noise_and_missing_thick_grains(original_np)
        
        # TFLite Inference
        input_data = np.expand_dims(noisy_np, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        denoised_np = np.clip(interpreter.get_tensor(output_details[0]['index'])[0], 0, 1)

        score = match_fingerprints(original_np, denoised_np)

        return {
            "original": encode_image_to_base64(original_np),
            "noisy": encode_image_to_base64(noisy_np),
            "denoised": encode_image_to_base64(denoised_np),
            "match_score": score
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

app.mount("/public", StaticFiles(directory=os.path.join(ROOT_DIR, "public")), name="public")
@app.get("/")
async def read_index(): return FileResponse(os.path.join(ROOT_DIR, 'index.html'))