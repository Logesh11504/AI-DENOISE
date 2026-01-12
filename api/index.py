from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false' # Not using GPU anyway
os.environ['OMP_NUM_THREADS'] = '1' # Limit CPU threads to save memory
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import base64
import random
from PIL import Image
import urllib.request
import requests # Add this to requirements.txt
from pathlib import Path
from contextlib import asynccontextmanager
import gc

# This defines ROOT_DIR as the directory where index.py lives
ROOT_DIR = Path(__file__).parent.parent

# Define a global variable for the model
model = None

# --- PATHS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "fingerprint_unet_final.h5")
DRIVE_ID = "18bLiNQd-yuTAxPT9bsZtdt2lV3qESCaW"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        download_large_file_from_drive(DRIVE_ID, MODEL_PATH)
    
    model = load_model(MODEL_PATH, compile=False)
    
    # Manually trigger garbage collection to free up RAM
    gc.collect() 
    print("Model loaded and memory cleared!")
    yield

app = FastAPI(lifespan=lifespan)

# --- DOWNLOADER THAT BYPASSES VIRUS WARNING ---
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
    noisy_image = image.copy()
    noisy_image = noisy_image.squeeze() * 255.0 

    noise = np.random.normal(0, 25, noisy_image.shape)
    noisy_image = noisy_image + noise
    noisy_image = np.clip(noisy_image, 0, 255)

    # Add thick white/black grains (same as your streamlit code)
    for _ in range(2):
        x, y = random.randint(0, 200), random.randint(0, 200)
        w, h = random.randint(20, 50), random.randint(20, 50)
        noisy_image[y:y+h, x:x+w] = 255
    for _ in range(2):
        x, y = random.randint(0, 200), random.randint(0, 200)
        w, h = random.randint(20, 50), random.randint(20, 50)
        noisy_image[y:y+h, x:x+w] = 0

    # Add dots
    num_dots = random.randint(4, 6)
    for _ in range(num_dots):
        dot_color = 255 if random.random() > 0.5 else 0
        size = 3 if dot_color == 255 else random.randint(4, 6)
        x, y = random.randint(0, 250), random.randint(0, 250)
        noisy_image[y:y+size, x:x+size] = dot_color

    noisy_image = noisy_image / 255.0
    return noisy_image[..., np.newaxis].astype(np.float32)

def encode_image_to_base64(image_np):
    img_uint8 = (image_np.squeeze() * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img_uint8)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"

def match_fingerprints(img1_np, img2_np):
    i1 = (img1_np.squeeze() * 255).astype(np.uint8)
    i2 = (img2_np.squeeze() * 255).astype(np.uint8)
    _, i1_bin = cv2.threshold(i1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, i2_bin = cv2.threshold(i2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(i1_bin, None)
    kp2, des2 = orb.detectAndCompute(i2_bin, None)

    if des1 is None or des2 is None: return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len([m for m in matches if m.distance < 50])

@app.post("/api/predict")
async def predict(file: UploadFile = File(None)): # Make file optional
    try:
        if file is None:
            default_path = os.path.join(CURRENT_DIR, "noise.png")
            if not os.path.exists(default_path):
                return JSONResponse(status_code=400, content={"error": "No file uploaded and no default image found."})
            img = Image.open(default_path).convert('L').resize((256, 256))
        else:
            # --- LOAD USER UPLOAD ---
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('L').resize((256, 256))

        original_np = np.array(img).astype(np.float32)[..., np.newaxis] / 255.0
        noisy_np = add_custom_noise_and_missing_thick_grains(original_np)
        
        input_tensor = np.expand_dims(noisy_np, axis=0)
        denoised_np = np.clip(model.predict(input_tensor)[0], 0, 1)

        score = match_fingerprints(original_np, denoised_np)

        return {
            "original": encode_image_to_base64(original_np),
            "noisy": encode_image_to_base64(noisy_np),
            "denoised": encode_image_to_base64(denoised_np),
            "match_score": score
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Static files
app.mount("/public", StaticFiles(directory=os.path.join(ROOT_DIR, "public")), name="public")
@app.get("/")
async def read_index(): return FileResponse(os.path.join(ROOT_DIR, 'index.html'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)