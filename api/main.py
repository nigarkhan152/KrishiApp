from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory="Frontend/images"), name="images")

app.mount("/static", StaticFiles(directory="static"), name="static")
MODEL = tf.keras.models.load_model("models/potato_disease_models.h5")

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage", shuffle=False
)

CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight","Potato___healthy"]
# Preprocess image
def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((256, 256))  
        image_array = np.array(image) / 255.0
        return image_array
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

# Predict route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Empty file received.")

    image = read_file_as_image(image_data)
    image_batch = np.expand_dims(image, axis=0)

    prediction = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": confidence
    })
@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join("Frontend", "first.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/disease", response_class=HTMLResponse)
async def disease_page():
    with open(os.path.join("static", "index.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/about", response_class=HTMLResponse)
async def about_page():
    with open(os.path.join("Frontend", "about.html"), encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
