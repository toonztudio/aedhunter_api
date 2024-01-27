import torch
import ssl
from PIL import Image
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import JSONResponse
import requests
from io import BytesIO

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

# model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")


@app.get("/")
async def root():
    return {"message": "hello"}


@app.get("/predicts")
async def predicts(
    urls: list = Query(..., description="List of URLs of the images to predict")
):
    detected_objects_list = []

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return JSONResponse(
                content={"error": f"Failed to download image from {url}: {e}"},
                status_code=400,
            )

        image = Image.open(BytesIO(response.content))
        results = model(image)

        # Process results for each image
        detected_objects = []
        for label, score, bbox in zip(
            results.names,
            results.xyxy[0][:, -1].tolist(),
            results.xyxy[0][:, :-1].tolist(),
        ):
            detected_objects.append(
                {
                    "class": label,
                    "confidence": score,
                    "bbox": {
                        "xmin": bbox[0],
                        "ymin": bbox[1],
                        "xmax": bbox[2],
                        "ymax": bbox[3],
                    },
                }
            )

        detected_objects_list.append({"url": url, "detected_objects": detected_objects})

    return JSONResponse(content={"results": detected_objects_list})


@app.get("/predict")
async def predict_url(url: str = Query(..., description="URL of the image to predict")):
    # Download the image from the URL
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            content={"error": f"Failed to download image: {e}"}, status_code=400
        )

    # Open the downloaded image using PIL
    image = Image.open(BytesIO(response.content))

    # Perform prediction
    results = model(image)

    # Process results as needed
    # For example, you can return the detected objects and their confidence scores
    detected_objects = [
        {"class": label, "confidence": score}
        for label, score in zip(results.names, results.xyxy[0][:, -1].tolist())
    ]

    return JSONResponse(content={"detected_objects": detected_objects})


@app.post("/file")
async def predict_image(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as image_file:
        image_file.write(file.file.read())

    results = model(file_path)
    print(results)

    detected_objects = [
        {"class": label, "confidence": score}
        for label, score in zip(results.names, results.xyxy[0][:, -1].tolist())
    ]

    return JSONResponse(content={"detected_objects": detected_objects})
