from pydantic import BaseModel
import torch
import ssl
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import io
import json
import requests
from typing import List
import http.client
import json
from fastapi.middleware.cors import CORSMiddleware
import base64
from backup import app
import uvicorn


def get_yolov5():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
    model.conf = 0.01
    # model.conf = 0.5
    return model


model = get_yolov5()

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_image_from_url(url):
    try:
        # Send a GET request to the URL to fetch the image
        response = requests.get(url, verify=False)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(io.BytesIO(response.content))

            return image
        else:
            # Raise an exception if the request was not successful
            raise Exception(
                f"Failed to fetch image from URL: {url}, Status code: {response.status_code}"
            )
    except Exception as e:
        # Handle any other exceptions that may occur
        raise Exception(f"Error fetching image from URL: {url}, {str(e)}")


class KeywordRequest(BaseModel):
    kw: List[str]
    num: int


@app.get("/")
async def root():
    return {"message": "ok"}


@app.post("/search")
async def search_by_kw(request_data: KeywordRequest):
    kw = request_data.kw
    num = request_data.num
    all_data = []

    for keyword in kw:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": keyword, "gl": "th", "hl": "th"})
        headers = {
            "X-API-KEY": "fbdd42c99b3ddf23c1fe5677b2c1414c0eff0984",
            "Content-Type": "application/json",
        }

        try:
            conn.request("POST", "/images", payload, headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
        finally:
            conn.close()

        response_data = json.loads(data)
        images = response_data.get("images", [])

        processed_images = []

        for image in images:
            try:
                # Get the image from the URL
                input_image = get_image_from_url(image["imageUrl"])

                # Perform object detection
                results = model(input_image)
                results.render()  # updates results.imgs with boxes and labels

                if not results.ims:
                    print("No images available after rendering.")

                img_array = results.ims[0]
                # Convert NumPy array to PIL Image
                img = Image.fromarray(img_array)

                # Convert the image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Extract and convert the detection results to JSON
                detect_res = results.pandas().xyxy[0].to_json(orient="records")
                detect_res = json.loads(detect_res)

                detect_res_df = results.pandas().xyxy[0]
                first_row_json = detect_res_df.iloc[0].to_json(orient="records")
                first_row_json = json.loads(first_row_json)

                processed_images.append(
                    {
                        "url": image["imageUrl"],
                        "confidence": first_row_json[4],
                        "title": image["title"],
                        "source": image["source"],
                        "domain": image["domain"],
                        "link": image["link"],
                        "google_url": image["googleUrl"],
                        "position": image["position"],
                        "base64": img_base64,
                        "keyword": keyword,
                    }
                )
                all_data.append(processed_images)
            except Exception as e:
                print(e)

    response = JSONResponse(
        content={
            "all_data": all_data,
            "max": num,
            "keywords": kw,
        }
    )
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
