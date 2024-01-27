from pydantic import BaseModel
import torch
import ssl
from PIL import Image
from fastapi import FastAPI, Query, File, UploadFile, HTTPException, Body
from fastapi.responses import Response, JSONResponse
import io
import json
import requests
from typing import List
import http.client
import json
from fastapi.middleware.cors import CORSMiddleware
import base64
from backup import app


def get_yolov5():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")
    model.conf = 0.01
    # model.conf = 0.5
    return model


def get_image_from_bytes(binary_image, max_size=1024):
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    width, height = input_image.size
    resize_factor = min(max_size / width, max_size / height)
    resized_image = input_image.resize(
        (
            int(input_image.width * resize_factor),
            int(input_image.height * resize_factor),
        )
    )
    return resized_image


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


def get_image_from_bytes(image_bytes):
    # Function to convert image bytes to a PIL Image
    return Image.open(io.BytesIO(image_bytes))


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
                        # "result": detect_res,
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

    # conn = http.client.HTTPSConnection("google.serper.dev")
    # payload = json.dumps({"q": "AED " + kw, "gl": "th", "hl": "th"})
    # headers = {
    #     "X-API-KEY": "fbdd42c99b3ddf23c1fe5677b2c1414c0eff0984",
    #     "Content-Type": "application/json",
    # }

    # try:
    #     conn.request("POST", "/images", payload, headers)
    #     res = conn.getresponse()
    #     data = res.read().decode("utf-8")

    # except Exception as e:
    #     # Handle exceptions, log the error, and return an appropriate response
    #     return JSONResponse(content={"error": str(e)}, status_code=500)
    # finally:
    #     conn.close()

    # response_data = json.loads(data)
    # images = response_data.get("images", [])

    # # Process each image using a for loop
    # processed_images = []

    # for image in images:
    #     try:
    #         # Get the image from the URL
    #         input_image = get_image_from_url(image["imageUrl"])

    #         # Perform object detection
    #         results = model(input_image)
    #         results.render()  # updates results.imgs with boxes and labels

    #         if not results.ims:
    #             print("No images available after rendering.")

    #         img_array = results.ims[0]
    #         # Convert NumPy array to PIL Image
    #         img = Image.fromarray(img_array)

    #         # Convert the image to base64
    #         buffered = io.BytesIO()
    #         img.save(buffered, format="JPEG")
    #         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    #         # Extract and convert the detection results to JSON
    #         detect_res = results.pandas().xyxy[0].to_json(orient="records")
    #         detect_res = json.loads(detect_res)

    #         processed_images.append(
    #             {
    #                 "url": image["imageUrl"],
    #                 "result": detect_res,
    #                 "title": image["title"],
    #                 "source": image["source"],
    #                 "domain": image["domain"],
    #                 "link": image["link"],
    #                 "google_url": image["googleUrl"],
    #                 "position": image["position"],
    #                 "base64": img_base64,
    #             }
    #         )
    #         # print(processed_images)
    #     except Exception as e:
    #         print(e)

    # # Modify the response by removing "searchParameters"
    # # response = JSONResponse(content={"images": images})
    # response = JSONResponse(
    #     content={
    #         "processed_images": processed_images,
    #         "max": num,
    #         "keywords": kw,
    #     }
    # )

    # return response


# @app.post("/search")
# async def search_by_kw(request_data: KeywordRequest):
#     kw = request_data.kw
#     num = request_data.num

#     conn = http.client.HTTPSConnection("google.serper.dev")
#     payload = json.dumps({"q": "AED " + kw, "gl": "th", "hl": "th"})
#     headers = {
#         "X-API-KEY": "fbdd42c99b3ddf23c1fe5677b2c1414c0eff0984",
#         "Content-Type": "application/json",
#     }

#     try:
#         conn.request("POST", "/images", payload, headers)
#         res = conn.getresponse()
#         data = res.read().decode("utf-8")

#     except Exception as e:
#         # Handle exceptions, log the error, and return an appropriate response
#         return JSONResponse(content={"error": str(e)}, status_code=500)
#     finally:
#         conn.close()

#     response_data = json.loads(data)
#     images = response_data.get("images", [])

#     # Process each image using a for loop
#     processed_images = []

#     for image in images:
#         try:
#             # Get the image from the URL
#             input_image = get_image_from_url(image["imageUrl"])

#             # Perform object detection
#             results = model(input_image)
#             results.render()  # updates results.imgs with boxes and labels

#             if not results.ims:
#                 print("No images available after rendering.")

#             img_array = results.ims[0]
#             # Convert NumPy array to PIL Image
#             img = Image.fromarray(img_array)

#             # Convert the image to base64
#             buffered = io.BytesIO()
#             img.save(buffered, format="JPEG")
#             img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#             # Extract and convert the detection results to JSON
#             detect_res = results.pandas().xyxy[0].to_json(orient="records")
#             detect_res = json.loads(detect_res)

#             processed_images.append(
#                 {
#                     "url": image["imageUrl"],
#                     "result": detect_res,
#                     "title": image["title"],
#                     "source": image["source"],
#                     "domain": image["domain"],
#                     "link": image["link"],
#                     "google_url": image["googleUrl"],
#                     "position": image["position"],
#                     "base64": img_base64,
#                 }
#             )
#             # print(processed_images)
#         except Exception as e:
#             print(e)

#     # Modify the response by removing "searchParameters"
#     # response = JSONResponse(content={"images": images})
#     response = JSONResponse(
#         content={
#             "processed_images": processed_images,
#             "max": num,
#             "keywords": kw,
#         }
#     )

#     return response


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = (
        results.pandas().xyxy[0].to_json(orient="records")
    )  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-jsons")
async def detect_food_return_json_result(urls: List[str] = Body(...)):
    # Check if the provided URLs list is not empty
    if not urls:
        raise HTTPException(status_code=400, detail="List of URLs is empty")

    # Initialize an empty list to store detection results for each image
    all_results = []

    # Loop through each URL in the list
    for url in urls:
        try:
            # Get the image from the URL
            input_image = get_image_from_url(url)

            # Perform object detection
            results = model(input_image)

            # Extract and convert the detection results to JSON
            detect_res = results.pandas().xyxy[0].to_json(orient="records")
            detect_res = json.loads(detect_res)

            # Append the detection results to the list
            all_results.append({"url": url, "result": detect_res})
        except Exception as e:
            # Handle errors for individual images if needed
            all_results.append({"url": url, "error": str(e)})

    # Return the list of detection results
    return {"results": all_results}


def get_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")


@app.post("/url-to-img")
async def detect_food_return_base64_img(url: str):
    try:
        input_image = get_image_from_url(url)
        results = model(input_image)
        results.render()  # updates results.imgs with boxes and labels

        if not results.ims:
            raise HTTPException(
                status_code=500, detail="No images available after rendering."
            )

        img_array = results.ims[0]  # Assuming you want the first image from results

        # Convert NumPy array to PIL Image
        img = Image.fromarray(img_array)

        # Convert the image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        response_headers = {
            "Content-Disposition": "inline; filename=detection_result.jpg",  # Optional
            "Cache-Control": "max-age=3600",  # Optional, cache for 1 hour
        }

        return Response(
            content=img_base64, media_type="text/plain", headers=response_headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
