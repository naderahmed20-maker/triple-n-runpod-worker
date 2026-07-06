import runpod
import requests
import base64
import io
from PIL import Image
from rembg import remove


def handler(job):
    job_input = job.get("input", {})

    image_url = job_input.get("image") or job_input.get("imageUrl")

    if not image_url:
        return {"error": "image is required"}

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    input_image = Image.open(io.BytesIO(response.content)).convert("RGBA")

    output = remove(input_image)

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image": f"data:image/png;base64,{encoded}"
    }


runpod.serverless.start({
    "handler": handler
})