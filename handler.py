import base64
import io
import requests
from PIL import Image
from rembg import remove

MAX_IMAGE_SIZE_MB = 12


def handler(job):
    try:
        job_input = job.get("input", {})
        image_url = job_input.get("image") or job_input.get("imageUrl")

        if not image_url:
            return {"success": False, "error": "image is required"}

        response = requests.get(image_url, timeout=45)
        response.raise_for_status()

        size_mb = len(response.content) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return {"success": False, "error": "image too large"}

        input_image = Image.open(io.BytesIO(response.content)).convert("RGBA")
        output = remove(input_image)

        buffer = io.BytesIO()
        output.save(buffer, format="PNG", optimize=True)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "image": f"data:image/png;base64,{encoded}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }