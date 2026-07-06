import base64
import io
from PIL import Image
from rembg import remove

MAX_IMAGE_SIZE_MB = 12


def handler(job):
    try:
        job_input = job.get("input", {})
        image_base64 = job_input.get("imageBase64")

        if not image_base64:
            return {"success": False, "error": "imageBase64 is required"}

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return {"success": False, "error": "image too large"}

        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
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