import base64
import io

from PIL import Image
from rembg import remove


def handler(job):
    try:
        job_input = job.get("input", {})

        image_base64 = job_input.get("imageBase64")

        if not image_base64:
            return {
                "success": False,
                "error": "imageBase64 is required"
            }

        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)

        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        output = remove(input_image)

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        cleaned = base64.b64encode(buffer.getvalue()).decode()

        return {
            "success": True,
            "image": f"data:image/png;base64,{cleaned}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }