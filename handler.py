from PIL import Image
from rembg import remove
import base64
import io


def handler(job):
    job_input = job["input"]

    image_b64 = job_input["image"]

    image_bytes = base64.b64decode(image_b64)

    input_image = Image.open(io.BytesIO(image_bytes))

    output = remove(input_image)

    buffer = io.BytesIO()
    output.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image": encoded
    }