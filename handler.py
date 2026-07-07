import base64
import io
from PIL import Image
from rembg import remove

MAX_IMAGE_SIZE_MB = 12
CANVAS_SIZE = 1024
ITEM_MAX_SIZE = 860


def get_image_data(job_input):
    return (
        job_input.get("imageBase64")
        or job_input.get("image")
        or job_input.get("base64")
    )


def strip_data_url(image_data):
    if "," in image_data:
        return image_data.split(",", 1)[1]
    return image_data


def crop_transparent(image):
    bbox = image.getbbox()
    if not bbox:
        return image
    return image.crop(bbox)


def fit_on_canvas(image):
    image = crop_transparent(image)

    w, h = image.size
    scale = min(ITEM_MAX_SIZE / w, ITEM_MAX_SIZE / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = image.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    x = (CANVAS_SIZE - new_w) // 2
    y = (CANVAS_SIZE - new_h) // 2
    canvas.paste(image, (x, y), image)

    return canvas


def handler(job):
    try:
        job_input = job.get("input", {})
        image_data = get_image_data(job_input)

        if not image_data:
            return {
                "success": False,
                "error": "imageBase64 is required"
            }

        image_data = strip_data_url(image_data)
        image_bytes = base64.b64decode(image_data)

        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return {
                "success": False,
                "error": "image too large"
            }

        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        removed = remove(input_image)
        cleaned = fit_on_canvas(removed)

        buffer = io.BytesIO()
        cleaned.save(buffer, format="PNG", optimize=True)

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