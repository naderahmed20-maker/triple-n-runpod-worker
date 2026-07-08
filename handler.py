import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

MAX_IMAGE_SIZE_MB = 12
MODEL_INPUT_SIZE = 1024
CANVAS_SIZE = 1024

CATEGORY_SIZE = {
    "Tops": 770,
    "Pants": 930,
    "Shorts": 700,
    "Shoes": 540,
    "Jackets": 860,
    "Dresses": 930,
    "Skirts": 760,
    "Accessories": 260,
}

DEFAULT_ITEM_SIZE = 850

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading BiRefNet on:", DEVICE)

model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True
)

model.to(DEVICE)
model.eval()

transform_image = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


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


def normalize_category(category):
    if not category:
        return "Tops"

    if category in ["Top", "T-Shirt", "TShirt", "Shirt", "Hoodie", "Sweater", "Polo"]:
        return "Tops"

    if category in ["Pant", "Bottom", "Bottoms", "Jeans", "Cargo", "Formal", "Joggers"]:
        return "Pants"

    if category in ["Jacket", "Coat"]:
        return "Jackets"

    if category in ["Shoe", "Sneakers", "Boots", "Loafers", "Sandals", "Heels"]:
        return "Shoes"

    if category in ["Accessory", "Watch", "Glasses", "Cap", "Bag", "Other"]:
        return "Accessories"

    return category


def get_prediction(output):
    if isinstance(output, (list, tuple)):
        pred = output[-1]
    else:
        pred = output

    if isinstance(pred, (list, tuple)):
        pred = pred[-1]

    return pred


def refine_mask(mask):
    mask_np = np.array(mask).astype(np.uint8)

    _, mask_np = cv2.threshold(mask_np, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, 8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_np = np.where(labels == largest, 255, 0).astype(np.uint8)

    mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)

    return Image.fromarray(mask_np)


def remove_background(image):
    original_size = image.size

    input_tensor = transform_image(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(input_tensor)
        pred = get_prediction(output)
        pred = torch.sigmoid(pred)
        pred = pred.squeeze().detach().cpu()

    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(original_size, Image.LANCZOS)
    mask = refine_mask(mask)

    rgba = image.convert("RGBA")
    rgba.putalpha(mask)

    return rgba


def crop_transparent(image, category):
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()

    if not bbox:
        return image

    left, top, right, bottom = bbox

    if category in ["Tops", "Jackets"]:
        top += 35
        bottom -= 25
    elif category == "Shoes":
        top += 10
        bottom -= 10
    else:
        top += 15
        bottom -= 15

    top = max(0, top)
    bottom = min(image.height, bottom)

    if bottom <= top:
        top, bottom = bbox[1], bbox[3]

    return image.crop((left, top, right, bottom))


def fit_on_canvas(image, category):
    category = normalize_category(category)
    image = crop_transparent(image, category)

    w, h = image.size

    if w == 0 or h == 0:
        return Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))

    target_size = CATEGORY_SIZE.get(category, DEFAULT_ITEM_SIZE)

    scale = min(target_size / w, target_size / h)

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
        category = normalize_category(job_input.get("category", "Tops"))

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

        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        removed = remove_background(input_image)
        cleaned = fit_on_canvas(removed, category)

        buffer = io.BytesIO()
        cleaned.save(buffer, format="PNG", compress_level=1)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "category": category,
            "image": f"data:image/png;base64,{encoded}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }