import base64
import binascii
import io
import time
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageSegmentation
from torchvision import transforms


MAX_IMAGE_SIZE_MB = 12
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

MODEL_INPUT_SIZE = 1024
CANVAS_SIZE = 1024

PRE_CROP_PREVIEW_SIZE = 512
PRE_CROP_PADDING_RATIO = 0.08

CATEGORY_SIZE = {
    "Tops": 930,
    "Pants": 980,
    "Shorts": 900,
    "Shoes": 780,
    "Heels": 780,
    "Jackets": 950,
    "Dresses": 980,
    "Skirts": 920,
    "Accessories": 700,
    "Bags": 760,
}

DEFAULT_ITEM_SIZE = 920

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

print(
    f"Loading BiRefNet on: {DEVICE} | "
    f"FP16: {USE_FP16}"
)

model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True,
)

model.to(DEVICE)
model.eval()

if USE_FP16:
    model.half()

transform_image = transforms.Compose([
    transforms.Resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def now_ms():
    return time.perf_counter() * 1000


def get_image_data(job_input):
    return (
        job_input.get("imageBase64")
        or job_input.get("image")
        or job_input.get("base64")
        or job_input.get("photo")
    )


def strip_data_url(image_data):
    value = str(image_data or "").strip()

    if "," in value:
        value = value.split(",", 1)[1]

    return "".join(value.split())


def decode_base64_image(image_data):
    encoded = strip_data_url(image_data)

    if not encoded:
        raise ValueError("imageBase64 is empty")

    missing_padding = len(encoded) % 4

    if missing_padding:
        encoded += "=" * (4 - missing_padding)

    try:
        image_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise ValueError(
            "Invalid base64 image"
        ) from error

    if not image_bytes:
        raise ValueError(
            "Decoded image is empty"
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large. Maximum is "
            f"{MAX_IMAGE_SIZE_MB} MB"
        )

    return image_bytes


def normalize_category(category):
    raw_category = str(
        category or "Tops"
    ).strip()

    key = (
        raw_category
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    mapping = {
        "top": "Tops",
        "tops": "Tops",
        "t shirt": "Tops",
        "tshirt": "Tops",
        "shirt": "Tops",
        "hoodie": "Tops",
        "sweater": "Tops",
        "polo": "Tops",

        "pant": "Pants",
        "pants": "Pants",
        "bottom": "Pants",
        "bottoms": "Pants",
        "jeans": "Pants",
        "cargo": "Pants",
        "formal": "Pants",
        "joggers": "Pants",

        "short": "Shorts",
        "shorts": "Shorts",

        "shoe": "Shoes",
        "shoes": "Shoes",
        "sneakers": "Shoes",
        "boots": "Shoes",
        "loafers": "Shoes",
        "sandals": "Shoes",

        "heel": "Heels",
        "heels": "Heels",

        "jacket": "Jackets",
        "jackets": "Jackets",
        "coat": "Jackets",

        "dress": "Dresses",
        "dresses": "Dresses",

        "skirt": "Skirts",
        "skirts": "Skirts",

        "accessory": "Accessories",
        "accessories": "Accessories",
        "watch": "Accessories",
        "glasses": "Accessories",
        "cap": "Accessories",
        "other": "Accessories",

        "bag": "Bags",
        "bags": "Bags",
    }

    normalized = mapping.get(
        key,
        raw_category,
    )

    if normalized not in CATEGORY_SIZE:
        return "Tops"

    return normalized


def get_prediction(output):
    while isinstance(output, (list, tuple)):
        output = output[-1]

    return output


def safe_plain_background_crop(image):
    """
    يعمل Crop سريع ومحافظ فقط لو الخلفية حول الحواف
    تبدو موحدة بوضوح.

    لو الكروب مش مضمون، يرجع الصورة الأصلية.
    """

    original = image.convert("RGB")

    original_width, original_height = (
        original.size
    )

    if (
        original_width < 64
        or original_height < 64
    ):
        return original, False

    preview_scale = min(
        1.0,
        PRE_CROP_PREVIEW_SIZE
        / max(
            original_width,
            original_height,
        ),
    )

    preview_width = max(
        32,
        int(
            original_width
            * preview_scale
        ),
    )

    preview_height = max(
        32,
        int(
            original_height
            * preview_scale
        ),
    )

    preview = original.resize(
        (
            preview_width,
            preview_height,
        ),
        Image.BILINEAR,
    )

    rgb = np.asarray(
        preview
    ).astype(np.int16)

    border_size = max(
        3,
        int(
            min(
                preview_width,
                preview_height,
            )
            * 0.025
        ),
    )

    border_pixels = np.concatenate([
        rgb[
            :border_size,
            :,
            :
        ].reshape(-1, 3),

        rgb[
            -border_size:,
            :,
            :
        ].reshape(-1, 3),

        rgb[
            :,
            :border_size,
            :
        ].reshape(-1, 3),

        rgb[
            :,
            -border_size:,
            :
        ].reshape(-1, 3),
    ], axis=0)

    background_color = np.median(
        border_pixels,
        axis=0,
    )

    border_distance = np.linalg.norm(
        border_pixels
        - background_color,
        axis=1,
    )

    border_p90 = float(
        np.percentile(
            border_distance,
            90,
        )
    )

    threshold = max(
        22.0,
        min(
            70.0,
            border_p90 * 2.2 + 12.0,
        ),
    )

    distance = np.linalg.norm(
        rgb - background_color,
        axis=2,
    )

    foreground_mask = (
        distance > threshold
    ).astype(np.uint8) * 255

    kernel_size = max(
        3,
        int(
            min(
                preview_width,
                preview_height,
            )
            * 0.012
        ),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        (
            kernel_size,
            kernel_size,
        ),
        dtype=np.uint8,
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    (
        number_of_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        foreground_mask,
        connectivity=8,
    )

    if number_of_labels <= 1:
        return original, False

    largest_label = (
        1
        + int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )
        )
    )

    object_area = int(
        stats[
            largest_label,
            cv2.CC_STAT_AREA,
        ]
    )

    preview_area = float(
        preview_width
        * preview_height
    )

    object_area_ratio = (
        object_area
        / preview_area
    )

    if (
        object_area_ratio < 0.015
        or object_area_ratio > 0.92
    ):
        return original, False

    x = int(
        stats[
            largest_label,
            cv2.CC_STAT_LEFT,
        ]
    )

    y = int(
        stats[
            largest_label,
            cv2.CC_STAT_TOP,
        ]
    )

    width = int(
        stats[
            largest_label,
            cv2.CC_STAT_WIDTH,
        ]
    )

    height = int(
        stats[
            largest_label,
            cv2.CC_STAT_HEIGHT,
        ]
    )

    padding_x = int(
        width
        * PRE_CROP_PADDING_RATIO
    )

    padding_y = int(
        height
        * PRE_CROP_PADDING_RATIO
    )

    left = max(
        0,
        x - padding_x,
    )

    top = max(
        0,
        y - padding_y,
    )

    right = min(
        preview_width,
        x + width + padding_x,
    )

    bottom = min(
        preview_height,
        y + height + padding_y,
    )

    cropped_area_ratio = (
        (right - left)
        * (bottom - top)
    ) / preview_area

    if (
        cropped_area_ratio > 0.90
        or cropped_area_ratio < 0.06
    ):
        return original, False

    inverse_scale_x = (
        original_width
        / preview_width
    )

    inverse_scale_y = (
        original_height
        / preview_height
    )

    crop_box = (
        max(
            0,
            int(
                left
                * inverse_scale_x
            ),
        ),

        max(
            0,
            int(
                top
                * inverse_scale_y
            ),
        ),

        min(
            original_width,
            int(
                right
                * inverse_scale_x
            ),
        ),

        min(
            original_height,
            int(
                bottom
                * inverse_scale_y
            ),
        ),
    )

    cropped = original.crop(
        crop_box
    )

    if (
        cropped.width < 32
        or cropped.height < 32
    ):
        return original, False

    return cropped, True


def refine_mask(mask):
    soft_mask = np.asarray(
        mask
    ).astype(np.uint8)

    _, binary_mask = cv2.threshold(
        soft_mask,
        110,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    binary_mask = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    (
        number_of_labels,
        labels,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    if number_of_labels > 1:
        largest_label = (
            1
            + int(
                np.argmax(
                    stats[
                        1:,
                        cv2.CC_STAT_AREA,
                    ]
                )
            )
        )

        largest_component = np.where(
            labels == largest_label,
            255,
            0,
        ).astype(np.uint8)

        largest_component = cv2.dilate(
            largest_component,
            np.ones(
                (7, 7),
                dtype=np.uint8,
            ),
            iterations=1,
        )

        soft_mask = cv2.bitwise_and(
            soft_mask,
            largest_component,
        )

    soft_mask = cv2.GaussianBlur(
        soft_mask,
        (5, 5),
        0,
    )

    return Image.fromarray(
        soft_mask
    )


def remove_background(image):
    original_size = image.size

    input_tensor = (
        transform_image(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    if USE_FP16:
        input_tensor = (
            input_tensor.half()
        )

    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
        if USE_FP16
        else nullcontext()
    )

    with (
        torch.inference_mode(),
        autocast_context,
    ):
        output = model(
            input_tensor
        )

        prediction = get_prediction(
            output
        )

        prediction = torch.sigmoid(
            prediction
        )

        prediction = (
            prediction
            .squeeze()
            .float()
            .cpu()
        )

    mask = transforms.ToPILImage()(
        prediction
    )

    mask = mask.resize(
        original_size,
        Image.LANCZOS,
    )

    mask = refine_mask(
        mask
    )

    rgba = image.convert(
        "RGBA"
    )

    rgba.putalpha(
        mask
    )

    return rgba


def crop_transparent(image):
    image = image.convert(
        "RGBA"
    )

    alpha_channel = (
        image.getchannel("A")
    )

    bounding_box = (
        alpha_channel.getbbox()
    )

    if not bounding_box:
        return image

    left, top, right, bottom = (
        bounding_box
    )

    item_width = right - left
    item_height = bottom - top

    padding_x = max(
        2,
        int(
            item_width * 0.025
        ),
    )

    padding_y = max(
        2,
        int(
            item_height * 0.025
        ),
    )

    left = max(
        0,
        left - padding_x,
    )

    top = max(
        0,
        top - padding_y,
    )

    right = min(
        image.width,
        right + padding_x,
    )

    bottom = min(
        image.height,
        bottom + padding_y,
    )

    return image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )


def fit_on_canvas(image, category):
    normalized_category = (
        normalize_category(category)
    )

    image = crop_transparent(
        image
    )

    width, height = image.size

    if width <= 0 or height <= 0:
        return Image.new(
            "RGBA",
            (
                CANVAS_SIZE,
                CANVAS_SIZE,
            ),
            (
                0,
                0,
                0,
                0,
            ),
        )

    target_size = CATEGORY_SIZE.get(
        normalized_category,
        DEFAULT_ITEM_SIZE,
    )

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = max(
        1,
        int(
            round(
                width * scale
            )
        ),
    )

    new_height = max(
        1,
        int(
            round(
                height * scale
            )
        ),
    )

    resized_image = image.resize(
        (
            new_width,
            new_height,
        ),
        Image.LANCZOS,
    )

    canvas = Image.new(
        "RGBA",
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
        ),
        (
            0,
            0,
            0,
            0,
        ),
    )

    x = (
        CANVAS_SIZE
        - new_width
    ) // 2

    y = (
        CANVAS_SIZE
        - new_height
    ) // 2

    canvas.alpha_composite(
        resized_image,
        (
            x,
            y,
        ),
    )

    return canvas


def encode_png(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=1,
    )

    image_bytes = (
        buffer.getvalue()
    )

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        f"data:image/png;base64,{encoded}",
        len(image_bytes),
    )


def handler(job):
    started_at = now_ms()

    timings = {
        "decode_ms": 0,
        "pre_crop_ms": 0,
        "inference_ms": 0,
        "canvas_ms": 0,
        "encode_ms": 0,
        "total_ms": 0,
    }

    try:
        job_input = (
            job.get("input")
            or {}
        )

        image_data = get_image_data(
            job_input
        )

        category = normalize_category(
            job_input.get(
                "category",
                "Tops",
            )
        )

        if not image_data:
            return {
                "success": False,
                "error":
                    "imageBase64 is required",
            }

        decode_started = now_ms()

        image_bytes = (
            decode_base64_image(
                image_data
            )
        )

        input_image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        input_image = (
            ImageOps.exif_transpose(
                input_image
            )
            .convert("RGB")
        )

        input_width, input_height = (
            input_image.size
        )

        timings["decode_ms"] = round(
            now_ms()
            - decode_started,
            1,
        )

        crop_started = now_ms()

        (
            model_image,
            pre_cropped,
        ) = safe_plain_background_crop(
            input_image
        )

        timings["pre_crop_ms"] = round(
            now_ms()
            - crop_started,
            1,
        )

        inference_started = now_ms()

        removed_background = (
            remove_background(
                model_image
            )
        )

        timings["inference_ms"] = round(
            now_ms()
            - inference_started,
            1,
        )

        canvas_started = now_ms()

        cleaned_image = fit_on_canvas(
            removed_background,
            category,
        )

        timings["canvas_ms"] = round(
            now_ms()
            - canvas_started,
            1,
        )

        encode_started = now_ms()

        (
            encoded_image,
            output_size_bytes,
        ) = encode_png(
            cleaned_image
        )

        timings["encode_ms"] = round(
            now_ms()
            - encode_started,
            1,
        )

        timings["total_ms"] = round(
            now_ms()
            - started_at,
            1,
        )

        print(
            "JOB COMPLETE | "
            f"category={category} | "
            f"input="
            f"{input_width}x{input_height} | "
            f"model_source="
            f"{model_image.width}x"
            f"{model_image.height} | "
            f"pre_cropped={pre_cropped} | "
            f"inference_ms="
            f"{timings['inference_ms']} | "
            f"total_ms="
            f"{timings['total_ms']}"
        )

        return {
            "success": True,
            "category": category,
            "image": encoded_image,
            "meta": {
                "device": DEVICE,
                "fp16": USE_FP16,
                "preCropped":
                    pre_cropped,
                "inputWidth":
                    input_width,
                "inputHeight":
                    input_height,
                "modelSourceWidth":
                    model_image.width,
                "modelSourceHeight":
                    model_image.height,
                "outputBytes":
                    output_size_bytes,
                "timings":
                    timings,
            },
        }

    except Exception as error:
        timings["total_ms"] = round(
            now_ms()
            - started_at,
            1,
        )

        print(
            "JOB FAILED | "
            f"total_ms="
            f"{timings['total_ms']} | "
            f"error={error}"
        )

        return {
            "success": False,
            "error": str(error),
            "meta": {
                "device": DEVICE,
                "fp16": USE_FP16,
                "timings": timings,
            },
        }