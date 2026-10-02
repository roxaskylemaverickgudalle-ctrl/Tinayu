import io
import os

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DETECTOR_PATH = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)

FACE_LANDMARKER_PATH = os.path.join(
    BASE_DIR,
    "face_landmarker.task"
)


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_DETECTOR_PATH
    ),
    running_mode=VisionRunningMode.IMAGE
)

face_detector = FaceDetector.create_from_options(
    face_detector_options
)


face_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_LANDMARKER_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(
    face_landmarker_options
)


# ============================================================
# IMAGE LOADING
# ============================================================

def decode_image(image_bytes):
    """
    Decode image bytes into RGB NumPy array.
    """

    if not image_bytes:
        raise ValueError(
            "Image data is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            "Could not decode the image. "
            "Please upload a valid JPG or PNG image."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    return image_rgb


def load_image(image_path):
    """
    Load an image from a local file path.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with open(
        image_path,
        "rb"
    ) as file:

        image_bytes = file.read()

    return decode_image(
        image_bytes
    )


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(image_rgb):

    image_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = face_detector.detect(
        image_mp
    )

    if not result.detections:
        raise ValueError(
            "No face detected in the image."
        )

    confidence = (
        result
        .detections[0]
        .categories[0]
        .score
    )

    return {
        "result": result,
        "confidence": float(confidence)
    }


# ============================================================
# LANDMARKS
# ============================================================

def detect_landmarks(image_rgb):

    image_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = face_landmarker.detect(
        image_mp
    )

    if not result.face_landmarks:
        raise ValueError(
            "No facial landmarks detected."
        )

    return result.face_landmarks[0]


def landmarks_to_pixels(
    landmarks,
    image_shape
):

    height, width = image_shape[:2]

    points = np.array([
        [
            int(lm.x * width),
            int(lm.y * height)
        ]
        for lm in landmarks
    ])

    return points


# ============================================================
# MASKS
# ============================================================

def create_landmark_mask(
    image_shape,
    points
):

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    polygon = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    return mask


def create_face_mask(
    image_shape,
    landmark_points
):

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    hull = cv2.convexHull(
        landmark_points.astype(np.int32)
    )

    cv2.fillConvexPoly(
        mask,
        hull,
        255
    )

    return mask


# ============================================================
# SKIN EXTRACTION
# ============================================================

def extract_cheek_pixels(
    image_rgb,
    landmark_points
):

    face_mask = create_face_mask(
        image_rgb.shape,
        landmark_points
    )

    left_indices = [
        50,
        101,
        118,
        119,
        100,
        47
    ]

    right_indices = [
        280,
        330,
        347,
        348,
        329,
        277
    ]

    left_points = landmark_points[
        left_indices
    ]

    right_points = landmark_points[
        right_indices
    ]

    left_mask = create_landmark_mask(
        image_rgb.shape,
        left_points
    )

    right_mask = create_landmark_mask(
        image_rgb.shape,
        right_points
    )

    combined_mask = cv2.bitwise_or(
        left_mask,
        right_mask
    )

    combined_mask = cv2.bitwise_and(
        combined_mask,
        face_mask
    )

    kernel = np.ones(
        (9, 9),
        dtype=np.uint8
    )

    combined_mask = cv2.dilate(
        combined_mask,
        kernel,
        iterations=1
    )

    hsv = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    valid_mask = (
        (combined_mask > 0)
        &
        (saturation < 220)
        &
        (value > 35)
        &
        (value < 250)
    )

    pixels = image_rgb[
        valid_mask
    ]

    return pixels


def calculate_skin_color(
    pixels
):

    if pixels is None or len(pixels) == 0:
        raise ValueError(
            "Not enough skin pixels were detected."
        )

    pixels = np.asarray(
        pixels,
        dtype=np.float32
    )

    brightness = pixels.mean(
        axis=1
    )

    low = np.percentile(
        brightness,
        10
    )

    high = np.percentile(
        brightness,
        90
    )

    valid = pixels[
        (brightness >= low)
        &
        (brightness <= high)
    ]

    if len(valid) == 0:
        valid = pixels

    return np.round(
        valid.mean(axis=0)
    ).astype(int)


# ============================================================
# COLOR CONVERSION
# ============================================================

def rgb_to_lab(
    rgb
):
    """
    Convert RGB to standard CIELAB.

    IMPORTANT:
    OpenCV stores LAB as:
        L = 0..255
        a = 0..255
        b = 0..255

    We convert that into:
        L = 0..100
        a = approximately -128..127
        b = approximately -128..127
    """

    rgb_array = np.asarray(
        rgb,
        dtype=np.uint8
    ).reshape(
        1,
        1,
        3
    )

    lab = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2LAB
    )[0, 0].astype(
        np.float32
    )

    L = (
        lab[0] *
        100.0 /
        255.0
    )

    a = (
        lab[1] -
        128.0
    )

    b = (
        lab[2] -
        128.0
    )

    return np.array([
        L,
        a,
        b
    ], dtype=np.float32)


def rgb_to_hsv_features(
    rgb
):

    rgb_array = np.asarray(
        rgb,
        dtype=np.uint8
    ).reshape(
        1,
        1,
        3
    )

    hsv = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2HSV
    )[0, 0]

    return {
        "hue": float(
            hsv[0] * 2
        ),

        "saturation": float(
            hsv[1]
        ),

        "value": float(
            hsv[2]
        )
    }


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_skin_profile(
    skin_rgb
):

    skin_rgb = np.asarray(
        skin_rgb,
        dtype=np.uint8
    )

    lab = rgb_to_lab(
        skin_rgb
    )

    L, a, b = lab

    reference_L = 60.0

    normalized_L = (
        L * 0.5
        +
        reference_L * 0.5
    )

    normalized_L = np.clip(
        normalized_L,
        0,
        100
    )

    # Convert normalized LAB back to OpenCV LAB.
    opencv_lab = np.array(
        [[[
            normalized_L * 255.0 / 100.0,
            a + 128.0,
            b + 128.0
        ]]],
        dtype=np.float32
    )

    opencv_lab = np.clip(
        opencv_lab,
        0,
        255
    ).astype(
        np.uint8
    )

    normalized_rgb = cv2.cvtColor(
        opencv_lab,
        cv2.COLOR_LAB2RGB
    )[0, 0]

    normalized_rgb = normalized_rgb.astype(
        int
    )

    # IMPORTANT:
    # Recalculate LAB from the final RGB so
    # every part of the pipeline uses the exact
    # same color representation.
    final_lab = rgb_to_lab(
        normalized_rgb
    )

    hsv = rgb_to_hsv_features(
        normalized_rgb
    )

    chroma = float(
        np.sqrt(
            final_lab[1] ** 2
            +
            final_lab[2] ** 2
        )
    )

    return {
        "rgb": normalized_rgb,
        "lab": final_lab,
        "hue": hsv["hue"],
        "chroma": chroma
    }


# ============================================================
# HAIR
# ============================================================

def extract_hair_pixels(
    image_rgb,
    landmark_points
):

    height, width = image_rgb.shape[:2]

    forehead_indices = [
        10,
        109,
        67,
        103,
        54,
        21,
        251,
        284,
        332,
        297,
        338
    ]

    forehead_points = landmark_points[
        forehead_indices
    ]

    min_x = int(
        np.min(
            forehead_points[:, 0]
        )
    )

    max_x = int(
        np.max(
            forehead_points[:, 0]
        )
    )

    min_y = int(
        np.min(
            forehead_points[:, 1]
        )
    )

    face_width = (
        max_x -
        min_x
    )

    if face_width <= 0:
        return None

    padding_x = int(
        face_width * 0.20
    )

    hair_height = int(
        face_width * 0.45
    )

    x1 = max(
        0,
        min_x - padding_x
    )

    x2 = min(
        width,
        max_x + padding_x
    )

    y2 = max(
        0,
        min_y + int(
            face_width * 0.05
        )
    )

    y1 = max(
        0,
        y2 - hair_height
    )

    if x2 <= x1 or y2 <= y1:
        return None

    region = image_rgb[
        y1:y2,
        x1:x2
    ]

    hsv = cv2.cvtColor(
        region,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    hair_mask = (
        (value < 190)
        &
        (saturation > 15)
    )

    pixels = region[
        hair_mask
    ]

    if len(pixels) < 20:
        return None

    return pixels


def calculate_hair_color(
    pixels
):

    if pixels is None or len(pixels) == 0:
        return None

    pixels = np.asarray(
        pixels,
        dtype=np.float32
    )

    brightness = pixels.mean(
        axis=1
    )

    low = np.percentile(
        brightness,
        10
    )

    high = np.percentile(
        brightness,
        90
    )

    valid = pixels[
        (brightness >= low)
        &
        (brightness <= high)
    ]

    if len(valid) == 0:
        valid = pixels

    return np.round(
        valid.mean(axis=0)
    ).astype(int)


# ============================================================
# EYES
# ============================================================

def extract_iris_pixels(
    image_rgb,
    landmark_points
):

    height, width = image_rgb.shape[:2]

    iris_groups = [
        [474, 475, 476, 477],
        [469, 470, 471, 472]
    ]

    all_pixels = []

    for indices in iris_groups:

        points = landmark_points[
            indices
        ]

        center_x = int(
            np.mean(
                points[:, 0]
            )
        )

        center_y = int(
            np.mean(
                points[:, 1]
            )
        )

        radius_x = max(
            2,
            int(
                np.ptp(
                    points[:, 0]
                ) * 1.5
            )
        )

        radius_y = max(
            2,
            int(
                np.ptp(
                    points[:, 1]
                ) * 1.5
            )
        )

        x1 = max(
            0,
            center_x - radius_x
        )

        x2 = min(
            width,
            center_x + radius_x + 1
        )

        y1 = max(
            0,
            center_y - radius_y
        )

        y2 = min(
            height,
            center_y + radius_y + 1
        )

        region = image_rgb[
            y1:y2,
            x1:x2
        ]

        if region.size == 0:
            continue

        all_pixels.append(
            region.reshape(
                -1,
                3
            )
        )

    if not all_pixels:
        return None

    return np.concatenate(
        all_pixels,
        axis=0
    )


def calculate_eye_color(
    pixels
):

    if pixels is None or len(pixels) == 0:
        return None

    pixels = np.asarray(
        pixels,
        dtype=np.float32
    )

    brightness = pixels.mean(
        axis=1
    )

    low = np.percentile(
        brightness,
        15
    )

    high = np.percentile(
        brightness,
        85
    )

    valid = pixels[
        (brightness >= low)
        &
        (brightness <= high)
    ]

    if len(valid) == 0:
        valid = pixels

    return np.round(
        valid.mean(axis=0)
    ).astype(int)


# ============================================================
# CONTRAST
# ============================================================

def calculate_contrast(
    color1_rgb,
    color2_rgb
):

    if (
        color1_rgb is None
        or
        color2_rgb is None
    ):
        return None

    lab1 = rgb_to_lab(
        color1_rgb
    )

    lab2 = rgb_to_lab(
        color2_rgb
    )

    return float(
        np.linalg.norm(
            lab1 - lab2
        )
    )


def classify_contrast(
    value
):

    if value is None:
        return "Unknown"

    if value < 10:
        return "Low"

    if value < 25:
        return "Moderate"

    return "High"


# ============================================================
# SKIN INTERPRETATION
# ============================================================

def classify_skin_depth(
    lab
):

    L = float(
        lab[0]
    )

    if L < 35:
        return "Deep"

    if L < 50:
        return "Medium"

    if L < 65:
        return "Medium-Light"

    return "Light"


def classify_temperature(
    lab
):

    a = float(
        lab[1]
    )

    b = float(
        lab[2]
    )

    if b > 12 and a > 5:
        return "Warm"

    if b < 5 and a < 5:
        return "Cool"

    return "Neutral"


def classify_saturation(
    chroma
):

    if chroma < 15:
        return "Muted"

    if chroma < 28:
        return "Moderate"

    return "Clear"


# ============================================================
# SEASON
# ============================================================

def suggest_season(
    temperature,
    skin_depth,
    skin_saturation
):

    if temperature == "Warm":

        if skin_saturation == "Clear":
            return "Warm Spring"

        if skin_depth in [
            "Medium",
            "Deep"
        ]:
            return "Warm Autumn"

        return "Warm Spring"

    if temperature == "Cool":

        if skin_saturation == "Clear":
            return "Cool Winter"

        return "Cool Summer"

    if skin_saturation == "Clear":
        return "Spring"

    return "Autumn"


# ============================================================
# PROFILE
# ============================================================

def build_automatic_profile(
    skin_rgb,
    hair_rgb=None,
    eye_rgb=None
):

    # --------------------------------------------------------
    # Normalize ONCE
    # --------------------------------------------------------

    skin_profile = normalize_skin_profile(
        skin_rgb
    )

    normalized_skin_rgb = (
        skin_profile["rgb"]
    )

    normalized_skin_lab = (
        skin_profile["lab"]
    )

    skin_depth = classify_skin_depth(
        normalized_skin_lab
    )

    temperature = classify_temperature(
        normalized_skin_lab
    )

    skin_saturation = classify_saturation(
        skin_profile["chroma"]
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    skin_hair_contrast = calculate_contrast(
        normalized_skin_rgb,
        hair_rgb
    )

    skin_eye_contrast = calculate_contrast(
        normalized_skin_rgb,
        eye_rgb
    )

    hair_eye_contrast = calculate_contrast(
        hair_rgb,
        eye_rgb
    )

    available = [
        value
        for value in [
            skin_hair_contrast,
            skin_eye_contrast
        ]
        if value is not None
    ]

    average_contrast = (
        float(
            np.mean(available)
        )
        if available
        else None
    )

    contrast_level = classify_contrast(
        average_contrast
    )

    season = suggest_season(
        temperature,
        skin_depth,
        skin_saturation
    )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile = {
        "skin": {
            "rgb": normalized_skin_rgb.tolist(),

            "lab": {
                "L": round(
                    float(
                        normalized_skin_lab[0]
                    ),
                    2
                ),

                "a": round(
                    float(
                        normalized_skin_lab[1]
                    ),
                    2
                ),

                "b": round(
                    float(
                        normalized_skin_lab[2]
                    ),
                    2
                )
            },

            "hue": round(
                float(
                    skin_profile["hue"]
                ),
                2
            ),

            "chroma": round(
                float(
                    skin_profile["chroma"]
                ),
                2
            )
        },

        "heuristics": {
            "skin_category": skin_depth,
            "temperature": temperature,
            "suggested_season": season,
            "skin_depth": skin_depth,
            "skin_saturation": skin_saturation,
            "contrast_level": contrast_level
        },

        "interpretation": {
            "skin_depth": skin_depth.lower(),
            "skin_saturation": skin_saturation.lower(),
            "contrast_level": contrast_level.lower()
        },

        "contrast": {
            "skin_hair": (
                round(
                    skin_hair_contrast,
                    2
                )
                if skin_hair_contrast is not None
                else None
            ),

            "skin_eye": (
                round(
                    skin_eye_contrast,
                    2
                )
                if skin_eye_contrast is not None
                else None
            ),

            "hair_eye": (
                round(
                    hair_eye_contrast,
                    2
                )
                if hair_eye_contrast is not None
                else None
            )
        }
    }

    # --------------------------------------------------------
    # Hair
    # --------------------------------------------------------

    if hair_rgb is not None:

        hair_lab = rgb_to_lab(
            hair_rgb
        )

        profile["hair"] = {
            "rgb": np.asarray(
                hair_rgb,
                dtype=int
            ).tolist(),

            "lab": {
                "L": round(
                    float(
                        hair_lab[0]
                    ),
                    2
                ),

                "a": round(
                    float(
                        hair_lab[1]
                    ),
                    2
                ),

                "b": round(
                    float(
                        hair_lab[2]
                    ),
                    2
                )
            }
        }

    # --------------------------------------------------------
    # Eyes
    # --------------------------------------------------------

    if eye_rgb is not None:

        eye_lab = rgb_to_lab(
            eye_rgb
        )

        profile["eye"] = {
            "rgb": np.asarray(
                eye_rgb,
                dtype=int
            ).tolist(),

            "lab": {
                "L": round(
                    float(
                        eye_lab[0]
                    ),
                    2
                ),

                "a": round(
                    float(
                        eye_lab[1]
                    ),
                    2
                ),

                "b": round(
                    float(
                        eye_lab[2]
                    ),
                    2
                )
            }
        }

    return profile


# ============================================================
# PALETTES
# ============================================================

CLOTHING_PALETTE = [
    ("Cream", "#FFF1D0"),
    ("Warm Beige", "#C8A27A"),
    ("Sand", "#D6B98C"),
    ("Taupe", "#A89A8A"),
    ("Mushroom", "#A89B8E"),
    ("Warm Gray", "#9A928A"),
    ("Chocolate Brown", "#5A3825"),
    ("Deep Brown", "#4A2C20"),
    ("Brown", "#8B5E3C"),
    ("Warm Brown", "#8B5A2B"),
    ("Olive", "#808000"),
    ("Deep Olive", "#556B2F"),
    ("Dusty Olive", "#8A8F68"),
    ("Sage", "#9CAF88"),
    ("Moss Green", "#687D3C"),
    ("Forest Green", "#355E3B"),
    ("Muted Teal", "#5F8A82"),
    ("Deep Teal", "#356F6B"),
    ("Dusty Blue", "#7F9BAA"),
    ("Warm Navy", "#34495E"),
    ("Muted Blue", "#6F8795"),
    ("Terracotta", "#C65D3B"),
    ("Rust", "#B7410E"),
    ("Soft Rust", "#C8755D"),
    ("Burnt Orange", "#CC5500"),
    ("Warm Coral", "#E8836B"),
    ("Peach", "#F2A07B"),
    ("Brick", "#A94442"),
    ("Dusty Rose", "#C08081"),
    ("Warm Rose", "#C46F72"),
    ("Muted Berry", "#8F4A5B"),
    ("Warm Plum", "#70435A"),
    ("Muted Plum", "#806276"),
    ("Mustard", "#C2A83E"),
    ("Camel", "#C19A6B")
]


MAKEUP_PALETTE = [
    ("Warm Nude", "#C98F78"),
    ("Peach", "#F2A07B"),
    ("Warm Coral", "#E8836B"),
    ("Terracotta", "#C65D3B"),
    ("Soft Rust", "#C8755D"),
    ("Warm Rose", "#C46F72"),
    ("Dusty Rose", "#C08081"),
    ("Brown", "#8B5E3C"),
    ("Warm Brown", "#8B5A2B"),
    ("Brick", "#A94442"),
    ("Warm Plum", "#70435A"),
    ("Muted Berry", "#8F4A5B")
]


ACCENT_PALETTE = [
    ("Bronze", "#CD7F32"),
    ("Copper", "#B87333"),
    ("Muted Gold", "#B08D57"),
    ("Honey", "#D4A017"),
    ("Antique Gold", "#C6A15B"),
    ("Warm Beige", "#C8A27A"),
    ("Olive", "#808000")
]


# ============================================================
# COLOR FEATURES
# ============================================================

def get_color_features(
    hex_color
):

    hex_color = hex_color.lstrip(
        "#"
    )

    rgb = np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    ])

    lab = rgb_to_lab(
        rgb
    )

    hsv = rgb_to_hsv_features(
        rgb
    )

    chroma = float(
        np.sqrt(
            lab[1] ** 2
            +
            lab[2] ** 2
        )
    )

    return {
        "rgb": rgb,
        "lab": lab,
        "hue": hsv["hue"],
        "saturation": hsv["saturation"],
        "value": hsv["value"],
        "chroma": chroma
    }


def circular_hue_distance(
    hue1,
    hue2
):

    difference = abs(
        hue1 - hue2
    )

    return min(
        difference,
        360 - difference
    )


# ============================================================
# CALIBRATED SCORING
# ============================================================

def score_color(
    color_name,
    hex_color,
    profile,
    category
):

    features = get_color_features(
        hex_color
    )

    skin = profile[
        "skin"
    ]

    skin_hue = skin[
        "hue"
    ]

    skin_chroma = skin[
        "chroma"
    ]

    skin_L = skin[
        "lab"
    ][
        "L"
    ]

    temperature = profile[
        "heuristics"
    ][
        "temperature"
    ]

    saturation = profile[
        "heuristics"
    ][
        "skin_saturation"
    ]

    depth = profile[
        "heuristics"
    ][
        "skin_depth"
    ]

    season = profile[
        "heuristics"
    ][
        "suggested_season"
    ]

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if features["lab"][2] >= 8:
        color_temperature = "Warm"

    elif features["lab"][2] <= 2:
        color_temperature = "Cool"

    else:
        color_temperature = "Neutral"

    if color_temperature == temperature:
        temperature_score = 1.0

    elif color_temperature == "Neutral":
        temperature_score = 0.65

    else:
        temperature_score = 0.25

    # --------------------------------------------------------
    # Hue
    # --------------------------------------------------------

    hue_distance = circular_hue_distance(
        skin_hue,
        features["hue"]
    )

    hue_score = (
        1.0 -
        min(
            hue_distance / 180.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma_difference = abs(
        features["chroma"]
        -
        skin_chroma
    )

    chroma_score = (
        1.0 -
        min(
            chroma_difference / 55.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Lightness
    # --------------------------------------------------------

    lightness_difference = abs(
        features["lab"][0]
        -
        skin_L
    )

    lightness_score = (
        1.0 -
        min(
            lightness_difference / 70.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Base score
    # --------------------------------------------------------

    if category == "clothing":

        compatibility = (
            temperature_score * 0.35
            +
            hue_score * 0.25
            +
            chroma_score * 0.20
            +
            lightness_score * 0.20
        )

    elif category == "makeup":

        compatibility = (
            temperature_score * 0.30
            +
            hue_score * 0.35
            +
            chroma_score * 0.25
            +
            lightness_score * 0.10
        )

    else:

        compatibility = (
            temperature_score * 0.40
            +
            hue_score * 0.25
            +
            chroma_score * 0.20
            +
            lightness_score * 0.15
        )

    # --------------------------------------------------------
    # Saturation preference
    # --------------------------------------------------------

    if saturation == "Clear":

        if features["chroma"] >= 25:
            compatibility += 0.035

    elif saturation == "Muted":

        if features["chroma"] <= 28:
            compatibility += 0.035

    elif saturation == "Moderate":

        if 15 <= features["chroma"] <= 35:
            compatibility += 0.025

    # --------------------------------------------------------
    # Depth preference
    # --------------------------------------------------------

    if depth == "Deep":

        if features["lab"][0] < 55:
            compatibility += 0.025

    elif depth == "Light":

        if features["lab"][0] > 55:
            compatibility += 0.025

    # --------------------------------------------------------
    # Season preference
    # --------------------------------------------------------

    if season in [
        "Warm Spring",
        "Spring"
    ]:

        if (
            color_temperature == "Warm"
            and
            features["chroma"] >= 25
        ):
            compatibility += 0.035

    elif season in [
        "Warm Autumn",
        "Autumn"
    ]:

        if (
            color_temperature == "Warm"
            and
            features["chroma"] <= 35
        ):
            compatibility += 0.035

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    compatibility = np.clip(
        compatibility,
        0.0,
        1.0
    )

    # Wider output range to preserve separation.
    final_score = (
        60.0
        +
        compatibility * 35.0
    )

    return round(
        float(final_score),
        1
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendation(
    palette,
    profile,
    category,
    limit=5
):

    scored = []

    for name, hex_color in palette:

        score = score_color(
            name,
            hex_color,
            profile,
            category
        )

        scored.append({
            "name": name,
            "hex": hex_color,
            "score": score
        })

    scored.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored[:limit]


def build_structured_recommendations(
    profile
):

    return {
        "clothing": build_recommendation(
            CLOTHING_PALETTE,
            profile,
            "clothing",
            5
        ),

        "makeup": build_recommendation(
            MAKEUP_PALETTE,
            profile,
            "makeup",
            5
        ),

        "accents": build_recommendation(
            ACCENT_PALETTE,
            profile,
            "accents",
            5
        )
    }


# ============================================================
# QUALITY
# ============================================================

def calculate_image_confidence(
    face_confidence,
    skin_pixels,
    hair_detected,
    eyes_detected
):

    score = (
        face_confidence * 60
    )

    if skin_pixels >= 300:
        score += 25

    elif skin_pixels >= 150:
        score += 18

    elif skin_pixels >= 50:
        score += 10

    else:
        score += 4

    if hair_detected:
        score += 7

    if eyes_detected:
        score += 8

    return round(
        min(
            score,
            100
        ),
        2
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def analyze_image(
    image_bytes
):

    image_rgb = decode_image(
        image_bytes
    )

    height, width = image_rgb.shape[:2]

    # --------------------------------------------------------
    # Face
    # --------------------------------------------------------

    face = detect_face(
        image_rgb
    )

    face_confidence = face[
        "confidence"
    ]

    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------

    landmarks = detect_landmarks(
        image_rgb
    )

    landmark_points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    # --------------------------------------------------------
    # Skin
    # --------------------------------------------------------

    skin_pixels = extract_cheek_pixels(
        image_rgb,
        landmark_points
    )

    if len(skin_pixels) < 20:
        raise ValueError(
            "Not enough skin pixels were detected. "
            "Please use a clearer front-facing photo."
        )

    raw_skin_rgb = calculate_skin_color(
        skin_pixels
    )

    normalized_skin = normalize_skin_profile(
        raw_skin_rgb
    )

    normalized_skin_rgb = normalized_skin[
        "rgb"
    ]

    # --------------------------------------------------------
    # Hair
    # --------------------------------------------------------

    hair_pixels = extract_hair_pixels(
        image_rgb,
        landmark_points
    )

    hair_rgb = calculate_hair_color(
        hair_pixels
    )

    # --------------------------------------------------------
    # Eyes
    # --------------------------------------------------------

    eye_pixels = extract_iris_pixels(
        image_rgb,
        landmark_points
    )

    eye_rgb = calculate_eye_color(
        eye_pixels
    )

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profile = build_automatic_profile(
        normalized_skin_rgb,
        hair_rgb=hair_rgb,
        eye_rgb=eye_rgb
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = build_structured_recommendations(
        profile
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    image_confidence = calculate_image_confidence(
        face_confidence,
        len(skin_pixels),
        hair_rgb is not None,
        eye_rgb is not None
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "success": True,

        "image": {
            "width": width,
            "height": height
        },

        "face": {
            "confidence": round(
                float(face_confidence),
                4
            ),
            "landmarks": len(
                landmark_points
            )
        },

        "quality": {
            "image_confidence": image_confidence,
            "hair_detected": hair_rgb is not None,
            "eyes_detected": eye_rgb is not None,
            "skin_pixels_analyzed": int(
                len(skin_pixels)
            )
        },

        "colors": {
            "skin_raw_rgb": raw_skin_rgb.tolist(),

            "skin_normalized_rgb": normalized_skin_rgb.tolist(),

            "skin_lab": {
                "L": round(
                    float(
                        normalized_skin["lab"][0]
                    ),
                    2
                ),
                "a": round(
                    float(
                        normalized_skin["lab"][1]
                    ),
                    2
                ),
                "b": round(
                    float(
                        normalized_skin["lab"][2]
                    ),
                    2
                )
            },

            "hair_rgb": (
                hair_rgb.tolist()
                if hair_rgb is not None
                else None
            ),

            "eye_rgb": (
                eye_rgb.tolist()
                if eye_rgb is not None
                else None
            )
        },

        "profile": profile,

        "recommendations": recommendations,

        "normalization": {
            "enabled": True,
            "method": "LAB lightness normalization",
            "raw_skin_rgb": raw_skin_rgb.tolist(),
            "normalized_skin_rgb": normalized_skin_rgb.tolist()
        }
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_path = os.path.join(
        BASE_DIR,
        "test.jpg"
    )

    print(
        "Running Tinayu engine test..."
    )

    image = load_image(
        test_path
    )

    print(
        "Image loaded:",
        image.shape
    )

    with open(
        test_path,
        "rb"
    ) as file:

        image_bytes = file.read()

    result = analyze_image(
        image_bytes
    )

    print(
        "\n========== TINAYU ANALYSIS =========="
    )

    print(
        "Success:",
        result["success"]
    )

    print(
        "Image:",
        result["image"]
    )

    print(
        "Face confidence:",
        result["face"]["confidence"]
    )

    print(
        "Landmarks:",
        result["face"]["landmarks"]
    )

    print(
        "Skin pixels:",
        result["quality"]["skin_pixels_analyzed"]
    )

    print(
        "Raw skin RGB:",
        result["colors"]["skin_raw_rgb"]
    )

    print(
        "Normalized skin RGB:",
        result["colors"]["skin_normalized_rgb"]
    )

    print(
        "Hair RGB:",
        result["colors"]["hair_rgb"]
    )

    print(
        "Eye RGB:",
        result["colors"]["eye_rgb"]
    )

    print(
        "Season:",
        result["profile"]["heuristics"]["suggested_season"]
    )

    print(
        "Clothing:",
        result["recommendations"]["clothing"]
    )

    print(
        "Makeup:",
        result["recommendations"]["makeup"]
    )

    print(
        "Accents:",
        result["recommendations"]["accents"]
    )

    print(
        "====================================="
    )