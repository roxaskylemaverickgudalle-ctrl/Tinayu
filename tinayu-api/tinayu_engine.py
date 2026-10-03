import os
import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FACE_DETECTOR_PATH = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)

FACE_LANDMARKER_PATH = os.path.join(
    BASE_DIR,
    "face_landmarker.task"
)


# ============================================================
# MEDIAPIPE
# ============================================================

face_detector = mp.tasks.vision.FaceDetector.create_from_options(
    mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=FACE_DETECTOR_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
)


face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=FACE_LANDMARKER_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )
)


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
    ("Honey", "#C99A3D"),
    ("Antique Gold", "#C6A15B"),

    ("Warm Beige", "#C8A27A"),
    ("Olive", "#808000")
]


# ============================================================
# IMAGE LOADING
# ============================================================

def decode_image(image_bytes):
    """
    Decode uploaded image bytes into RGB image.
    """

    if not image_bytes:
        raise ValueError(
            "The uploaded image is empty."
        )

    buffer = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image_bgr = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            "Unable to decode the image. "
            "Please upload a valid JPG or PNG image."
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


def load_image(image_path):
    """
    Load an image from a local file.
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
    """
    Detect a face using MediaPipe BlazeFace.
    """

    image_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    result = face_detector.detect(
        image_mp
    )

    if not result.detections:
        raise ValueError(
            "No face detected in the image. "
            "Please use a clear front-facing photo."
        )

    confidence = (
        result.detections[0]
        .categories[0]
        .score
    )

    return result


# ============================================================
# FACIAL LANDMARKS
# ============================================================

def detect_landmarks(image_rgb):
    """
    Detect 478 facial landmarks.
    """

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
    """
    Convert normalized MediaPipe coordinates
    into image pixel coordinates.
    """

    height, width = image_shape[:2]

    points = np.array([
        [
            int(
                np.clip(
                    lm.x * width,
                    0,
                    width - 1
                )
            ),
            int(
                np.clip(
                    lm.y * height,
                    0,
                    height - 1
                )
            )
        ]

        for lm in landmarks
    ])

    return points


# ============================================================
# COLOR SPACE UTILITIES
# ============================================================

def rgb_to_lab(rgb):
    """
    Convert RGB color to standard-ish LAB values.

    Returns:
        L: 0-100
        a: approximately -128 to 127
        b: approximately -128 to 127
    """

    rgb_array = np.asarray(
        rgb,
        dtype=np.uint8
    ).reshape(1, 1, 3)

    lab = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2LAB
    )[0, 0]

    L = float(
        lab[0]
    ) * 100.0 / 255.0

    a = float(
        lab[1]
    ) - 128.0

    b = float(
        lab[2]
    ) - 128.0

    return np.array([
        L,
        a,
        b
    ])


def rgb_to_hsv_features(rgb):
    """
    Extract HSV features safely.
    """

    rgb_array = np.asarray(
        rgb,
        dtype=np.uint8
    ).reshape(1, 1, 3)

    hsv = cv2.cvtColor(
        rgb_array,
        cv2.COLOR_RGB2HSV
    )[0, 0]

    hue = float(
        hsv[0]
    ) * 2.0

    saturation = float(
        hsv[1]
    )

    value = float(
        hsv[2]
    )

    return {
        "hue": hue,
        "saturation": saturation,
        "value": value
    }


def rgb_distance(
    color_a,
    color_b
):
    """
    Euclidean RGB distance.
    """

    a = np.asarray(
        color_a,
        dtype=float
    )

    b = np.asarray(
        color_b,
        dtype=float
    )

    return float(
        np.linalg.norm(
            a - b
        )
    )


# ============================================================
# REGION EXTRACTION
# ============================================================

def polygon_mask(
    image_shape,
    points
):
    """
    Create a filled polygon mask.
    """

    height, width = image_shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    points = np.asarray(
        points,
        dtype=np.int32
    )

    if len(points) >= 3:

        cv2.fillPoly(
            mask,
            [points],
            255
        )

    return mask


def extract_region_mean(
    image_rgb,
    mask
):
    """
    Extract mean RGB color from a mask.
    """

    pixels = image_rgb[
        mask > 0
    ]

    if len(pixels) == 0:
        return None

    return np.mean(
        pixels,
        axis=0
    ).astype(
        int
    ).tolist()


def extract_skin_color(
    image_rgb,
    landmarks
):
    """
    Extract skin from left and right cheek regions.
    """

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
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

    left_points = points[
        left_indices
    ]

    right_points = points[
        right_indices
    ]

    left_mask = polygon_mask(
        image_rgb.shape,
        left_points
    )

    right_mask = polygon_mask(
        image_rgb.shape,
        right_points
    )

    combined_mask = cv2.bitwise_or(
        left_mask,
        right_mask
    )

    pixels = image_rgb[
        combined_mask > 0
    ]

    if len(pixels) == 0:
        raise ValueError(
            "Unable to extract enough skin pixels."
        )

    return np.mean(
        pixels,
        axis=0
    ).astype(
        int
    ).tolist()


def extract_hair_color(
    image_rgb,
    landmarks
):
    """
    Estimate hair color from areas immediately above and beside
    the forehead.

    The previous implementation sampled a large rectangle above
    the forehead. That can accidentally capture sky or a bright
    background and then treat it as hair. This version uses a few
    smaller candidate regions and rejects pixels that look like
    bright background rather than plausible hair.
    """

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    height, width = image_rgb.shape[:2]

    # Facial landmarks around the upper face / temples.
    top_indices = [
        10,
        338,
        297,
        332,
        284
    ]

    left_indices = [
        127,
        234,
        93,
        132
    ]

    right_indices = [
        356,
        454,
        323,
        361
    ]

    top_points = points[top_indices]
    left_points = points[left_indices]
    right_points = points[right_indices]

    face_top_y = int(
        np.min(top_points[:, 1])
    )

    face_left_x = int(
        np.min(left_points[:, 0])
    )

    face_right_x = int(
        np.max(right_points[:, 0])
    )

    # Small regions are more reliable than one large background-heavy
    # rectangle.
    regions = []

    forehead_width = max(
        face_right_x - face_left_x,
        40
    )

    x_pad = max(
        int(forehead_width * 0.08),
        8
    )

    x1 = max(
        face_left_x + x_pad,
        0
    )

    x2 = min(
        face_right_x - x_pad,
        width
    )

    y1 = max(
        face_top_y - 45,
        0
    )

    y2 = max(
        face_top_y - 4,
        0
    )

    if x2 > x1 and y2 > y1:
        regions.append(
            image_rgb[y1:y2, x1:x2]
        )

    # Temple strips catch hair when the forehead is partly covered,
    # while keeping the sample close to the face.
    temple_height = max(
        int(height * 0.08),
        20
    )

    left_x1 = max(
        face_left_x - max(int(forehead_width * 0.10), 8),
        0
    )

    left_x2 = min(
        face_left_x + max(int(forehead_width * 0.02), 5),
        width
    )

    right_x1 = max(
        face_right_x - max(int(forehead_width * 0.02), 5),
        0
    )

    right_x2 = min(
        face_right_x + max(int(forehead_width * 0.10), 8),
        width
    )

    temple_y1 = max(
        face_top_y,
        0
    )

    temple_y2 = min(
        face_top_y + temple_height,
        height
    )

    if left_x2 > left_x1 and temple_y2 > temple_y1:
        regions.append(
            image_rgb[
                temple_y1:temple_y2,
                left_x1:left_x2
            ]
        )

    if right_x2 > right_x1 and temple_y2 > temple_y1:
        regions.append(
            image_rgb[
                temple_y1:temple_y2,
                right_x1:right_x2
            ]
        )

    if not regions:
        return None

    candidates = []

    for region in regions:

        if region.size == 0:
            continue

        pixels = region.reshape(
            -1,
            3
        ).astype(
            np.uint8
        )

        hsv = cv2.cvtColor(
            pixels.reshape(1, -1, 3),
            cv2.COLOR_RGB2HSV
        )[0]

        # OpenCV HSV:
        # H = 0..179, S = 0..255, V = 0..255
        hue = hsv[:, 0].astype(float) * 2.0
        saturation = hsv[:, 1].astype(float)
        value = hsv[:, 2].astype(float)

        # Reject obvious bright blue/cyan sky and similarly bright
        # background pixels. These were responsible for colors such
        # as [114, 193, 250] being detected as "hair".
        bright_cool_background = (
            (value > 220)
            &
            (saturation > 45)
            &
            (hue >= 150)
            &
            (hue <= 270)
        )

        # Reject near-white/gray background.
        bright_neutral_background = (
            (value > 238)
            &
            (saturation < 35)
        )

        valid = ~(
            bright_cool_background
            |
            bright_neutral_background
        )

        filtered = pixels[valid]

        if len(filtered) == 0:
            continue

        # Remove extreme highlights/shadows so a small reflection or
        # border pixel does not dominate the estimate.
        filtered_hsv = cv2.cvtColor(
            filtered.reshape(1, -1, 3),
            cv2.COLOR_RGB2HSV
        )[0]

        filtered_value = filtered_hsv[:, 2]

        low = np.percentile(
            filtered_value,
            10
        )

        high = np.percentile(
            filtered_value,
            90
        )

        trimmed = filtered[
            (filtered_value >= low)
            &
            (filtered_value <= high)
        ]

        if len(trimmed) >= 5:
            candidates.append(trimmed)

    if not candidates:
        return None

    pixels = np.concatenate(
        candidates,
        axis=0
    )

    if len(pixels) < 5:
        return None

    # Median is more robust than a mean when a few background pixels
    # survive the filtering.
    return np.median(
        pixels,
        axis=0
    ).astype(
        int
    ).tolist()


def extract_eye_color(
    image_rgb,
    landmarks
):
    """
    Estimate eye/iris color.

    Uses the iris landmark groups and samples
    the central iris area.
    """

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
    )

    left_iris_indices = [
        474,
        475,
        476,
        477
    ]

    right_iris_indices = [
        469,
        470,
        471,
        472
    ]

    iris_colors = []

    for iris_indices in [
        left_iris_indices,
        right_iris_indices
    ]:

        iris_points = points[
            iris_indices
        ]

        center = np.mean(
            iris_points,
            axis=0
        ).astype(
            int
        )

        center_x = int(
            center[0]
        )

        center_y = int(
            center[1]
        )

        height, width = image_rgb.shape[:2]

        radius = 4

        x1 = max(
            center_x - radius,
            0
        )

        x2 = min(
            center_x + radius + 1,
            width
        )

        y1 = max(
            center_y - radius,
            0
        )

        y2 = min(
            center_y + radius + 1,
            height
        )

        region = image_rgb[
            y1:y2,
            x1:x2
        ]

        if region.size == 0:
            continue

        iris_colors.append(
            np.median(
                region.reshape(
                    -1,
                    3
                ),
                axis=0
            )
        )

    if not iris_colors:
        return None

    return np.mean(
        iris_colors,
        axis=0
    ).astype(
        int
    ).tolist()


# ============================================================
# LIGHTING NORMALIZATION
# ============================================================

def normalize_skin_profile(
    skin_rgb
):
    """
    Normalize skin lightness toward a reference
    while preserving hue-related LAB components.
    """

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

    normalized_lab = np.array([
        normalized_L,
        a,
        b
    ])

    opencv_lab = np.array([[
        [
            normalized_lab[0]
            * 255.0
            / 100.0,

            normalized_lab[1]
            + 128.0,

            normalized_lab[2]
            + 128.0
        ]
    ]], dtype=np.float32)

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

    normalized_rgb = (
        normalized_rgb.astype(
            int
        )
    )

    final_lab = rgb_to_lab(
        normalized_rgb
    )

    hsv = rgb_to_hsv_features(
        normalized_rgb
    )

    return {
        "rgb": normalized_rgb,
        "lab": final_lab,
        "hue": hsv["hue"],
        "chroma": float(
            np.sqrt(
                final_lab[1] ** 2
                +
                final_lab[2] ** 2
            )
        )
    }


# ============================================================
# COLOR FEATURE EXTRACTION
# ============================================================

def get_color_features(
    hex_color
):
    """
    Convert palette HEX color into
    RGB, LAB, HSV, hue and chroma features.
    """

    hex_color = hex_color.lstrip(
        "#"
    )

    rgb = tuple(
        int(
            hex_color[i:i + 2],
            16
        )

        for i in (
            0,
            2,
            4
        )
    )

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


# ============================================================
# COLOR CLASSIFICATION
# ============================================================

# ============================================================
# RECOMMENDATION ENGINE V2
# Season/profile-aware palette evaluation
# ============================================================

SEASON_PROFILES = {
    "Warm Spring": {
        "temperature": {
            "Warm": 1.00,
            "Neutral": 0.70,
            "Cool": 0.25
        },
        "chroma_range": (25, 70),
        "lightness_range": (55, 92),
        "preferred_families": {
            "Orange": 1.00,
            "Yellow": 0.95,
            "Red": 0.90,
            "Green": 0.82,
            "Pink": 0.78,
            "Brown": 0.72,
            "Blue": 0.58,
            "Purple": 0.45,
            "Cyan": 0.50
        }
    },

    "Warm Autumn": {
        "temperature": {
            "Warm": 1.00,
            "Neutral": 0.72,
            "Cool": 0.25
        },
        "chroma_range": (5, 45),
        "lightness_range": (30, 75),
        "preferred_families": {
            "Brown": 1.00,
            "Orange": 0.95,
            "Yellow": 0.90,
            "Green": 0.88,
            "Red": 0.82,
            "Pink": 0.60,
            "Blue": 0.48,
            "Purple": 0.45,
            "Cyan": 0.40
        }
    },

    "Cool Summer": {
        "temperature": {
            "Cool": 1.00,
            "Neutral": 0.72,
            "Warm": 0.30
        },
        "chroma_range": (5, 32),
        "lightness_range": (45, 88),
        "preferred_families": {
            "Blue": 1.00,
            "Purple": 0.95,
            "Pink": 0.90,
            "Cyan": 0.90,
            "Green": 0.76,
            "Red": 0.65,
            "Brown": 0.48,
            "Orange": 0.32,
            "Yellow": 0.30
        }
    },

    "Cool Winter": {
        "temperature": {
            "Cool": 1.00,
            "Neutral": 0.68,
            "Warm": 0.25
        },
        "chroma_range": (25, 75),
        "lightness_range": (25, 90),
        "preferred_families": {
            "Blue": 1.00,
            "Purple": 0.98,
            "Red": 0.95,
            "Pink": 0.90,
            "Cyan": 0.88,
            "Green": 0.70,
            "Brown": 0.35,
            "Orange": 0.25,
            "Yellow": 0.25
        }
    },

    # Fallback profiles used when the neutral heuristic
    # returns a broad season rather than one of the four
    # primary seasonal profiles.
    "Spring": {
        "temperature": {
            "Warm": 0.90,
            "Neutral": 0.75,
            "Cool": 0.40
        },
        "chroma_range": (20, 65),
        "lightness_range": (55, 92),
        "preferred_families": {
            "Orange": 0.90,
            "Yellow": 0.90,
            "Pink": 0.82,
            "Green": 0.80,
            "Red": 0.78,
            "Blue": 0.62,
            "Purple": 0.55,
            "Brown": 0.55,
            "Cyan": 0.60
        }
    },

    "Autumn": {
        "temperature": {
            "Warm": 0.90,
            "Neutral": 0.75,
            "Cool": 0.40
        },
        "chroma_range": (5, 45),
        "lightness_range": (30, 75),
        "preferred_families": {
            "Brown": 0.95,
            "Orange": 0.92,
            "Yellow": 0.88,
            "Green": 0.85,
            "Red": 0.80,
            "Pink": 0.58,
            "Blue": 0.48,
            "Purple": 0.45,
            "Cyan": 0.40
        }
    },

    "Neutral": {
        "temperature": {
            "Warm": 0.75,
            "Neutral": 1.00,
            "Cool": 0.75
        },
        "chroma_range": (8, 45),
        "lightness_range": (35, 88),
        "preferred_families": {
            "Blue": 0.78,
            "Green": 0.78,
            "Pink": 0.76,
            "Red": 0.72,
            "Brown": 0.72,
            "Orange": 0.68,
            "Purple": 0.70,
            "Yellow": 0.65,
            "Cyan": 0.72
        }
    }
}


def classify_color_temperature(features):
    """
    Classify a candidate color as Warm, Neutral, or Cool.

    This is a heuristic used by Tinayu's recommendation engine.
    """

    hue = float(features["hue"]) % 360

    if hue <= 70 or hue >= 330:
        return "Warm"

    if 160 <= hue <= 290:
        return "Cool"

    return "Neutral"


def classify_color_chroma(chroma):
    """
    Convert LAB chroma into a simple palette characteristic.
    """

    chroma = float(chroma)

    if chroma < 12:
        return "Muted"

    if chroma < 28:
        return "Soft"

    if chroma < 50:
        return "Clear"

    return "Vivid"


def get_hue_family(hue):
    """
    Convert hue into a broad color family.

    The recommendation engine uses broad families rather than
    relying only on numerical hue distance.
    """

    hue = float(hue) % 360

    if hue < 20 or hue >= 340:
        return "Red"

    if hue < 45:
        return "Orange"

    if hue < 70:
        return "Yellow"

    if hue < 160:
        return "Green"

    if hue < 200:
        return "Cyan"

    if hue < 255:
        return "Blue"

    if hue < 290:
        return "Purple"

    return "Pink"


def get_recommendation_hue_family(features):
    """
    Resolve a palette color into a recommendation-specific family.

    Dark warm orange hues are treated as Brown because HSV hue
    alone cannot distinguish brown from orange.
    """

    hue = float(features["hue"]) % 360
    lightness = float(features["lab"][0])

    if 20 <= hue < 45 and lightness < 55:
        return "Brown"

    return get_hue_family(hue)


def range_score(value, minimum, maximum):
    """
    Score how well a value fits inside a preferred range.

    1.0 means the value is inside the range.
    Values farther outside the range receive progressively lower scores.
    """

    value = float(value)

    if minimum <= value <= maximum:
        return 1.0

    if value < minimum:
        distance = minimum - value
    else:
        distance = value - maximum

    return float(max(0.0, 1.0 - (distance / 40.0)))


def get_season_profile(season):
    """
    Return a configured season profile.

    Falls back to Neutral if a season is not explicitly configured.
    """

    return SEASON_PROFILES.get(
        season,
        SEASON_PROFILES["Neutral"]
    )


def season_compatibility(features, season):
    """
    Evaluate how well a candidate color matches the detected season.
    """

    profile = get_season_profile(season)

    temperature = classify_color_temperature(features)
    chroma = float(features["chroma"])
    lightness = float(features["lab"][0])
    family = get_recommendation_hue_family(features)

    temperature_score = profile["temperature"].get(
        temperature,
        0.50
    )

    chroma_score = range_score(
        chroma,
        profile["chroma_range"][0],
        profile["chroma_range"][1]
    )

    lightness_score = range_score(
        lightness,
        profile["lightness_range"][0],
        profile["lightness_range"][1]
    )

    family_score = profile["preferred_families"].get(
        family,
        0.45
    )

    return (
        temperature_score * 0.30
        + chroma_score * 0.25
        + lightness_score * 0.20
        + family_score * 0.25
    )


def category_compatibility(
    features,
    category,
    season
):
    """
    Apply category-specific palette preferences.

    Clothing:
        Broad palette harmony.

    Makeup:
        More wearable chroma and lightness.

    Accents:
        Allows richer colors and stronger contrast.
    """

    profile = get_season_profile(season)

    chroma = float(features["chroma"])
    lightness = float(features["lab"][0])
    family = get_recommendation_hue_family(features)

    family_score = profile["preferred_families"].get(
        family,
        0.45
    )

    if category == "clothing":

        chroma_score = range_score(
            chroma,
            profile["chroma_range"][0],
            profile["chroma_range"][1]
        )

        lightness_score = range_score(
            lightness,
            profile["lightness_range"][0],
            profile["lightness_range"][1]
        )

        return (
            chroma_score * 0.45
            + lightness_score * 0.35
            + family_score * 0.20
        )

    if category == "makeup":

        wearable_chroma = range_score(
            chroma,
            8,
            42
        )

        wearable_lightness = range_score(
            lightness,
            25,
            82
        )

        return (
            wearable_chroma * 0.45
            + wearable_lightness * 0.30
            + family_score * 0.25
        )

    if category == "accents":

        accent_chroma = range_score(
            chroma,
            12,
            65
        )

        accent_lightness = range_score(
            lightness,
            20,
            90
        )

        return (
            accent_chroma * 0.35
            + accent_lightness * 0.25
            + family_score * 0.40
        )

    return 0.50


def score_color(
    name,
    hex_color,
    profile,
    category="clothing"
):
    """
    Calculate Tinayu's relative palette compatibility.

    This score is NOT:
        - probability
        - model accuracy
        - scientific confidence
        - a guaranteed color-analysis result

    It is a ranking score used to order Tinayu's candidate palette.
    """

    features = get_color_features(
        hex_color
    )

    season = profile.get(
        "heuristics",
        {}
    ).get(
        "suggested_season",
        "Neutral"
    )

    # --------------------------------------------------------
    # 1. Seasonal compatibility
    # --------------------------------------------------------

    season_score = season_compatibility(
        features,
        season
    )

    # --------------------------------------------------------
    # 2. Temperature compatibility
    # --------------------------------------------------------

    season_profile = get_season_profile(
        season
    )

    temperature = classify_color_temperature(
        features
    )

    temperature_score = season_profile[
        "temperature"
    ].get(
        temperature,
        0.50
    )

    # --------------------------------------------------------
    # 3. Category compatibility
    # --------------------------------------------------------

    category_score = category_compatibility(
        features,
        category,
        season
    )

    # --------------------------------------------------------
    # 4. Small skin-harmony component
    #
    # Skin still matters, but it is deliberately not dominant.
    # --------------------------------------------------------

    skin_data = profile.get(
        "skin",
        {}
    )

    skin_rgb = skin_data.get(
        "rgb"
    )

    skin_lab = skin_data.get(
        "lab",
        {}
    )

    if skin_rgb is not None:

        skin_features = {
            "rgb": skin_rgb,
            "lab": np.array([
                float(skin_lab.get("L", 50)),
                float(skin_lab.get("a", 0)),
                float(skin_lab.get("b", 0))
            ])
        }

        skin_chroma = float(
            np.sqrt(
                skin_features["lab"][1] ** 2
                + skin_features["lab"][2] ** 2
            )
        )

        skin_lightness = float(
            skin_features["lab"][0]
        )

        chroma_relationship = 1.0 - min(
            abs(
                features["chroma"]
                - skin_chroma
            ) / 70.0,
            1.0
        )

        lightness_relationship = 1.0 - min(
            abs(
                features["lab"][0]
                - skin_lightness
            ) / 70.0,
            1.0
        )

        skin_harmony = (
            chroma_relationship * 0.55
            + lightness_relationship * 0.45
        )

    else:

        skin_harmony = 0.50

    # --------------------------------------------------------
    # 5. Contrast awareness
    #
    # High-contrast profiles can tolerate more variation.
    # Softer profiles get a mild preference for closer values.
    # --------------------------------------------------------

    contrast_level = profile.get(
        "heuristics",
        {}
    ).get(
        "contrast_level",
        "Unknown"
    )

    candidate_lightness = float(
        features["lab"][0]
    )

    if contrast_level == "High":

        contrast_bonus = range_score(
            candidate_lightness,
            20,
            92
        )

    elif contrast_level == "Low":

        contrast_bonus = range_score(
            candidate_lightness,
            40,
            82
        )

    else:

        contrast_bonus = range_score(
            candidate_lightness,
            30,
            88
        )

    # --------------------------------------------------------
    # 6. Final weighted score
    # --------------------------------------------------------

    raw_score = (
        season_score * 0.40
        + category_score * 0.28
        + temperature_score * 0.12
        + skin_harmony * 0.12
        + contrast_bonus * 0.08
    )

    return float(
        np.clip(
            raw_score,
            0.0,
            1.0
        )
    )


def build_recommendation(
    palette,
    profile,
    category,
    limit=5
):
    """
    Rank palette colors and return diverse recommendations.

    The returned list is always sorted from highest to lowest
    displayed score. Diversity is applied during selection, not
    after the final ranking.
    """

    scored = []

    for name, hex_color in palette:

        compatibility = score_color(
            name,
            hex_color,
            profile,
            category
        )

        features = get_color_features(
            hex_color
        )

        scored.append({
            "name": name,
            "hex": hex_color,
            "compatibility": compatibility,
            "hue": features["hue"],
            "family": get_recommendation_hue_family(
                features
            ),
            "temperature": classify_color_temperature(
                features
            ),
            "chroma_profile": classify_color_chroma(
                features["chroma"]
            )
        })

    if not scored:
        return []

    # --------------------------------------------------------
    # Normalize the underlying compatibility values first.
    # --------------------------------------------------------

    values = [
        item["compatibility"]
        for item in scored
    ]

    minimum = min(values)
    maximum = max(values)

    score_range = (
        maximum - minimum
    )

    # --------------------------------------------------------
    # Greedy diversity selection.
    #
    # At every step choose the strongest remaining candidate,
    # applying only a small penalty when its hue family is already
    # represented twice. This prevents five near-duplicates while
    # still allowing strong colors to remain in the list.
    # --------------------------------------------------------

    remaining = list(scored)
    selected = []

    while remaining and len(selected) < limit:

        best_index = 0
        best_adjusted = -1.0

        family_counts = {}

        for item in selected:
            family = item["family"]
            family_counts[family] = (
                family_counts.get(family, 0)
                + 1
            )

        for index, candidate in enumerate(remaining):

            family = candidate["family"]

            penalty = 0.0

            if family_counts.get(family, 0) >= 2:
                penalty = 0.025

            adjusted = (
                candidate["compatibility"]
                - penalty
            )

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index

        chosen = remaining.pop(
            best_index
        )

        chosen = {
            **chosen,
            "adjusted_score": best_adjusted
        }

        selected.append(
            chosen
        )

    # --------------------------------------------------------
    # Final ordering MUST match the displayed score.
    # --------------------------------------------------------

    selected.sort(
        key=lambda item: item["compatibility"],
        reverse=True
    )

    results = []

    for item in selected:

        raw_score = item["compatibility"]

        if score_range < 1e-9:

            relative_score = 75.0

        else:

            normalized = (
                raw_score - minimum
            ) / score_range

            relative_score = (
                60.0
                + normalized * 35.0
            )

        results.append({
            "name": item["name"],
            "hex": item["hex"],
            "score": round(
                float(
                    np.clip(
                        relative_score,
                        60.0,
                        95.0
                    )
                ),
                1
            )
        })

    # Defensive final sort so the API can never return a lower
    # displayed score before a higher displayed score.
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:limit]


def build_structured_recommendations(
    profile
):
    """
    Build all recommendation categories.

    Keeps the existing API contract used by analyze_image().
    """

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
# PROFILE BUILDING
# ============================================================

def classify_skin_category(
    lab_L
):
    """
    Basic skin-depth category.
    """

    if lab_L < 35:
        return "Deep"

    if lab_L < 50:
        return "Medium"

    if lab_L < 68:
        return "Medium-Light"

    return "Light"


def classify_temperature(
    lab_a,
    lab_b
):
    """
    Estimate skin temperature from LAB.
    """

    if lab_b >= 18 and lab_a >= 8:
        return "Warm"

    if lab_b <= 10 and lab_a <= 8:
        return "Cool"

    return "Neutral"


def classify_skin_saturation(
    chroma
):
    """
    Estimate skin color saturation.
    """

    if chroma >= 30:
        return "Clear"

    if chroma >= 18:
        return "Moderate"

    return "Muted"


def classify_contrast(
    skin_rgb,
    hair_rgb=None,
    eye_rgb=None
):
    """
    Estimate visual contrast using LAB lightness differences.

    LAB lightness is used instead of raw RGB distance because RGB
    distance can exaggerate hue differences (for example a blue
    background accidentally detected as hair).
    """

    skin_lab = rgb_to_lab(
        skin_rgb
    )

    contrasts = {}

    if hair_rgb is not None:

        hair_lab = rgb_to_lab(
            hair_rgb
        )

        contrasts["skin_hair"] = float(
            abs(
                skin_lab[0]
                -
                hair_lab[0]
            )
        )

    if eye_rgb is not None:

        eye_lab = rgb_to_lab(
            eye_rgb
        )

        contrasts["skin_eye"] = float(
            abs(
                skin_lab[0]
                -
                eye_lab[0]
            )
        )

    if (
        hair_rgb is not None
        and
        eye_rgb is not None
    ):

        hair_lab = rgb_to_lab(
            hair_rgb
        )

        eye_lab = rgb_to_lab(
            eye_rgb
        )

        contrasts["hair_eye"] = float(
            abs(
                hair_lab[0]
                -
                eye_lab[0]
            )
        )

    if not contrasts:
        return (
            {},
            "Unknown"
        )

    max_contrast = max(
        contrasts.values()
    )

    if max_contrast < 20:
        level = "Low"

    elif max_contrast < 40:
        level = "Moderate"

    else:
        level = "High"

    return (
        {
            key: round(
                float(value),
                2
            )
            for key, value in contrasts.items()
        },
        level
    )


def suggest_season(
    temperature,
    saturation,
    skin_depth
):
    """
    Estimate a broad seasonal palette.

    This is a heuristic starting point,
    not a definitive diagnosis.
    """

    if temperature == "Warm":

        if saturation == "Clear":

            return "Warm Spring"

        if saturation == "Muted":

            return "Warm Autumn"

        if skin_depth in [
            "Deep",
            "Medium"
        ]:

            return "Warm Autumn"

        return "Warm Spring"

    if temperature == "Cool":

        if saturation == "Clear":

            return "Cool Winter"

        return "Cool Summer"

    if saturation == "Clear":

        return "Spring"

    if saturation == "Muted":

        return "Autumn"

    return "Neutral"


def build_automatic_profile(
    skin_rgb,
    hair_rgb=None,
    eye_rgb=None
):
    """
    Build structured color profile.
    """

    skin_rgb = np.asarray(
        skin_rgb,
        dtype=np.uint8
    )

    skin_lab = rgb_to_lab(
        skin_rgb
    )

    skin_hsv = rgb_to_hsv_features(
        skin_rgb
    )

    skin_chroma = float(
        np.sqrt(
            skin_lab[1] ** 2
            +
            skin_lab[2] ** 2
        )
    )

    skin_category = (
        classify_skin_category(
            skin_lab[0]
        )
    )

    temperature = (
        classify_temperature(
            skin_lab[1],
            skin_lab[2]
        )
    )

    saturation = (
        classify_skin_saturation(
            skin_chroma
        )
    )

    contrast_values, contrast_level = (
        classify_contrast(
            skin_rgb,
            hair_rgb,
            eye_rgb
        )
    )

    season = suggest_season(
        temperature,
        saturation,
        skin_category
    )

    profile = {

        "skin": {

            "rgb": skin_rgb.astype(
                int
            ).tolist(),

            "lab": {

                "L": round(
                    float(
                        skin_lab[0]
                    ),
                    2
                ),

                "a": round(
                    float(
                        skin_lab[1]
                    ),
                    2
                ),

                "b": round(
                    float(
                        skin_lab[2]
                    ),
                    2
                )
            },

            "hue": round(
                float(
                    skin_hsv["hue"]
                ),
                2
            ),

            "chroma": round(
                skin_chroma,
                2
            )
        },

        "heuristics": {

            "skin_category":
                skin_category,

            "temperature":
                temperature,

            "suggested_season":
                season,

            "skin_depth":
                skin_category,

            "skin_saturation":
                saturation,

            "contrast_level":
                contrast_level
        },

        "interpretation": {

            "skin_depth":
                skin_category.lower(),

            "skin_saturation":
                saturation.lower(),

            "contrast_level":
                contrast_level.lower()
        },

        "contrast":
            contrast_values
    }

    if hair_rgb is not None:

        hair_rgb_array = np.asarray(
            hair_rgb,
            dtype=np.uint8
        )

        hair_lab = rgb_to_lab(
            hair_rgb_array
        )

        profile["hair"] = {

            "rgb":
                hair_rgb_array.astype(
                    int
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

    if eye_rgb is not None:

        eye_rgb_array = np.asarray(
            eye_rgb,
            dtype=np.uint8
        )

        eye_lab = rgb_to_lab(
            eye_rgb_array
        )

        profile["eye"] = {

            "rgb":
                eye_rgb_array.astype(
                    int
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
# IMAGE QUALITY
# ============================================================

def calculate_image_confidence(
    face_confidence,
    skin_pixels,
    hair_detected,
    eyes_detected
):
    """
    Calculate a heuristic image-quality score.

    This is NOT model accuracy.
    """

    score = (
        face_confidence
        * 70.0
    )

    if skin_pixels >= 300:
        score += 15.0

    elif skin_pixels >= 150:
        score += 10.0

    elif skin_pixels >= 75:
        score += 5.0

    if hair_detected:
        score += 7.5

    if eyes_detected:
        score += 7.5

    return round(
        float(
            np.clip(
                score,
                0.0,
                100.0
            )
        ),
        2
    )


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_image(
    image_bytes
):
    """
    Complete Tinayu image analysis pipeline.
    """

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    image_rgb = decode_image(
        image_bytes
    )

    height, width = (
        image_rgb.shape[:2]
    )

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    face_result = detect_face(
        image_rgb
    )

    face_confidence = float(
        face_result
        .detections[0]
        .categories[0]
        .score
    )

    # --------------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------------

    landmarks = detect_landmarks(
        image_rgb
    )

    landmark_count = len(
        landmarks
    )

    # --------------------------------------------------------
    # RAW COLORS
    # --------------------------------------------------------

    raw_skin_rgb = (
        extract_skin_color(
            image_rgb,
            landmarks
        )
    )

    hair_rgb = (
        extract_hair_color(
            image_rgb,
            landmarks
        )
    )

    eye_rgb = (
        extract_eye_color(
            image_rgb,
            landmarks
        )
    )

    # --------------------------------------------------------
    # SKIN NORMALIZATION
    # --------------------------------------------------------

    normalized_skin = (
        normalize_skin_profile(
            raw_skin_rgb
        )
    )

    normalized_skin_rgb = (
        normalized_skin["rgb"]
        .astype(int)
        .tolist()
    )

    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    profile = build_automatic_profile(
        normalized_skin_rgb,
        hair_rgb,
        eye_rgb
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = (
        build_structured_recommendations(
            profile
        )
    )

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    skin_pixels = 0

    # Estimate usable skin pixels
    # using cheek masks again.

    points = landmarks_to_pixels(
        landmarks,
        image_rgb.shape
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

    left_mask = polygon_mask(
        image_rgb.shape,
        points[left_indices]
    )

    right_mask = polygon_mask(
        image_rgb.shape,
        points[right_indices]
    )

    combined_mask = cv2.bitwise_or(
        left_mask,
        right_mask
    )

    skin_pixels = int(
        np.sum(
            combined_mask > 0
        )
    )

    quality_score = (
        calculate_image_confidence(
            face_confidence,
            skin_pixels,
            hair_rgb is not None,
            eye_rgb is not None
        )
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "success": True,

        "image": {

            "width": width,

            "height": height
        },

        "face": {

            "confidence":
                round(
                    face_confidence,
                    4
                ),

            "landmarks":
                landmark_count
        },

        "quality": {

            "image_confidence":
                quality_score,

            "hair_detected":
                hair_rgb is not None,

            "eyes_detected":
                eye_rgb is not None,

            "normalization_applied":
                True,

            "skin_pixels_analyzed":
                skin_pixels
        },

        "colors": {

            "skin_raw_rgb":
                raw_skin_rgb,

            "skin_normalized_rgb":
                normalized_skin_rgb,

            "skin_lab": {

                "L":
                    profile[
                        "skin"
                    ][
                        "lab"
                    ][
                        "L"
                    ],

                "a":
                    profile[
                        "skin"
                    ][
                        "lab"
                    ][
                        "a"
                    ],

                "b":
                    profile[
                        "skin"
                    ][
                        "lab"
                    ][
                        "b"
                    ]
            },

            "hair_rgb":
                hair_rgb,

            "eye_rgb":
                eye_rgb
        },

        "profile":
            profile,

        "recommendations":
            recommendations
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_path = os.path.join(
        BASE_DIR,
        "test.jpg"
    )

    if not os.path.exists(
        test_path
    ):

        print(
            "No test.jpg found."
        )

    else:

        print(
            "Running Tinayu local test..."
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
            "\n========== TINAYU ANALYSIS ==========\n"
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
            "Raw skin RGB:",
            result["colors"][
                "skin_raw_rgb"
            ]
        )

        print(
            "Normalized skin RGB:",
            result["colors"][
                "skin_normalized_rgb"
            ]
        )

        print(
            "Skin LAB:",
            result["colors"][
                "skin_lab"
            ]
        )

        print(
            "Season:",
            result["profile"][
                "heuristics"
            ][
                "suggested_season"
            ]
        )

        print(
            "\nClothing:"
        )

        for item in result[
            "recommendations"
        ][
            "clothing"
        ]:

            print(
                " ",
                item
            )

        print(
            "\nMakeup:"
        )

        for item in result[
            "recommendations"
        ][
            "makeup"
        ]:

            print(
                " ",
                item
            )

        print(
            "\nAccents:"
        )

        for item in result[
            "recommendations"
        ][
            "accents"
        ]:

            print(
                " ",
                item
            )

        print(
            "\n=====================================\n"
        )   