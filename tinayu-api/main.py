import logging
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from tinayu_engine import analyze_image
from llama_explainer import generate_explanation


# ============================================================
# LOGGING AND UPLOAD CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tinayu")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="Tinayu API",
    description="Backend API for Tinayu Personal Color Analysis AI",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Tinayu API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
# ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return {
                "success": False,
                "error": "Uploaded file has no filename.",
            }

        allowed_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }

        if file.content_type not in allowed_types:
            return {
                "success": False,
                "error": (
                    "Unsupported image format. "
                    "Please upload a JPG, PNG, or WebP image."
                ),
            }

        # Read at most 10 MB plus one byte to detect oversized uploads.
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

        if not image_bytes:
            return {
                "success": False,
                "error": "The uploaded image is empty.",
            }

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return {
                "success": False,
                "error": "Image is too large. Maximum size is 10 MB.",
            }

        # Run the existing deterministic color-analysis engine.
        result = analyze_image(image_bytes)

        # Llama is optional; its failure must not break color analysis.
        if result.get("success"):
            try:
                result["explanation"] = generate_explanation(
                    result.get("profile", {}),
                    result.get("recommendations", {}),
                )
            except Exception:
                logger.exception("Tinayu explanation generation failed")
                result["explanation"] = None

        result["filename"] = file.filename
        return result

    except ValueError as error:
        return {
            "success": False,
            "error": str(error),
        }

    except Exception:
        logger.exception("Unexpected error during image analysis")
        return {
            "success": False,
            "error": "Unexpected server error. Please try again.",
        }

    finally:
        await file.close()
