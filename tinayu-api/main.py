from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from tinayu_engine import analyze_image
from llama_explainer import generate_explanation


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="Tinayu API",
    description="Backend API for Tinayu Personal Color Analysis AI",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Tinayu API",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:

        # ------------------------------------------------------
        # Validate upload
        # ------------------------------------------------------

        if not file:
            return {
                "success": False,
                "error": "No image file was provided."
            }

        if not file.filename:
            return {
                "success": False,
                "error": "Uploaded file has no filename."
            }

        # ------------------------------------------------------
        # Validate file type
        # ------------------------------------------------------

        allowed_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/jpg"
        }

        if file.content_type not in allowed_types:
            return {
                "success": False,
                "error": (
                    "Unsupported image format. "
                    "Please upload a JPG, PNG, or WebP image."
                )
            }

        # ------------------------------------------------------
        # Read uploaded image
        # ------------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            return {
                "success": False,
                "error": "The uploaded image is empty."
            }

        # ------------------------------------------------------
        # Run Tinayu analysis engine
        # ------------------------------------------------------

        result = analyze_image(
            image_bytes
        )

        # ------------------------------------------------------
        # Generate Llama explanation
        # ------------------------------------------------------

        if result.get("success"):

            try:

                result["explanation"] = generate_explanation(
                    result.get("profile", {}),
                    result.get("recommendations", {})
                )

            except Exception:

                # Llama failure should never break
                # the main Tinayu color analysis.
                result["explanation"] = None

        # ------------------------------------------------------
        # Add upload information
        # ------------------------------------------------------

        result["filename"] = file.filename

        # ------------------------------------------------------
        # Return complete Tinayu analysis
        # ------------------------------------------------------

        return result

    except ValueError as error:

        return {
            "success": False,
            "error": str(error)
        }

    except Exception as error:

        return {
            "success": False,
            "error": "Unexpected server error.",
            "details": str(error)
        }