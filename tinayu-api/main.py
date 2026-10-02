from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from tinayu_engine import analyze_image


app = FastAPI(
    title="Tinayu API",
    description="Backend API for Tinayu Personal Color Analysis AI",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Tinayu API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:

        # -----------------------------
        # Read uploaded image
        # -----------------------------

        image_bytes = await file.read()

        # -----------------------------
        # Run Tinayu analysis engine
        # -----------------------------

        result = analyze_image(
            image_bytes
        )

        # -----------------------------
        # Add upload information
        # -----------------------------

        result["filename"] = file.filename

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