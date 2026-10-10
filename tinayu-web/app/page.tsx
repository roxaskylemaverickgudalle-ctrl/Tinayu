"use client";

import Image from "next/image";
import Link from "next/link";
import {
  ChangeEvent,
  ClipboardEvent,
  DragEvent,
  useEffect,
  useRef,
  useState,
} from "react";

const TINAYU_API_URL = process.env.NEXT_PUBLIC_TINAYU_API_URL || "http://127.0.0.1:8000";

type ColorRecommendation = {
  name: string;
  hex: string;
  score: number;
};

type AnalysisData = {
  success: boolean;
  filename?: string;

  image?: {
    width: number;
    height: number;
  };

  face?: {
    confidence: number;
    landmarks: number;
  };

  quality?: {
    image_confidence: number;
    hair_detected: boolean;
    eyes_detected: boolean;
    skin_pixels_analyzed: number;
    normalization_applied?: boolean;
  };

  colors?: {
    skin_raw_rgb: number[];
    skin_normalized_rgb: number[];
    skin_lab: {
      L: number;
      a: number;
      b: number;
    };
    hair_rgb?: number[] | null;
    eye_rgb?: number[] | null;
  };

  profile?: {
    skin?: {
      rgb: number[];
      lab: {
        L: number;
        a: number;
        b: number;
      };
      hue: number;
      chroma: number;
    };

    heuristics?: {
      skin_category: string;
      temperature: string;
      suggested_season: string;
      skin_depth: string;
      skin_saturation: string;
      contrast_level: string;
    };

    interpretation?: {
      skin_depth: string;
      skin_saturation: string;
      contrast_level: string;
    };

    contrast?: {
      skin_hair: number;
      skin_eye: number;
      hair_eye: number;
    };

    hair?: {
      rgb: number[];
      lab: {
        L: number;
        a: number;
        b: number;
      };
    } | null;

    eye?: {
      rgb: number[];
      lab: {
        L: number;
        a: number;
        b: number;
      };
    } | null;
  };

  recommendations?: {
    clothing: ColorRecommendation[];
    makeup: ColorRecommendation[];
    accents: ColorRecommendation[];
  };

  explanation?: {
    profile: string;
    why: string;
    color_direction: string;
  };

  normalization?: {
    enabled: boolean;
    method: string;
    raw_skin_rgb: number[];
    normalized_skin_rgb: number[];
  };

  error?: string;
  details?: string;
};

function rgbToCss(rgb: number[] | undefined | null) {
  if (!rgb || rgb.length < 3) {
    return "rgb(200, 200, 200)";
  }

  return (
    "rgb(" +
    rgb[0] +
    ", " +
    rgb[1] +
    ", " +
    rgb[2] +
    ")"
  );
}

function formatScore(score: number | undefined) {
  if (typeof score !== "number") {
    return "—";
  }

  return Math.round(score) + "/100";
}

function formatRgb(rgb: number[] | undefined | null) {
  if (!rgb || rgb.length < 3) {
    return "Not detected";
  }

  return rgb[0] + ", " + rgb[1] + ", " + rgb[2];
}

function getInitials(text: string) {
  if (!text) {
    return "T";
  }

  return text
    .split(" ")
    .map(function (word) {
      return word.charAt(0);
    })
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function ColorSection({
  title,
  description,
  colors,
}: {
  title: string;
  description: string;
  colors: ColorRecommendation[];
}) {
  return (
    <section className="space-y-5">
      <div>
        <h3 className="text-xl font-semibold tracking-tight text-stone-900">
          {title}
        </h3>

        <p className="mt-1 text-sm text-stone-500">
          {description}
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {colors.map(function (color, index) {
          return (
            <div
              key={color.name + "-" + index}
              className="group overflow-hidden rounded-2xl border border-stone-200 bg-white transition hover:-translate-y-1 hover:shadow-lg"
            >
              <div
                className="h-28 w-full"
                style={{
                  backgroundColor: color.hex,
                }}
              />

              <div className="space-y-2 p-4">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-semibold text-stone-900">
                      {color.name}
                    </p>

                    <p className="mt-1 text-xs text-stone-400">
                      {color.hex}
                    </p>
                  </div>

                  <span className="rounded-full bg-stone-100 px-2.5 py-1 text-xs font-semibold text-stone-700">
                    {formatScore(color.score)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CameraIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M4 7.5C4 6.67157 4.67157 6 5.5 6H8L9.2 4.5H14.8L16 6H18.5C19.3284 6 20 6.67157 20 7.5V17.5C20 18.3284 19.3284 19 18.5 19H5.5C4.67157 19 4 18.3284 4 17.5V7.5Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      <circle
        cx="12"
        cy="13"
        r="3.5"
        stroke="currentColor"
        strokeWidth="1.7"
      />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      width="26"
      height="26"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M12 16V4"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />

      <path
        d="M7 9L12 4L17 9"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      <path
        d="M5 20H19"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M12 3L13.4 8.6L19 10L13.4 11.4L12 17L10.6 11.4L5 10L10.6 8.6L12 3Z"
        fill="currentColor"
      />

      <path
        d="M19 15L19.7 17.3L22 18L19.7 18.7L19 21L18.3 18.7L16 18L18.3 17.3L19 15Z"
        fill="currentColor"
      />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M5 12.5L9.5 17L19 7.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function XIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M6 6L18 18"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />

      <path
        d="M18 6L6 18"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function AnalysisJourney({
  analysis,
}: {
  analysis: AnalysisData;
}) {
  const hasFace =
    typeof analysis.face?.confidence === "number";

  const hasLandmarks =
    typeof analysis.face?.landmarks === "number" &&
    analysis.face.landmarks > 0;

  const hasSkin =
    !!analysis.colors?.skin_normalized_rgb &&
    analysis.colors.skin_normalized_rgb.length >= 3;

  const hasNormalization =
    analysis.quality?.normalization_applied === true ||
    analysis.normalization?.enabled === true;

  const hasProfile =
    !!analysis.profile?.heuristics?.suggested_season;

  const hasRecommendations =
    !!analysis.recommendations &&
    (
      analysis.recommendations.clothing.length > 0 ||
      analysis.recommendations.makeup.length > 0 ||
      analysis.recommendations.accents.length > 0
    );

  const steps = [
    {
      title: "Face detected",
      description: hasFace
        ? "Tinayu located a face in your photo."
        : "A face could not be confirmed.",
      complete: hasFace,
    },
    {
      title: "Facial landmarks mapped",
      description: hasLandmarks
        ? analysis.face?.landmarks +
          " facial landmarks were detected."
        : "Facial landmarks were not available.",
      complete: hasLandmarks,
    },
    {
      title: "Skin color extracted",
      description: hasSkin
        ? "Skin-region pixels were sampled for color analysis."
        : "Skin color could not be extracted.",
      complete: hasSkin,
    },
    {
      title: "Lighting normalized",
      description: hasNormalization
        ? "LAB lightness normalization was applied."
        : "Lighting normalization was not applied.",
      complete: hasNormalization,
    },
    {
      title: "Color profile generated",
      description: hasProfile
        ? "Temperature, depth, saturation, and contrast were evaluated."
        : "A complete color profile was not generated.",
      complete: hasProfile,
    },
    {
      title: "Personalized palette matched",
      description: hasRecommendations
        ? "Clothing, makeup, and accent colors were ranked."
        : "Palette recommendations were not generated.",
      complete: hasRecommendations,
    },
  ];

  return (
    <div className="rounded-[2rem] border border-stone-200 bg-white p-7 shadow-sm sm:p-9">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
            Your analysis
          </p>

          <h2 className="mt-2 text-3xl font-semibold tracking-tight text-stone-950">
            How Tinayu read your photo
          </h2>
        </div>

        <div className="rounded-full bg-stone-100 px-3 py-1.5 text-xs font-medium text-stone-500">
          Computer vision pipeline
        </div>
      </div>

      <div className="mt-8">
        {steps.map(function (step, index) {
          const isLast = index === steps.length - 1;

          return (
            <div
              key={step.title}
              className="relative flex gap-4"
            >
              {!isLast && (
                <div className="absolute left-[15px] top-9 h-[calc(100%-18px)] w-px bg-stone-200" />
              )}

              <div
                className={
                  "relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition " +
                  (step.complete
                    ? "bg-stone-900 text-white"
                    : "bg-stone-100 text-stone-300")
                }
              >
                {step.complete ? (
                  <CheckIcon />
                ) : (
                  <span className="h-2 w-2 rounded-full bg-current" />
                )}
              </div>

              <div
                className={
                  "pb-7 " +
                  (isLast ? "pb-0" : "")
                }
              >
                <div className="flex flex-wrap items-center gap-2">
                  <h3
                    className={
                      "text-sm font-semibold " +
                      (step.complete
                        ? "text-stone-900"
                        : "text-stone-400")
                    }
                  >
                    {step.title}
                  </h3>

                  {step.complete && (
                    <span className="rounded-full bg-stone-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-stone-500">
                      Complete
                    </span>
                  )}
                </div>

                <p className="mt-1 text-sm leading-6 text-stone-500">
                  {step.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function Home() {
  useEffect(function () {
    try {
      const stored = sessionStorage.getItem("tinayu-scan-result");
      if (!stored) return;
      const payload = JSON.parse(stored) as { analysis?: AnalysisData; image?: string };
      if (payload.analysis) {
        setAnalysis(payload.analysis);
        setShowResults(true);
      }
      if (payload.image) setImage(payload.image);
      sessionStorage.removeItem("tinayu-scan-result");
    } catch (storageError) {
      console.error("Could not restore Tinayu scan results:", storageError);
      sessionStorage.removeItem("tinayu-scan-result");
    }
  }, []);

  const [guideCardIndex, setGuideCardIndex] = useState(0);

  const [cameraMode, setCameraMode] = useState<"upload" | "camera">(
    "upload"
  );

  const [image, setImage] = useState<string | null>(null);

  const [imageFile, setImageFile] = useState<File | null>(null);

  const [showResults, setShowResults] = useState(false);

  const [analyzing, setAnalyzing] = useState(false);

  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);

  const [error, setError] = useState<string | null>(null);

  const [dragActive, setDragActive] = useState(false);

  const [cameraStarting, setCameraStarting] = useState(false);

  const [cameraActive, setCameraActive] = useState(false);

  const [cameraError, setCameraError] = useState<string | null>(null);

  const [scanActive, setScanActive] = useState(false);

  const [scanStep, setScanStep] = useState(0);

  const [scanStatus, setScanStatus] = useState(
    "Position your face inside the guide."
  );

  const [scanProgress, setScanProgress] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const streamRef = useRef<MediaStream | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  function releaseCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(function (track) {
        track.stop();
      });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function stopCamera() {
    releaseCamera();
    setCameraActive(false);
  }

  async function startCamera() {
    setCameraError(null);
    setError(null);

    if (!navigator.mediaDevices) {
      setCameraError(
        "Your browser does not provide camera access. Try Chrome or Edge on localhost."
      );
      return;
    }

    if (!navigator.mediaDevices.getUserMedia) {
      setCameraError(
        "Camera access is not supported by this browser."
      );
      return;
    }

    stopCamera();

    setCameraStarting(true);

    try {
      const devices =
        await navigator.mediaDevices.enumerateDevices();

      const videoDevices = devices.filter(function (device) {
        return device.kind === "videoinput";
      });

      if (videoDevices.length === 0) {
        throw new Error(
          "No webcam was detected. Please make sure your laptop camera is enabled."
        );
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: {
              ideal: 1280,
            },
            height: {
              ideal: 720,
            },
          },
          audio: false,
        });

      streamRef.current = stream;

      if (!videoRef.current) {
        stream.getTracks().forEach(function (track) {
          track.stop();
        });

        streamRef.current = null;

        throw new Error(
          "The camera opened, but the video element was not ready."
        );
      }

      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      setCameraActive(true);
    } catch (cameraStartError) {
      console.error(
        "Tinayu camera error:",
        cameraStartError
      );

      let message =
        "Unable to access your camera.";

      if (cameraStartError instanceof DOMException) {
        if (
          cameraStartError.name ===
          "NotAllowedError"
        ) {
          message =
            "Camera permission was blocked. Click the camera icon in your browser address bar and allow camera access, then try again.";
        } else if (
          cameraStartError.name ===
          "NotFoundError"
        ) {
          message =
            "No camera was found. Make sure your laptop webcam is connected and enabled.";
        } else if (
          cameraStartError.name ===
          "NotReadableError"
        ) {
          message =
            "Your camera is already being used by another app. Close apps like Camera, Zoom, Teams, or Discord and try again.";
        } else if (
          cameraStartError.name ===
          "SecurityError"
        ) {
          message =
            "The browser blocked camera access for security reasons. Open Tinayu through http://localhost:3000.";
        } else if (
          cameraStartError.name ===
          "AbortError"
        ) {
          message =
            "Camera startup was interrupted. Please try starting the camera again.";
        } else {
          message =
            cameraStartError.message ||
            "Unable to access your camera.";
        }
      } else if (
        cameraStartError instanceof Error
      ) {
        message = cameraStartError.message;
      }

      setCameraError(message);
      setCameraActive(false);
    } finally {
      setCameraStarting(false);
    }
  }

  function switchToUpload() {
    stopCamera();

    setCameraMode("upload");
    setCameraError(null);
  }

  function switchToCamera() {
    setCameraMode("camera");
    setCameraError(null);
    setError(null);
  }

  function handleFile(file: File | null) {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please select an image file.");
      return;
    }

    stopCamera();

    const imageUrl =
      URL.createObjectURL(file);

    setImageFile(file);
    setImage(imageUrl);
    setShowResults(false);
    setAnalysis(null);
    setError(null);
    setCameraError(null);
  }

  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>
  ) {
    const file =
      event.target.files?.[0] || null;

    handleFile(file);
  }

  function handleDrop(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    setDragActive(false);

    const file =
      event.dataTransfer.files?.[0] || null;

    handleFile(file);
  }

  function handleDragOver(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    setDragActive(true);
  }

  function handleDragLeave(
    event: DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();

    setDragActive(false);
  }

  function handlePaste(
    event: ClipboardEvent<HTMLDivElement>
  ) {
    const items =
      event.clipboardData?.items;

    if (!items) {
      return;
    }

    for (let i = 0; i < items.length; i++) {
      const item = items[i];

      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();

        if (file) {
          handleFile(file);
        }

        break;
      }
    }
  }

  async function capturePhoto() {
    if (!videoRef.current || !cameraActive) {
      setCameraError("Please start the camera first.");
      return;
    }

    if (videoRef.current.readyState < 2) {
      setCameraError(
        "The camera is still starting. Please wait a moment and try again."
      );
      return;
    }

    const canvas = canvasRef.current;

    if (!canvas) {
      setCameraError("Camera capture is unavailable.");
      return;
    }

    setCameraError(null);
    setScanActive(true);
    setScanStep(1);
    setScanProgress(0);
    setScanStatus("Look straight at the camera and hold still.");

    const capturedFrames: Blob[] = [];

    try {
      for (let step = 0; step < 3; step++) {
        setScanStep(step + 1);

        if (step === 0) {
          setScanStatus("Look straight at the camera and hold still.");
        } else if (step === 1) {
          setScanStatus("Keep your face centered and relax your expression.");
        } else {
          setScanStatus("Hold still while Tinayu captures the final frame.");
        }

        await new Promise<void>(function (resolve) {
          window.setTimeout(resolve, 900);
        });

        const video = videoRef.current;

        if (!video) {
          throw new Error("Camera preview was lost.");
        }

        const width = video.videoWidth || 1280;
        const height = video.videoHeight || 720;

        canvas.width = width;
        canvas.height = height;

        const context = canvas.getContext("2d");

        if (!context) {
          throw new Error("Unable to capture the camera frame.");
        }

        context.drawImage(video, 0, 0, width, height);

        const blob = await new Promise<Blob | null>(function (resolve) {
          canvas.toBlob(resolve, "image/jpeg", 0.92);
        });

        if (!blob) {
          throw new Error("Tinayu could not capture one of the scan frames.");
        }

        capturedFrames.push(blob);
        setScanProgress(Math.round(((step + 1) / 3) * 100));
      }

      setScanStatus("Analyzing your scan...");

      const frameResults: Array<{
        analysis: AnalysisData;
        blob: Blob;
      }> = [];

      for (let i = 0; i < capturedFrames.length; i++) {
        const formData = new FormData();
        formData.append(
          "file",
          new File([capturedFrames[i]], `tinayu-scan-${i + 1}.jpg`, {
            type: "image/jpeg",
          })
        );

        try {
          const response = await fetch(`${TINAYU_API_URL}/analyze`, {
            method: "POST",
            body: formData,
          });

          const data: AnalysisData = await response.json();

          if (response.ok && data.success) {
            frameResults.push({
              analysis: data,
              blob: capturedFrames[i],
            });
          }
        } catch (frameError) {
          console.warn(`Tinayu scan frame ${i + 1} failed:`, frameError);
        }
      }

      if (frameResults.length === 0) {
        throw new Error(
          "Tinayu couldn't get a reliable facial reading. Face the camera directly, move into even lighting, and try again."
        );
      }

      const bestFrame = frameResults.reduce(function (
        best: { analysis: AnalysisData; blob: Blob },
        current: { analysis: AnalysisData; blob: Blob }
      ) {
        const bestScore = best.analysis.quality?.image_confidence ?? 0;
        const currentScore = current.analysis.quality?.image_confidence ?? 0;
        return currentScore > bestScore ? current : best;
      });

      setAnalysis(bestFrame.analysis);

      const bestFile = new File(
        [bestFrame.blob],
        "tinayu-smart-scan.jpg",
        { type: "image/jpeg" }
      );

      setImageFile(bestFile);

      const bestImageUrl = URL.createObjectURL(bestFrame.blob);
      setImage(function (previousImage) {
        if (previousImage) {
          URL.revokeObjectURL(previousImage);
        }
        return bestImageUrl;
      });

      setShowResults(true);
      setScanStatus(
        `Scan complete — ${frameResults.length}/3 frames successfully analyzed.`
      );

      stopCamera();
      setCameraMode("upload");

      window.setTimeout(function () {
        const resultElement = document.getElementById("results");
        if (resultElement) {
          resultElement.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      }, 100);
    } catch (scanError) {
      console.error("Tinayu smart scan error:", scanError);
      setCameraError(
        scanError instanceof Error
          ? scanError.message
          : "Something went wrong during the smart scan."
      );
    } finally {
      setScanActive(false);
      setScanProgress(0);
      setScanStep(0);
    }
  }

  async function handleAnalyze() {
    if (!imageFile) {
      setError(
        "Please upload or capture a photo first."
      );
      return;
    }

    setAnalyzing(true);
    setError(null);
    setShowResults(false);

    try {
      const formData =
        new FormData();

      formData.append(
        "file",
        imageFile
      );

      const response =
        await fetch(
          `${TINAYU_API_URL}/analyze`,
          {
            method: "POST",
            body: formData,
          }
        );

      if (!response.ok) {
        throw new Error(
          "API request failed with status " +
            response.status +
            "."
        );
      }

      const data: AnalysisData =
        await response.json();

      console.log(
        "Tinayu API result:",
        data
      );

      if (!data.success) {
        throw new Error(
          data.error ||
            "Tinayu could not analyze this image."
        );
      }

      setAnalysis(data);
      setShowResults(true);

      window.setTimeout(
        function () {
          const resultElement =
            document.getElementById(
              "results"
            );

          if (resultElement) {
            resultElement.scrollIntoView(
              {
                behavior: "smooth",
                block: "start",
              }
            );
          }
        },
        100
      );
    } catch (analysisError) {
      console.error(
        "Tinayu analysis error:",
        analysisError
      );

      setError(
        analysisError instanceof Error
          ? analysisError.message
          : "Something went wrong while analyzing the image."
      );
    } finally {
      setAnalyzing(false);
    }
  }

  function resetAnalysis() {
    stopCamera();

    if (image) {
      URL.revokeObjectURL(image);
    }

    setImage(null);
    setImageFile(null);
    setAnalysis(null);
    setShowResults(false);
    setError(null);
    setCameraError(null);
    setCameraMode("upload");
    setScanActive(false);
    setScanStep(0);
    setScanProgress(0);
    setScanStatus("Position your face inside the guide.");

    if (fileInputRef.current) {
      fileInputRef.current.value =
        "";
    }

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  const handleFileRef = useRef(handleFile);

  useEffect(function () {
    handleFileRef.current = handleFile;
  });

  useEffect(function () {
    function handleGlobalPaste(
      event: ClipboardEvent
    ) {
      const items =
        event.clipboardData?.items;

      if (!items) {
        return;
      }

      for (
        let i = 0;
        i < items.length;
        i++
      ) {
        const item = items[i];

        if (
          item.type.startsWith(
            "image/"
          )
        ) {
          const file =
            item.getAsFile();

          if (file) {
            handleFileRef.current(file);
          }

          break;
        }
      }
    }

    window.addEventListener(
      "paste",
      handleGlobalPaste as unknown as EventListener
    );

    return function () {
      window.removeEventListener(
        "paste",
        handleGlobalPaste as unknown as EventListener
      );
    };
  }, []);

  useEffect(function () {
    return function () {
      releaseCamera();

      if (image) {
        URL.revokeObjectURL(image);
      }
    };
  }, [image]);

  useEffect(
    function () {
      if (cameraMode === "camera") {
        return;
      }

      releaseCamera();
    },
    [cameraMode]
  );

  const heuristics =
    analysis?.profile?.heuristics;

  const recommendations =
    analysis?.recommendations;

  const quality =
    analysis?.quality;

  const colors =
    analysis?.colors;

  return (
    <main
      className="min-h-screen bg-[#f7f5f0] text-stone-900"
      onPaste={handlePaste}
    >
      <nav className="border-b border-stone-200 bg-[#f7f5f0]/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 sm:px-8">
          <button
            onClick={resetAnalysis}
            className="flex items-center gap-3"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-stone-900 text-white">
              <SparkleIcon />
            </div>

            <div>
              <div className="text-lg font-bold tracking-tight">
                Tinayu
              </div>

              <div className="text-[10px] uppercase tracking-[0.2em] text-stone-400">
                Personal Color AI
              </div>
            </div>
          </button>

          <div className="hidden items-center gap-6 text-sm text-stone-500 sm:flex">
            <span>Computer Vision</span>
            <span>Color Analysis</span>
            <span>AI Recommendations</span>
          </div>
        </div>
      </nav>

      {!showResults && (
        <section className="mx-auto max-w-7xl px-5 pb-24 pt-5 sm:px-8 sm:pt-8 lg:pt-10">
          <div className="grid items-stretch gap-6 lg:grid-cols-2 lg:gap-8">
            <div className="text-center lg:text-left">
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-[#eadbd8] bg-white/80 px-4 py-2 text-xs font-medium tracking-wide text-[#875f63] shadow-sm">
                <SparkleIcon />
                YOUR PERSONAL COLOR DISCOVERY
              </div>

              <h1 className="text-5xl font-semibold leading-[1.08] tracking-[-0.055em] text-stone-950 sm:text-6xl lg:text-7xl">
                Find the colors
                <br />
                that feel like{" "}
                <span className="font-serif italic font-normal text-[#b7797e]">
                  you.
                </span>
              </h1>

              <p className="mx-auto mt-6 max-w-xl text-base leading-7 text-stone-600 sm:text-lg lg:mx-0">
                Get a more guided color analysis with Tinayu Smart Scan.
                Scan your face live for a more interactive experience,
                or upload a photo whenever that works better for you.
              </p>

              <div className="mt-7 flex flex-col items-stretch gap-3 sm:flex-row sm:items-center lg:justify-start">
                <Link
                  href="/scan"
                  className="group relative inline-flex min-h-14 items-center justify-center gap-3 overflow-hidden rounded-2xl bg-stone-950 px-6 py-4 text-sm font-semibold text-white shadow-[0_12px_30px_rgba(28,25,23,0.22)] ring-1 ring-white/10 transition duration-300 hover:-translate-y-0.5 hover:bg-stone-800 hover:shadow-[0_18px_38px_rgba(28,25,23,0.28)] focus:outline-none focus:ring-2 focus:ring-[#bd8587] focus:ring-offset-2">
                  <span className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/[0.09] to-white/0 opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
                  <span className="relative flex h-9 w-9 items-center justify-center rounded-xl border border-white/15 bg-white/10 text-[#e8c6c3]">
                    <CameraIcon />
                  </span>
                  <span className="relative text-left">
                    <span className="block">Scan with camera</span>
                    <span className="mt-0.5 block text-xs font-normal text-stone-300">
                      Guided, real-time color discovery
                    </span>
                  </span>
                  <span aria-hidden="true" className="relative ml-1 text-base text-[#e8c6c3] transition-transform group-hover:translate-x-1">→</span>
                </Link>
                <span className="inline-flex items-center justify-center gap-3 px-2 py-1 text-xs font-medium uppercase tracking-[0.22em] text-stone-400 sm:px-1">
                  <span className="h-px w-8 bg-stone-300 sm:hidden" />
                  or
                  <span className="h-px w-8 bg-stone-300 sm:hidden" />
                </span>
              </div>
            </div>

            <div className="relative mx-auto w-full max-w-md">
              <div className="absolute -left-5 top-10 h-24 w-24 rounded-full bg-[#ead5d2]/70 blur-2xl" />
              <div className="absolute -right-4 bottom-8 h-28 w-28 rounded-full bg-[#d9dfcb]/80 blur-2xl" />

              <div className="relative overflow-hidden rounded-[2rem] border border-white/80 bg-white/80 p-5 shadow-[0_24px_80px_rgba(105,75,67,0.12)] backdrop-blur sm:p-7">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#a77b7b]">
                      The color edit
                    </p>
                    <h2 className="mt-1 text-xl font-semibold tracking-tight text-stone-900">
                      Your palette starts here
                    </h2>
                  </div>
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-[#f4e8e5] text-[#a66f75]">
                    <SparkleIcon />
                  </div>
                </div>

                <div className="mt-7 grid grid-cols-5 gap-2">
                  {["#B96F76", "#D8A89C", "#E8D6BC", "#AAB59A", "#647C70", "#805C68", "#C5A6B4", "#E5C8BE", "#B7B9A1", "#D2B16F"].map((color, index) => (
                    <span key={`${color}-${index}`} className="aspect-square rounded-xl shadow-sm ring-1 ring-black/[0.04]" style={{ backgroundColor: color }} />
                  ))}
                </div>
                <div className="mt-5 flex items-center gap-3 rounded-2xl bg-[#f8f4f0] p-4">
                  <div className="flex -space-x-2" aria-hidden="true">
                    {["#D8A89C", "#AAB59A", "#805C68"].map((color) => <span key={color} className="h-7 w-7 rounded-full border-2 border-[#f8f4f0]" style={{ backgroundColor: color }} />)}
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-stone-800">Made around your colors</p>
                    <p className="mt-0.5 text-xs leading-5 text-stone-500">Start with a photo to explore your palette.</p>
                  </div>
                </div>

              </div>
            </div>

            <div className="flex h-full flex-col rounded-[1.75rem] border border-[#eadbd8] bg-white/85 p-5 shadow-[0_12px_36px_rgba(105,75,67,0.07)] sm:p-6 order-4 lg:order-none lg:col-start-2 lg:row-start-2">
              {(() => {
                  const guideCards = [
                    {
                      eyebrow: "01 · YOUR SEASON",
                      title: "What does your season mean?",
                      description:
                        "A color season groups colors that tend to harmonize with your overall coloring. Spring, Summer, Autumn, and Winter are useful guides—not rules for what you can or cannot wear.",
                      tip: "Use your palette as a starting point, then choose the shades that make you feel most like yourself.",
                      colors: ["#E8C28D", "#D88E79", "#A8B88E", "#719080", "#B96F76"],
                    },
                    {
                      eyebrow: "02 · UNDERTONE",
                      title: "What does neutral mean?",
                      description:
                        "A neutral undertone means your skin may show a balance of warm and cool qualities, rather than leaning strongly toward either one. It does not mean your skin has no undertone.",
                      tip: "Neutral? You may find both warm and cool shades wearable. Compare colors in natural light instead of relying on one test alone.",
                      colors: ["#D9B49B", "#C9B7A5", "#B9B5A0", "#B8A0AA", "#C7A96B"],
                    },
                    {
                      eyebrow: "03 · COLOR HARMONY",
                      title: "What are contrast and depth?",
                      description:
                        "Contrast describes how much light-dark difference your features show together. Depth describes whether lighter or deeper colors tend to complement your overall coloring.",
                      tip: "These are clues—not a score of your appearance. They help you explore combinations that feel balanced.",
                      colors: ["#F0E2D0", "#CDA79D", "#AAB59A", "#805C68", "#35483F"],
                    },
                    {
                      eyebrow: "04 · MEET TINAYU",
                      title: "Your colors, made easier.",
                      description:
                        "Tinayu brings your photo-based analysis and color guidance together so you can better understand the terms and explore shades for clothing, makeup, and personal style.",
                      tip: "Ready to explore? Upload a clear photo or try Smart Scan below. Results are guidance, not a fixed label.",
                      colors: ["#B96F76", "#D8A89C", "#E8D6BC", "#AAB59A", "#647C70"],
                    },
                  ];
                  const activeCard = guideCards[guideCardIndex];
                  return (
                    <div aria-live="polite">
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#a77b7b]">{activeCard.eyebrow}</p>
                        <p className="shrink-0 text-xs font-medium text-stone-400">{guideCardIndex + 1} / {guideCards.length}</p>
                      </div>
                      <h3 className="mt-2 text-lg font-semibold leading-snug tracking-tight text-stone-900 sm:text-xl">{activeCard.title}</h3>
                      <p className="mt-2 text-sm leading-6 text-stone-600">{activeCard.description}</p>
                      <div className="mt-4 flex items-center gap-2" aria-label="Example color palette">
                        {activeCard.colors.map((color) => <span key={color} className="h-7 flex-1 rounded-lg ring-1 ring-black/[0.04]" style={{ backgroundColor: color }} />)}
                      </div>
                      <p className="mt-4 rounded-xl border border-[#eadbd5] bg-[#f8f4f0] p-3 text-xs leading-5 text-stone-600"><span className="font-semibold text-stone-800">A helpful note: </span>{activeCard.tip}</p>
                      <div className="mt-4 flex items-center justify-between gap-2">
                        <button type="button" onClick={() => setGuideCardIndex((current) => (current - 1 + guideCards.length) % guideCards.length)} className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-2.5 text-sm font-semibold text-stone-700 transition hover:border-[#c99a98] hover:bg-[#fffaf8] focus:outline-none focus:ring-2 focus:ring-[#c99a98] focus:ring-offset-2" aria-label="Previous color guide card"><span aria-hidden="true">←</span> Previous</button>
                        <div className="flex gap-1.5" aria-label="Choose guide card">
                          {guideCards.map((card, index) => <button key={card.eyebrow} type="button" onClick={() => setGuideCardIndex(index)} className={`h-2.5 rounded-full transition-all ${index === guideCardIndex ? "w-6 bg-[#a77b7b]" : "w-2.5 bg-stone-300 hover:bg-stone-400"}`} aria-label={`Go to ${card.title}`} aria-current={index === guideCardIndex ? "step" : undefined} />)}
                        </div>
                        <button type="button" onClick={() => setGuideCardIndex((current) => (current + 1) % guideCards.length)} className="inline-flex items-center gap-2 rounded-full bg-stone-900 px-3 py-2.5 text-sm font-semibold text-white transition hover:bg-stone-700 focus:outline-none focus:ring-2 focus:ring-stone-500 focus:ring-offset-2" aria-label="Next color guide card">Next <span aria-hidden="true">→</span></button>
                      </div>
                    </div>
                  );
                })()}
              <div className="mt-5 rounded-2xl border border-[#eadbd5] bg-[#f8f4f0] p-4 sm:p-5">
                <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#a77b7b]">HOW TINAYU WORKS</p>
                <div className="mt-3 grid grid-cols-3 gap-3">
                  <div>
                    <p className="text-sm font-semibold text-stone-900">01 · Choose</p>
                    <p className="mt-1 text-xs leading-5 text-stone-500">Upload a photo or use Smart Scan.</p>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-stone-900">02 · Analyze</p>
                    <p className="mt-1 text-xs leading-5 text-stone-500">Tinayu checks your image and colors.</p>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-stone-900">03 · Explore</p>
                    <p className="mt-1 text-xs leading-5 text-stone-500">Learn about your palette and shades.</p>
                  </div>
                </div>
              </div>
            </div>

          <div className="order-3 lg:order-none lg:col-start-1 lg:row-start-2">
            <div className="h-full rounded-[2rem] border border-stone-200 bg-white p-3 shadow-[0_20px_70px_rgba(28,25,23,0.08)]">
              <div className="p-5 sm:p-8">
                {cameraMode === "upload" && (
                  <div>
                    {!image ? (
                      <div
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onClick={function () {
                          if (
                            fileInputRef.current
                          ) {
                            fileInputRef.current.click();
                          }
                        }}
                        className={
                          "flex min-h-[360px] cursor-pointer flex-col items-center justify-center rounded-[1.5rem] border-2 border-dashed px-6 text-center transition " +
                          (dragActive
                            ? "border-stone-900 bg-stone-50"
                            : "border-stone-200 bg-[#faf9f6] hover:border-stone-400 hover:bg-stone-50")
                        }
                      >
                        <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white text-stone-700 shadow-sm">
                          <UploadIcon />
                        </div>

                        <h2 className="mt-6 text-lg font-semibold">
                          Drop your photo here
                        </h2>

                        <p className="mt-2 text-sm text-stone-400">
                          or click to browse your files
                        </p>

                        <p className="mt-5 text-xs text-stone-400">
                          JPG, JPEG, PNG, or other supported image formats
                        </p>

                        <div className="mt-6 rounded-full bg-stone-100 px-4 py-2 text-xs text-stone-500">
                          You can also paste an image with Ctrl + V
                        </div>

                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={handleFileChange}
                        />
                      </div>
                    ) : (
                      <div className="overflow-hidden rounded-[1.5rem] border border-stone-200 bg-stone-50">
                        <div className="relative">
                          <Image
                            src={image}
                            alt="Selected photo"
                            width={1200}
                            height={900}
                            unoptimized
                            className="max-h-[560px] w-full object-contain"
                            style={{ height: "auto" }}
                          />

                          <button
                            type="button"
                            onClick={resetAnalysis}
                            className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-full bg-white/90 text-stone-700 shadow-lg backdrop-blur transition hover:bg-white"
                            aria-label="Remove photo"
                          >
                            <XIcon />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {cameraMode === "camera" && (
                  <div className="space-y-5">
                    <div className="overflow-hidden rounded-[1.5rem] bg-stone-950">
                      <div className="relative aspect-[4/3] w-full">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="absolute inset-0 h-full w-full object-cover scale-x-[-1]"
                        />

                        {!cameraActive && (
                          <div className="absolute inset-0 flex flex-col items-center justify-center bg-stone-950 px-6 text-center text-white">
                            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white/10">
                              <CameraIcon />
                            </div>

                            <h2 className="mt-5 text-lg font-semibold">
                              Your camera preview will appear here
                            </h2>

                            <p className="mt-2 max-w-md text-sm leading-6 text-white/60">
                              Click the button below to start a guided three-frame scan. Your browser will ask for camera permission if needed.
                            </p>
                          </div>
                        )}

                        {cameraStarting && (
                          <div className="absolute inset-0 flex items-center justify-center bg-stone-950/60 backdrop-blur-sm">
                            <div className="rounded-2xl bg-white px-5 py-4 shadow-xl">
                              <div className="flex items-center gap-3">
                                <div className="h-5 w-5 animate-spin rounded-full border-2 border-stone-200 border-t-stone-900" />

                                <span className="text-sm font-medium text-stone-800">
                                  Starting camera...
                                </span>
                              </div>
                            </div>
                          </div>
                        )}

                        {cameraActive && (
                          <>
                            <div className="pointer-events-none absolute inset-0">
                              <div className="absolute inset-x-[16%] top-[10%] bottom-[10%] rounded-[45%] border-2 border-white/80 shadow-[0_0_0_9999px_rgba(0,0,0,0.18)]" />

                              <div
                                className={
                                  "absolute left-[18%] right-[18%] h-0.5 bg-white/80 shadow-[0_0_12px_rgba(255,255,255,0.8)] transition-all duration-500 " +
                                  (scanActive ? "animate-pulse" : "")
                                }
                                style={{
                                  top: scanActive
                                    ? `${18 + scanProgress * 0.55}%`
                                    : "50%",
                                }}
                              />

                              <div className="absolute left-1/2 top-5 -translate-x-1/2 rounded-full bg-black/55 px-4 py-2 text-center text-xs font-medium text-white backdrop-blur">
                                {scanActive
                                  ? `Step ${scanStep} of 3 · ${scanStatus}`
                                  : "Center your face inside the guide"}
                              </div>
                            </div>

                            <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/50 px-3 py-2 text-xs font-medium text-white backdrop-blur">
                              <span className="h-2 w-2 animate-pulse rounded-full bg-red-400" />
                              {scanActive ? "Scanning" : "Camera active"}
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {cameraError && (
                      <div className="rounded-2xl border border-red-200 bg-red-50 p-4">
                        <div className="flex gap-3">
                          <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-red-100 text-xs font-bold text-red-700">
                            !
                          </div>

                          <div>
                            <p className="text-sm font-semibold text-red-900">
                              Camera problem
                            </p>

                            <p className="mt-1 text-sm leading-6 text-red-700">
                              {cameraError}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="flex flex-col gap-3 sm:flex-row">
                      {!cameraActive ? (
                        <button
                          type="button"
                          onClick={startCamera}
                          disabled={cameraStarting}
                          className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-stone-900 px-6 py-4 text-sm font-semibold text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          <CameraIcon />

                          {cameraStarting
                            ? "Starting camera..."
                            : "Start camera"}
                        </button>
                      ) : (
                        <>
                          <button
                            type="button"
                            onClick={capturePhoto}
                            disabled={scanActive}
                            className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-stone-900 px-6 py-4 text-sm font-semibold text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            {scanActive ? (
                              <>
                                <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                                Scanning {scanStep}/3
                              </>
                            ) : (
                              <>
                                <CameraIcon />
                                Start Smart Scan
                              </>
                            )}
                          </button>

                          <button
                            type="button"
                            onClick={stopCamera}
                            className="rounded-2xl border border-stone-200 bg-white px-6 py-4 text-sm font-semibold text-stone-700 transition hover:bg-stone-50"
                          >
                            Stop camera
                          </button>
                        </>
                      )}
                    </div>

                    {scanActive && (
                      <div className="rounded-2xl border border-stone-200 bg-stone-50 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                            Smart Scan
                          </p>
                          <span className="text-xs font-semibold text-stone-600">
                            {scanProgress}%
                          </span>
                        </div>

                        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-stone-200">
                          <div
                            className="h-full rounded-full bg-stone-900 transition-all duration-500"
                            style={{ width: `${scanProgress}%` }}
                          />
                        </div>

                        <p className="mt-3 text-center text-xs leading-5 text-stone-500">
                          {scanStatus}
                        </p>
                      </div>
                    )}

                    <p className="text-center text-xs leading-5 text-stone-400">
                      Tinayu captures three guided frames, checks each one, and uses the strongest valid frame for your analysis.
                    </p>

                    <canvas
                      ref={canvasRef}
                      className="hidden"
                    />
                  </div>
                )}

                {image && cameraMode === "upload" && (
                  <div className="mt-5">
                    <button
                      type="button"
                      onClick={handleAnalyze}
                      disabled={analyzing}
                      className="flex w-full items-center justify-center gap-3 rounded-2xl bg-stone-900 px-6 py-4 text-sm font-semibold text-white shadow-lg shadow-stone-900/10 transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {analyzing ? (
                        <>
                          <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                          Analyzing your colors...
                        </>
                      ) : (
                        <>
                          <SparkleIcon />
                          Analyze my colors
                        </>
                      )}
                    </button>
                  </div>
                )}

                {error && (
                  <div className="mt-5 rounded-2xl border border-red-200 bg-red-50 p-4">
                    <p className="text-sm font-medium text-red-800">
                      {error}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-7 grid gap-3 sm:grid-cols-3">
              <div className="rounded-2xl border border-stone-200 bg-white p-4 text-center">
                <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                  Step 01
                </p>

                <p className="mt-2 text-sm font-medium">
                  Upload or smart scan
                </p>
              </div>

              <div className="rounded-2xl border border-stone-200 bg-white p-4 text-center">
                <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                  Step 02
                </p>

                <p className="mt-2 text-sm font-medium">
                  Tinayu validates and analyzes
                </p>
              </div>

              <div className="rounded-2xl border border-stone-200 bg-white p-4 text-center">
                <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                  Step 03
                </p>

                <p className="mt-2 text-sm font-medium">
                  Explore your palette
                </p>
              </div>
            </div>
          </div>
          </div>
        </section>
      )}

      {showResults && analysis && (
        <section
          id="results"
          className="mx-auto max-w-7xl px-5 pb-24 pt-12 sm:px-8 sm:pt-16"
        >
          <div className="flex flex-col gap-5 border-b border-stone-200 pb-8 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs font-medium text-stone-500">
                <SparkleIcon />
                Analysis complete
              </div>

              <h1 className="text-4xl font-semibold tracking-[-0.035em] text-stone-950 sm:text-5xl">
                Your Tinayu profile
              </h1>

              <p className="mt-3 max-w-2xl text-sm leading-6 text-stone-500 sm:text-base">
                Your palette is based on the colors detected from your
                photo and Tinayu&apos;s color-matching engine.
              </p>
            </div>

            <button
              type="button"
              onClick={resetAnalysis}
              className="rounded-2xl border border-stone-200 bg-white px-5 py-3 text-sm font-semibold text-stone-700 transition hover:bg-stone-50"
            >
              New analysis
            </button>
          </div>

          <div className="mt-10">
            <AnalysisJourney
              analysis={analysis}
            />
          </div>

          <div className="mt-10 grid gap-5 lg:grid-cols-[1.4fr_0.8fr]">
            <div className="overflow-hidden rounded-[2rem] border border-stone-200 bg-white shadow-sm">
              <div className="grid md:grid-cols-[0.8fr_1.2fr]">
                <div className="relative min-h-[360px] bg-stone-100">
                  {image && (
                    <Image
                      src={image}
                      alt="Analyzed photo"
                      fill
                      sizes="(max-width: 768px) 100vw, 40vw"
                      unoptimized
                      className="object-cover"
                    />
                  )}
                </div>

                <div className="flex flex-col justify-center p-7 sm:p-10">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
                    Suggested season
                  </p>

                  <h2 className="mt-3 text-4xl font-semibold tracking-tight text-stone-950 sm:text-5xl">
                    {heuristics?.suggested_season ||
                      "Your palette"}
                  </h2>

                  <div className="mt-5 flex flex-wrap gap-2">
                    <span className="rounded-full bg-stone-100 px-3 py-1.5 text-xs font-medium text-stone-600">
                      {heuristics?.temperature || "—"}
                    </span>

                    <span className="rounded-full bg-stone-100 px-3 py-1.5 text-xs font-medium text-stone-600">
                      {heuristics?.skin_depth || "—"}
                    </span>

                    <span className="rounded-full bg-stone-100 px-3 py-1.5 text-xs font-medium text-stone-600">
                      {heuristics?.skin_saturation || "—"}
                    </span>

                    <span className="rounded-full bg-stone-100 px-3 py-1.5 text-xs font-medium text-stone-600">
                      {heuristics?.contrast_level || "—"} contrast
                    </span>
                  </div>

                  <p className="mt-7 text-sm leading-7 text-stone-500">
                    Tinayu uses your detected skin characteristics,
                    color relationships, and lighting-normalized values
                    to create a palette that works as a starting point
                    for your style.
                  </p>
                </div>
              </div>
            </div>

            <div className="rounded-[2rem] border border-stone-200 bg-white p-7 shadow-sm sm:p-8">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
                    Analysis quality
                  </p>

                  <p className="mt-2 text-sm text-stone-500">
                    Photo-quality heuristic
                  </p>
                </div>

                <div className="text-right">
                  <span className="text-4xl font-semibold tracking-tight">
                    {quality
                      ? Math.round(
                          quality.image_confidence
                        )
                      : "—"}
                  </span>

                  <span className="text-sm text-stone-400">
                    / 100
                  </span>
                </div>
              </div>

              <div className="mt-6 h-2 overflow-hidden rounded-full bg-stone-100">
                <div
                  className="h-full rounded-full bg-stone-900 transition-all"
                  style={{
                    width:
                      (quality?.image_confidence || 0) +
                      "%",
                  }}
                />
              </div>

              <p className="mt-4 text-xs leading-5 text-stone-400">
                A photo-quality heuristic based on face detection and
                available facial regions.
              </p>

              <div className="mt-7 grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-stone-50 p-4">
                  <div className="flex items-center gap-2 text-xs font-medium text-stone-500">
                    {quality?.hair_detected ? (
                      <span className="text-emerald-600">
                        <CheckIcon />
                      </span>
                    ) : (
                      <span className="text-stone-300">—</span>
                    )}
                    Hair detected
                  </div>
                </div>

                <div className="rounded-2xl bg-stone-50 p-4">
                  <div className="flex items-center gap-2 text-xs font-medium text-stone-500">
                    {quality?.eyes_detected ? (
                      <span className="text-emerald-600">
                        <CheckIcon />
                      </span>
                    ) : (
                      <span className="text-stone-300">—</span>
                    )}
                    Eyes detected
                  </div>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <div className="rounded-2xl border border-stone-200 p-4">
                  <p className="text-xs text-stone-400">
                    Face confidence
                  </p>

                  <p className="mt-1 text-lg font-semibold">
                    {analysis.face
                      ? Math.round(
                          analysis.face.confidence * 100
                        ) + "%"
                      : "—"}
                  </p>
                </div>

                <div className="rounded-2xl border border-stone-200 p-4">
                  <p className="text-xs text-stone-400">
                    Skin pixels
                  </p>

                  <p className="mt-1 text-lg font-semibold">
                    {quality?.skin_pixels_analyzed || "—"}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {analysis.explanation && (
            <section className="mt-10 overflow-hidden rounded-[2rem] border border-stone-200 bg-stone-900 text-white shadow-sm">
              <div className="p-7 sm:p-9">
                <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
                  <div className="max-w-3xl">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10">
                        <SparkleIcon />
                      </div>
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-white/40">
                          Tinayu AI interpretation
                        </p>
                        <h2 className="mt-1 text-2xl font-semibold tracking-tight">
                          Why this palette was suggested
                        </h2>
                      </div>
                    </div>
                    <p className="mt-6 text-sm leading-7 text-white/70">
                      {analysis.explanation.profile}
                    </p>
                  </div>
                  <span className="shrink-0 rounded-full bg-white/10 px-3 py-1.5 text-xs font-medium text-white/60">
                    AI-assisted explanation
                  </span>
                </div>
                <div className="mt-8 grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl bg-white/5 p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Why</p>
                    <p className="mt-3 text-sm leading-7 text-white/70">{analysis.explanation.why}</p>
                  </div>
                  <div className="rounded-2xl bg-white/5 p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-white/40">Color direction</p>
                    <p className="mt-3 text-sm leading-7 text-white/70">{analysis.explanation.color_direction}</p>
                  </div>
                </div>
                <p className="mt-6 text-xs leading-5 text-white/35">
                  This explanation is generated from Tinayu&apos;s detected color profile and recommendations.
                  It is an AI-assisted interpretation, not a definitive personal color diagnosis.
                </p>
              </div>
            </section>
          )}

          <div className="mt-16">
            <div className="mb-8">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
                Your palette
              </p>

              <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                Colors to explore
              </h2>
            </div>

            <div className="space-y-14">
              <ColorSection
                title="Clothing"
                description="Colors selected from Tinayu's clothing palette."
                colors={
                  recommendations?.clothing || []
                }
              />

              <ColorSection
                title="Makeup"
                description="Tones that complement the detected color profile."
                colors={
                  recommendations?.makeup || []
                }
              />

              <ColorSection
                title="Accents"
                description="Accent colors for accessories and finishing touches."
                colors={
                  recommendations?.accents || []
                }
              />
            </div>
          </div>

          <div className="mt-20">
            <div className="mb-8">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
                Detected colors
              </p>

              <h2 className="mt-2 text-3xl font-semibold tracking-tight">
                Your color profile
              </h2>
            </div>

            <div className="grid gap-5 md:grid-cols-3">
              <div className="overflow-hidden rounded-[2rem] border border-stone-200 bg-white shadow-sm">
                <div
                  className="h-36"
                  style={{
                    backgroundColor:
                      rgbToCss(
                        colors?.skin_normalized_rgb
                      ),
                  }}
                />

                <div className="p-6">
                  <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                    Skin
                  </p>

                  <p className="mt-2 text-lg font-semibold">
                    RGB{" "}
                    {formatRgb(
                      colors?.skin_normalized_rgb
                    )}
                  </p>

                  {colors?.skin_lab && (
                    <p className="mt-2 text-xs text-stone-400">
                      LAB {Math.round(colors.skin_lab.L)},{" "}
                      {Math.round(colors.skin_lab.a)},{" "}
                      {Math.round(colors.skin_lab.b)}
                    </p>
                  )}
                </div>
              </div>

              <div className="overflow-hidden rounded-[2rem] border border-stone-200 bg-white shadow-sm">
                <div
                  className="h-36"
                  style={{
                    backgroundColor:
                      rgbToCss(
                        colors?.hair_rgb
                      ),
                  }}
                />

                <div className="p-6">
                  <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                    Hair
                  </p>

                  <p className="mt-2 text-lg font-semibold">
                    RGB{" "}
                    {formatRgb(
                      colors?.hair_rgb
                    )}
                  </p>

                  {!quality?.hair_detected && (
                    <p className="mt-2 text-xs text-stone-400">
                      Hair region was not confidently detected.
                    </p>
                  )}
                </div>
              </div>

              <div className="overflow-hidden rounded-[2rem] border border-stone-200 bg-white shadow-sm">
                <div
                  className="h-36"
                  style={{
                    backgroundColor:
                      rgbToCss(
                        colors?.eye_rgb
                      ),
                  }}
                />

                <div className="p-6">
                  <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                    Eyes
                  </p>

                  <p className="mt-2 text-lg font-semibold">
                    RGB{" "}
                    {formatRgb(
                      colors?.eye_rgb
                    )}
                  </p>

                  {!quality?.eyes_detected && (
                    <p className="mt-2 text-xs text-stone-400">
                      Eye region was not confidently detected.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-20 grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="rounded-[2rem] border border-stone-200 bg-white p-7 shadow-sm sm:p-9">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-900 text-white">
                  <SparkleIcon />
                </div>

                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-stone-400">
                    About your result
                  </p>

                  <h2 className="mt-1 text-2xl font-semibold tracking-tight">
                    A starting point, not a strict rule.
                  </h2>
                </div>
              </div>

              <p className="mt-7 max-w-3xl text-sm leading-7 text-stone-500">
                Tinayu uses computer vision, color-space analysis,
                lighting normalization, and a recommendation engine to
                suggest colors from your photo. Results can vary with
                lighting, camera settings, filters, makeup, dyed hair,
                and image quality.
              </p>

              <div className="mt-7 rounded-2xl bg-stone-50 p-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-stone-400">
                  Important
                </p>

                <p className="mt-2 text-sm leading-6 text-stone-600">
                  Your suggested season is an AI-assisted color
                  interpretation rather than a definitive personal
                  color diagnosis.
                </p>
              </div>
            </div>

            <div className="rounded-[2rem] border border-stone-200 bg-stone-900 p-7 text-white shadow-sm sm:p-9">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-white/40">
                Tinayu profile
              </p>

              <div className="mt-7 flex items-center gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/10 text-lg font-semibold">
                  {getInitials(
                    heuristics?.suggested_season ||
                      "Tinayu"
                  )}
                </div>

                <div>
                  <p className="text-lg font-semibold">
                    {heuristics?.suggested_season ||
                      "Color profile"}
                  </p>

                  <p className="mt-1 text-sm text-white/50">
                    {heuristics?.temperature ||
                      "Personalized palette"}
                  </p>
                </div>
              </div>

              <div className="mt-8 grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-white/5 p-4">
                  <p className="text-xs text-white/40">
                    Skin
                  </p>

                  <p className="mt-1 text-sm font-medium">
                    {heuristics?.skin_depth || "—"}
                  </p>
                </div>

                <div className="rounded-2xl bg-white/5 p-4">
                  <p className="text-xs text-white/40">
                    Saturation
                  </p>

                  <p className="mt-1 text-sm font-medium">
                    {heuristics?.skin_saturation || "—"}
                  </p>
                </div>

                <div className="rounded-2xl bg-white/5 p-4">
                  <p className="text-xs text-white/40">
                    Temperature
                  </p>

                  <p className="mt-1 text-sm font-medium">
                    {heuristics?.temperature || "—"}
                  </p>
                </div>

                <div className="rounded-2xl bg-white/5 p-4">
                  <p className="text-xs text-white/40">
                    Contrast
                  </p>

                  <p className="mt-1 text-sm font-medium">
                    {heuristics?.contrast_level || "—"}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-12 text-center">
            <button
              type="button"
              onClick={resetAnalysis}
              className="rounded-2xl bg-stone-900 px-7 py-4 text-sm font-semibold text-white transition hover:bg-stone-800"
            >
              Analyze another photo
            </button>
          </div>
        </section>
      )}

      <footer className="border-t border-stone-200">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-5 py-8 text-xs text-stone-400 sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <p>© 2026 Tinayu</p>

          <p>
            Computer vision · Color analysis · AI-assisted recommendations
          </p>
        </div>
      </footer>
    </main>
  );
}