"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_TINAYU_API_URL || "http://127.0.0.1:8000";

export default function ScanPage() {const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [starting, setStarting] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [preview, setPreview] = useState("");

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraReady(false);
  }, []);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function startCamera() {
    setError("");
    setStarting(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not available in this browser. Try opening the site in a secure browser tab.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("The camera opened, but the preview was not ready. Please try again.");
      }

      video.srcObject = stream;

      // Wait until the browser has attached the stream and received a frame.
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => {
          cleanup();
          reject(new Error("The camera is taking too long to start. Close other camera apps and try again."));
        }, 10000);

        const cleanup = () => {
          window.clearTimeout(timeout);
          video.removeEventListener("loadedmetadata", onReady);
          video.removeEventListener("error", onError);
        };

        const onReady = () => {
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            cleanup();
            resolve();
          }
        };

        const onError = () => {
          cleanup();
          reject(new Error("The browser could not display the camera preview."));
        };

        video.addEventListener("loadedmetadata", onReady);
        video.addEventListener("error", onError);

        // Metadata may already have arrived before the listeners were added.
        onReady();
      });

      await video.play();
      setCameraReady(true);
    } catch (e) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraReady(false);
      setError(e instanceof Error ? e.message : "Could not start the camera. Check your browser permissions and try again.");
    } finally {
      setStarting(false);
    }
  }

  async function analyzeImage(file: File) {
    setError("");
    setResult(null);
    setPreview(URL.createObjectURL(file));
    setScanning(true);
    try {
      if (!file.type.startsWith("image/")) {
        throw new Error("Please choose a valid image file.");
      }

      const form = new FormData();
      form.append("file", file);

      let response: Response;
      try {
        response = await fetch(`${API_URL}/analyze`, {
          method: "POST",
          body: form,
        });
      } catch {
        throw new Error(`Tinayu could not reach the analysis API at ${API_URL}. Check that the backend is running and try again.`);
      }

      const responseText = await response.text();
      let data: Record<string, unknown> = {};
      try {
        data = responseText ? JSON.parse(responseText) as Record<string, unknown> : {};
      } catch {
        throw new Error(
          `The analysis API returned an unexpected response (HTTP ${response.status}). Check the backend URL and API logs.`,
        );
      }

      if (!response.ok) {
        const detail = data.detail ?? data.message ?? data.error;
        throw new Error(typeof detail === "string" ? detail : `Analysis failed (HTTP ${response.status}). Please try another photo.`);
      }

      if (data.success === false) {
        throw new Error(typeof data.error === "string" ? data.error : "Tinayu could not analyze this photo. Try a clearer image with your face visible.");
      }

      // Reuse the homepage's full results screen instead of showing a reduced Scan-only summary.
      const imageDataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => typeof reader.result === "string"
          ? resolve(reader.result)
          : reject(new Error("Could not prepare the analyzed photo."));
        reader.onerror = () => reject(new Error("Could not prepare the analyzed photo."));
        reader.readAsDataURL(file);
      });

      sessionStorage.setItem("tinayu-scan-result", JSON.stringify({
        analysis: data,
        image: imageDataUrl,
      }));
      stopCamera();
      router.push("/?fromScan=1");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong during analysis.");
    } finally {
      setScanning(false);
    }
  }

  async function captureAndAnalyze() {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      setError("Wait for the camera preview to appear, then try again.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) {
      setError("Could not prepare the camera image. Please try again.");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.94));
    if (!blob) {
      setError("Could not capture the photo. Please try again.");
      return;
    }
    await analyzeImage(new File([blob], "tinayu-camera-scan.jpg", { type: "image/jpeg" }));
  }

  const profile = result && typeof result.profile === "object" && result.profile !== null ? result.profile as Record<string, unknown> : null;
  const heuristics = profile && typeof profile.heuristics === "object" && profile.heuristics !== null ? profile.heuristics as Record<string, unknown> : null;

  return (
    <main className="min-h-screen bg-[#f7f4ef] text-stone-900">
      <header className="border-b border-stone-200/80 bg-[#f7f4ef]/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 sm:px-8">
          <Link href="/" className="flex items-center gap-3" aria-label="Tinayu home">
            <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-stone-950 text-lg text-[#f2d7d3] shadow-lg">✦</span>
            <span><span className="block text-base font-semibold tracking-tight">Tinayu</span><span className="block text-[9px] tracking-[0.25em] text-stone-400">PERSONAL COLOR AI</span></span>
          </Link>
          <Link href="/" className="rounded-full border border-stone-200 bg-white/70 px-4 py-2 text-sm font-medium text-stone-600 transition hover:border-stone-400 hover:text-stone-950">← Back home</Link>
        </div>
      </header>

      <section className="mx-auto max-w-7xl px-5 py-8 sm:px-8 sm:py-12">
        <div className="mx-auto mb-9 max-w-3xl text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-[#e8d5d1] bg-white/80 px-4 py-2 text-[10px] font-bold uppercase tracking-[0.2em] text-[#a16f72] shadow-sm">✦ Tinayu Smart Scan</span>
          <h1 className="mt-5 text-4xl font-semibold leading-tight tracking-[-0.05em] text-stone-950 sm:text-6xl">Meet your colors, <span className="font-serif italic font-normal text-[#b7797e]">in real time.</span></h1>
          <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-stone-600 sm:text-base">Use the guided camera experience to capture a clear photo for your personal color analysis. Soft, even lighting helps; your results are guidance, not rules.</p>
        </div>

        <div className="mx-auto grid max-w-5xl items-start gap-6 lg:grid-cols-[1.25fr_0.75fr]">
          <section className="overflow-hidden rounded-[2rem] border border-white bg-white/90 p-3 shadow-[0_24px_80px_rgba(76,56,47,0.12)] sm:p-5">
            <div className="relative flex min-h-[340px] items-center justify-center overflow-hidden rounded-[1.5rem] bg-[#201e1c] sm:min-h-[470px]">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                aria-label="Live camera preview"
                className={`absolute inset-0 h-full w-full object-cover ${cameraReady ? "opacity-100" : "pointer-events-none opacity-0"}`}
              />
              {!cameraReady && preview && <img src={preview} alt="Photo selected for color analysis" className="absolute inset-0 h-full w-full object-contain bg-stone-950" />}
              {!cameraReady && !preview && <div className="max-w-sm px-8 text-center text-white"><div className="mx-auto flex h-16 w-16 items-center justify-center rounded-3xl border border-white/15 bg-white/10 text-3xl text-[#e8c6c3]">✦</div><h2 className="mt-5 text-xl font-semibold">Your scan space</h2><p className="mt-2 text-sm leading-6 text-white/60">Start your camera to align your face inside the guide. You can also choose a photo instead.</p></div>}
              {cameraReady && <div className="pointer-events-none absolute inset-0"><div className="absolute inset-x-[19%] top-[8%] bottom-[8%] rounded-[48%] border-2 border-white/80 shadow-[0_0_0_9999px_rgba(0,0,0,0.22)]"/><div className="absolute left-1/2 top-5 -translate-x-1/2 rounded-full border border-white/15 bg-black/45 px-4 py-2 text-xs font-medium text-white backdrop-blur">Center your face in the frame</div><div className="absolute bottom-5 left-1/2 -translate-x-1/2 rounded-full bg-black/45 px-4 py-2 text-xs text-white backdrop-blur">Live camera preview</div></div>}
            </div>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row">
              {!cameraReady ? <button onClick={startCamera} disabled={starting || scanning} className="group flex min-h-14 flex-1 items-center justify-center gap-3 rounded-2xl bg-stone-950 px-5 py-4 text-sm font-semibold text-white shadow-[0_12px_30px_rgba(28,25,23,0.2)] transition hover:-translate-y-0.5 hover:bg-stone-800 disabled:opacity-60">{starting ? "Opening camera…" : "✦ Enable camera"}<span className="text-[#e8c6c3]">→</span></button> : <><button onClick={captureAndAnalyze} disabled={scanning} className="flex min-h-14 flex-1 items-center justify-center gap-2 rounded-2xl bg-stone-950 px-5 py-4 text-sm font-semibold text-white shadow-lg transition hover:bg-stone-800 disabled:opacity-60">{scanning ? "Analyzing your photo…" : "✦ Capture & analyze"}</button><button onClick={stopCamera} className="rounded-2xl border border-stone-200 bg-white px-5 py-4 text-sm font-semibold text-stone-600 transition hover:bg-stone-50">Stop camera</button></>}
              <button onClick={() => fileRef.current?.click()} disabled={scanning} className="min-h-14 rounded-2xl border border-[#e5d3cf] bg-[#fbf5f2] px-5 py-4 text-sm font-semibold text-[#8f6668] transition hover:bg-[#f4e8e5] disabled:opacity-60">Choose a photo</button>
              <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(event) => { const file = event.target.files?.[0]; if (file) void analyzeImage(file); event.currentTarget.value = ""; }} />
            </div>
            {error && <p role="alert" className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm leading-6 text-red-700">{error}</p>}
          </section>

          <aside className="rounded-[2rem] border border-[#eadbd8] bg-white/85 p-6 shadow-[0_16px_50px_rgba(105,75,67,0.07)] sm:p-7">
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#a77b7b]">Your guided scan</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight">A calmer way to discover your palette.</h2>
            <div className="mt-6 space-y-5">
              {[{n:"01",title:"Find soft, even light",body:"Face a window or a well-lit room. Avoid strong shadows and color filters."},{n:"02",title:"Center your face",body:"Keep your face inside the guide and look toward the camera."},{n:"03",title:"Explore your results",body:"Tinayu analyzes the image and presents color insights to help you explore."}].map((item) => <div key={item.n} className="flex gap-4"><span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#f5e9e6] text-xs font-bold text-[#a16f72]">{item.n}</span><div><h3 className="text-sm font-semibold text-stone-900">{item.title}</h3><p className="mt-1 text-sm leading-6 text-stone-500">{item.body}</p></div></div>)}
            </div>
            <div className="mt-7 rounded-2xl border border-[#eadbd8] bg-[#faf5f1] p-4"><p className="text-xs font-semibold uppercase tracking-wider text-[#a16f72]">A quick note</p><p className="mt-2 text-sm leading-6 text-stone-600">Camera access is requested by your browser. Tinayu only sends the image when you choose to capture or analyze it.</p></div>
          </aside>
        </div>

        {result && <section aria-live="polite" className="mx-auto mt-6 max-w-5xl rounded-[2rem] border border-[#e8d5d1] bg-white p-6 shadow-lg sm:p-8"><p className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#a77b7b]">Your scan results</p><h2 className="mt-2 text-2xl font-semibold">Your color insights are ready</h2><div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{Object.entries(heuristics || {}).map(([key, value]) => <div key={key} className="rounded-2xl bg-[#f8f4f0] p-4"><p className="text-xs capitalize text-stone-500">{key.replaceAll("_", " ")}</p><p className="mt-1 text-base font-semibold capitalize text-stone-900">{String(value).replaceAll("_", " ")}</p></div>)}</div><p className="mt-5 text-xs leading-5 text-stone-400">Use these insights as a starting point; color harmony is personal and flexible.</p></section>}
      </section>
    </main>
  );
}
