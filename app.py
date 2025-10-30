# app.py
# SE3071 – Education/e-Learning (IT last digit = 2)
# Allowed libs: cv2, numpy, matplotlib.pyplot + streamlit

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pytesseract

st.set_page_config(page_title="EduVision - Educational Image Processing App", layout="wide")


# -------------------------
# Helpers
# -------------------------
def read_image(file) -> np.ndarray:
    """Read uploaded file into BGR image using cv2 only."""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def ensure_gray(img: np.ndarray) -> np.ndarray:
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_histogram(gray: np.ndarray):
    fig = plt.figure()
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title("Histogram - Grayscale")
    plt.xlabel("Intensity"); plt.ylabel("Frequency")
    st.pyplot(fig)

def frequency_filter(gray: np.ndarray, filter_type='low', method='ideal', d0=30):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    d2 = (x - ccol) ** 2 + (y - crow) ** 2
    if method == 'ideal':
        mask = (d2 <= d0 ** 2).astype(np.float32)
    else:  # gaussian
        mask = np.exp(-d2 / (2 * (d0 ** 2))).astype(np.float32)
    if filter_type == 'high':
        mask = 1.0 - mask
    filtered = fshift * mask
    out = np.fft.ifft2(np.fft.ifftshift(filtered))
    out = np.abs(out)
    return np.uint8(np.clip(out, 0, 255))

def watershed_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers)
    result = img_bgr.copy()
    result[markers == -1] = [0, 0, 255]  # red boundaries
    return result

def download_button(image_bgr: np.ndarray, label: str, key: str):
    ok, buf = cv2.imencode(".png", image_bgr)
    if ok:
        st.download_button(label, buf.tobytes(), file_name="output.png", mime="image/png", key=key)

# -------------------------
# -------------------------
# 🌈 Enhanced Colorful UI
# -------------------------
st.markdown(
    """
    <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #f0f7ff 0%, #e8fff3 50%, #fff5ef 100%);
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0077b6, #00b4d8);
            color: white;
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] div {
            color: white !important;
        }

        /* Title and caption styling */
        h1 {
            color: #004aad !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        p {
            font-size: 16px;
            color: #333333;
        }

        /* Upload box styling */
        .uploadedFile {
            border: 2px dashed #0096c7 !important;
            border-radius: 10px;
            background-color: #f0fcff !important;
        }

        /* Divider line */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(to right, #48cae4, #00b4d8, #0096c7);
            margin-top: 25px;
            margin-bottom: 25px;
        }

        /* Subheaders */
        .stMarkdown h3, .stMarkdown h2, .stMarkdown h4 {
            color: #0077b6 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and caption
st.markdown("<h1 style='text-align:center;'>🎓 VisionBoard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px; color:#444;'>✨ Empowering learning through intelligent image enhancement ✨</p>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("📂 Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    img_bgr = read_image(uploaded)
    if img_bgr is None:
        st.error("❌ Could not read the image. Try another file.")
        st.stop()

    # Show original + metadata
    col0, col1 = st.columns([2, 1])
    with col0:
        st.markdown("### 🖼️ Original Image")
        st.image(to_rgb(img_bgr), channels="RGB", use_column_width=True, caption="Uploaded Image")

    with col1:
        h, w = img_bgr.shape[:2]
        ch = img_bgr.shape[2] if len(img_bgr.shape) == 3 else 1
        st.markdown("### 📊 Image Metadata")
        st.success(f"**Resolution:** {w} × {h}")
        st.info(f"**Channels:** {ch}")
        st.warning(f"**Data Type:** {img_bgr.dtype}")
        st.write(f"**Total Pixels:** {int(img_bgr.size):,}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar style
    st.sidebar.markdown("## 🎛️ Workspace Controls")
    st.sidebar.markdown(
        "<p style='font-size:14px;color:white;'>Select which module of VisionBoard you want to explore:</p>",
        unsafe_allow_html=True
    )
    mode = st.sidebar.radio(
        "Choose a mode:",
        ["🎨 Part A (Basics)", "⚙️ Part B (Advanced)", "🧾 Whiteboard Preset"]
    )


    # ============================================================
    # PART A
    # ============================================================
    if mode == "Part A (Basics)":
        st.header("Part A – Basic Processing")

        a1 = st.checkbox("Show Grayscale / HSV / Binary", value=True)
        if a1:
            colA1, colA2, colA3 = st.columns(3)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            _, binary = cv2.threshold(gray, st.slider("Binary threshold", 0, 255, 128), 255, cv2.THRESH_BINARY)
            with colA1:
                st.image(gray, caption="Grayscale", use_column_width=True, clamp=True)
            with colA2:
                st.image(to_rgb(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)), caption="HSV (shown as BGR)", use_column_width=True)
            with colA3:
                st.image(binary, caption="Binary", use_column_width=True, clamp=True)

        a2 = st.checkbox("Histogram & Equalization", value=True)
        if a2:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            st.subheader("Grayscale Histogram")
            show_histogram(gray)
            gray_eq = cv2.equalizeHist(gray)
            st.image([gray, gray_eq], caption=["Original Gray", "Equalized Gray"], use_column_width=True, clamp=True)

            # Color equalization via Y channel (YCrCb)
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            color_eq = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
            st.image([to_rgb(img_bgr), to_rgb(color_eq)], caption=["Original", "Color Equalized (Y channel)"], use_column_width=True)
            download_button(color_eq, "Download Color-Equalized", "dl_color_eq")

        a3 = st.checkbox("Geometric Transforms", value=True)
        if a3:
            colT1, colT2 = st.columns(2)
            with colT1:
                deg = st.slider("Rotate (degrees)", -180, 180, 15)
                h, w = img_bgr.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), deg, 1.0)
                rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                st.image(to_rgb(rotated), caption="Rotated", use_column_width=True)
                download_button(rotated, "Download Rotated", "dl_rot")

            with colT2:
                scale = st.slider("Scale factor", 10, 300, 100) / 100.0
                resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
                st.image(to_rgb(resized), caption=f"Scaled x{scale:.2f}", use_column_width=True)
                download_button(resized, "Download Scaled", "dl_scaled")

            colT3, colT4 = st.columns(2)
            with colT3:
                tx = st.slider("Translate X", -200, 200, 30)
                ty = st.slider("Translate Y", -200, 200, 20)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                st.image(to_rgb(translated), caption=f"Translated ({tx},{ty})", use_column_width=True)
                download_button(translated, "Download Translated", "dl_trans")

            with colT4:
                x = st.slider("Crop X", 0, max(0, w - 1), min(100, w - 1))
                y = st.slider("Crop Y", 0, max(0, h - 1), min(100, h - 1))
                cw = st.slider("Crop Width", 1, max(1, w - x), min(300, w - x))
                ch = st.slider("Crop Height", 1, max(1, h - y), min(300, h - y))
                crop = img_bgr[y:y + ch, x:x + cw]
                st.image(to_rgb(crop), caption="Cropped", use_column_width=True)
                download_button(crop, "Download Cropped", "dl_crop")

    # ============================================================
    # PART B
    # ============================================================
    elif mode == "Part B (Advanced)":
        st.header("Part B – Advanced Processing")

        # 1) Smoothing
        st.subheader("1) Smoothing & Noise Reduction")
        k = st.slider("Kernel size", 3, 15, 5, step=2)
        avg = cv2.blur(img_bgr, (k, k))
        gauss = cv2.GaussianBlur(img_bgr, (k, k), 0)
        med = cv2.medianBlur(img_bgr, k)
        st.image([to_rgb(avg), to_rgb(gauss), to_rgb(med)], caption=["Averaging", "Gaussian", "Median"], use_column_width=True)

        # 2) Edges & Sharpening
        st.subheader("2) Edge Detection & Sharpening")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
        lap = np.uint8(np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), 0, 255))
        t1 = st.slider("Canny T1", 0, 255, 50)
        t2 = st.slider("Canny T2", 0, 255, 150)
        canny = cv2.Canny(gray, t1, t2)
        sharp = cv2.addWeighted(img_bgr, 1.5, cv2.GaussianBlur(img_bgr, (0, 0), 1.0), -0.5, 0)
        st.image([sobel, lap, canny, to_rgb(sharp)],
                 caption=["Sobel Magnitude", "Laplacian", "Canny", "Sharpened"],
                 use_column_width=True, clamp=True)

        # 3) Morphology
        st.subheader("3) Morphological Operations")
        _, bin_ = cv2.threshold(gray, st.slider("Binary Threshold", 0, 255, 127), 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        er = cv2.erode(bin_, kernel, iterations=1)
        di = cv2.dilate(bin_, kernel, iterations=1)
        op = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel)
        cl = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, kernel)
        st.image([bin_, er, di, op, cl], caption=["Binary", "Erode", "Dilate", "Open", "Close"], use_column_width=True, clamp=True)

        # 4) Segmentation
        st.subheader("4) Segmentation (Global, Otsu, Watershed)")
        _, th_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ws = watershed_segmentation(img_bgr)
        st.image([th_global, th_otsu, to_rgb(ws)],
                 caption=["Global Threshold", "Otsu Threshold", "Watershed"],
                 use_column_width=True, clamp=True)

        # 5) Frequency Domain
        st.subheader("5) Frequency Domain Filters (Ideal/Gaussian - Low/High)")
        d0 = st.slider("Cutoff (d0)", 5, 100, 30)
        gray = ensure_gray(img_bgr)
        ilp = frequency_filter(gray, 'low', 'ideal', d0)
        ihp = frequency_filter(gray, 'high', 'ideal', d0)
        glp = frequency_filter(gray, 'low', 'gaussian', d0)
        ghp = frequency_filter(gray, 'high', 'gaussian', d0)
        st.image([ilp, ihp, glp, ghp],
                 caption=["Ideal LP", "Ideal HP", "Gaussian LP", "Gaussian HP"],
                 use_column_width=True, clamp=True)

        # Optional download (pick one to download)
        st.subheader("Download any result (choose a preview above and re-run below):")
        choice = st.selectbox("Select result to download",
                              ["Sharpened (from edges)", "Watershed (boundaries in red)", "Gaussian LP (frequency)"])
        if st.button("Prepare download"):
            if choice == "Sharpened (from edges)":
                download_button(sharp, "Download Sharpened", "dl_sharp")
            elif choice == "Watershed (boundaries in red)":
                download_button(ws, "Download Watershed", "dl_ws")
            else:
                download_button(cv2.cvtColor(glp, cv2.COLOR_GRAY2BGR), "Download Gaussian LP", "dl_glp")

    # ============================================================
    # Whiteboard preset (Domain helper)
    # ============================================================
    else:
        st.header("Whiteboard / Slide Cleanup Preset")
        # 1) Median denoise
        k = st.slider("Median ksize", 3, 11, 5, step=2)
        den = cv2.medianBlur(img_bgr, k)
        # 2) Equalize luminance
        ycrcb = cv2.cvtColor(den, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        eq = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)
        # 3) Adaptive threshold to isolate ink
        gray = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
        block = st.slider("Adaptive block size (odd)", 11, 51, 31, step=2)
        C = st.slider("Adaptive constant C", 0, 20, 10)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, C)
        # 4) Morph close to thicken strokes
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        st.image([to_rgb(img_bgr), to_rgb(den), to_rgb(eq), closed],
                 caption=["Original", "Median Denoised", "Luminance Equalized", "Adaptive Threshold + Close"],
                 use_column_width=True)
        download_button(cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR), "Download Cleaned Board (Binary as BGR)", "dl_board")

else:
    st.info("👆 Upload a PNG/JPG to begin.")

# -------------------------
# 🔤 Text Extract (OCR) — Add-on (append to end of file)
# -------------------------
if uploaded:
    # Sidebar toggle for the new feature (kept separate from your existing mode switch)
    st.sidebar.markdown("## 🔤 Text Extraction")
    enable_ocr = st.sidebar.checkbox("Enable Text Extraction", value=False, key="enable_ocr")

    if enable_ocr:
        # Try to import Tesseract OCR locally so we don't touch your top imports
        try:
            import pytesseract  # type: ignore
            _HAS_TESS = True
        except Exception:
            _HAS_TESS = False

        # --- small helpers (local to this block to avoid changing the top of your file) ---
        def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
            """Grayscale + light denoise + adaptive threshold for better OCR."""
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            th = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # fixed constant name
                cv2.THRESH_BINARY,
                31,
                10
            )
            return th

        def _text_to_pdf_bytes(text: str, title: str = "Extracted Text") -> bytes:
            """Render plain text to a single-page PDF (A4) via Matplotlib."""
            from io import BytesIO
            buf = BytesIO()
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.set_title(title, fontsize=14)
            ax.text(
                0, 1,
                text.replace("\t", "    "),
                va="top", ha="left",
                fontsize=10, family="monospace", wrap=True
            )
            fig.savefig(buf, format="pdf")
            plt.close(fig)
            return buf.getvalue()
        # --- end helpers ---

        st.header("Text Extract (OCR)")
        st.caption("Convert the image to text and download as a PDF.")

        # Show preprocessing preview
        proc = _preprocess_for_ocr(img_bgr)
        st.image(
            [to_rgb(img_bgr), proc],
            caption=["Original", "Binarized for OCR"],
            use_column_width=True, clamp=True
        )

        if _HAS_TESS:
            # Try OCR on both normal and inverted; keep the better one
            txt1 = pytesseract.image_to_string(proc)
            txt2 = pytesseract.image_to_string(255 - proc)
            text = txt1 if len(txt1.strip()) >= len(txt2.strip()) else txt2
            text = text.strip()

            st.text_area("Extracted Text", value=text, height=260, key="ocr_text_area")

            if st.button("Generate PDF from Text", key="make_pdf_btn"):
                pdf_bytes = _text_to_pdf_bytes(text or "(No text detected)")
                st.download_button(
                    "📥 Download Text as PDF",
                    data=pdf_bytes,
                    file_name="extracted_text.pdf",
                    mime="application/pdf",
                    key="pdf_dl_btn"
                )
        else:
            # Graceful fallback: guide install + offer binary PDF (image, not selectable text)
            st.error(
                "Tesseract OCR not detected. Install system Tesseract and the `pytesseract` Python package "
                "to enable true text extraction."
            )
            st.info(
                "Fallback: you can still download a cleaned **binary PDF** (image-only). "
                "Use the Whiteboard Preset first if needed for better contrast."
            )

            if st.button("Generate Binary Image PDF", key="bin_pdf_btn"):
                from io import BytesIO
                buf = BytesIO()
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.axis("off")
                plt.imshow(proc, cmap="gray")
                fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                st.download_button(
                    "📥 Download Binary as PDF",
                    data=buf.getvalue(),
                    file_name="binary_image.pdf",
                    mime="application/pdf",
                    key="bin_pdf_dl"
                )










