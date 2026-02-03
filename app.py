import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av


# ---------------- Load Face Detector ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- UI ----------------
st.title("🎭 Real-Time Face Detection with Filters")

st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Select Input Source",
    ["Webcam", "Upload Video", "Upload Image"]
)

filter_option = st.sidebar.selectbox(
    "Select Filter",
    ["Bounding Box", "Emoji", "Blur Face"]
)

# Emoji selection
emoji_map = {
    "😈": "smiling_imp.png",
    "😁": "1f601.png",
    "🥸": "disguised_face.png",
    "🤬": "face_with_symbols_on_mouth.png",
    "😍": "heart_eyes.png",
    "😡": "rage.png"
}

if filter_option == "Emoji":
    emoji_choice = st.sidebar.selectbox(
        "Choose Emoji",
        list(emoji_map.keys())
    )

    emoji_img = cv2.imread(
        emoji_map[emoji_choice],
        cv2.IMREAD_UNCHANGED
    )

snapshot = st.sidebar.checkbox("Auto Snapshot")

FRAME_WINDOW = st.image([])
snapshot_counter = 0


# ---------------- Emoji Overlay Function ----------------
def overlay_emoji(frame, emoji_img, x, y, w, h):

    # Prevent overflow outside frame
    h_frame, w_frame = frame.shape[:2]
    if y + h > h_frame or x + w > w_frame:
        return frame

    emoji_resized = cv2.resize(emoji_img, (w, h))

    if emoji_resized.shape[2] == 4:  # Transparent PNG
        alpha = emoji_resized[:, :, 3] / 255.0

        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                alpha * emoji_resized[:, :, c] +
                (1 - alpha) * frame[y:y+h, x:x+w, c]
            )
    else:
        frame[y:y+h, x:x+w] = emoji_resized

    return frame

# ---------------- Filter Function ----------------
def apply_filter(frame, faces):
    global snapshot_counter

    for (x, y, w, h) in faces:

        if filter_option == "Bounding Box":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Face", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        elif filter_option == "Emoji":
            frame = overlay_emoji(frame, emoji_img, x, y, w, h)

        elif filter_option == "Blur Face":
            face_region = frame[y:y+h, x:x+w]
            blur = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = blur

        # Snapshot
        if snapshot:
            filename = f"snapshot_{snapshot_counter}.jpg"
            cv2.imwrite(filename, frame)
            snapshot_counter += 1

    # Face count display
    cv2.putText(frame, f"Faces: {len(faces)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    return frame

class FaceFilter(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        img = apply_filter(img, faces)

        return img

# ---------------- Webcam Mode ----------------
if mode == "Webcam":
    webrtc_streamer(
        key="face-detect",
        video_transformer_factory=FaceFilter
    )

# ---------------- Video Upload Mode ----------------
elif mode == "Upload Video":

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file is not None:

        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            frame = apply_filter(frame, faces)

            FRAME_WINDOW.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

        cap.release()


# ---------------- Image Upload Mode ----------------
elif mode == "Upload Image":

    img_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if img_file is not None:
        img = Image.open(img_file)
        frame = np.array(img)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame = apply_filter(frame, faces)

        st.image(frame)

