import av
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# ------------------- Streamlit Page Config -------------------
st.set_page_config(page_title="AI Vision Suite", page_icon="👁", layout="wide")
st.title("👁 AI Vision Suite")
st.caption("Face Detection (Haar + MediaPipe), Eye Analysis & Hand Sign Recognition")
st.markdown("---")

# ------------------- Sidebar Controls -------------------
st.sidebar.header("⚙ Settings")
mode = st.sidebar.radio("Choose Mode", ["📷 Image Upload", "🎥 Webcam (Real-time)", "🎞 Video File"])

# Feature toggles
enable_face_detection = st.sidebar.checkbox("Face Detection", value=True)
enable_eye_analysis = st.sidebar.checkbox("Eye Analysis", value=True)
enable_hand_signs = st.sidebar.checkbox("Hand Sign Recognition", value=True)

# Haar Cascade params
st.sidebar.subheader("Haar Cascade Params")
scale_factor = st.sidebar.slider("Scale factor", 1.01, 1.5, 1.1, 0.01)
min_neighbors = st.sidebar.slider("Min neighbors", 1, 12, 5, 1)
min_size_px = st.sidebar.slider("Min face size (px)", 20, 200, 40, 5)
draw_score = st.sidebar.checkbox("Show detection size", value=True)
show_fps = st.sidebar.checkbox("Show FPS (webcam)", value=True)
max_width = st.sidebar.slider("Max video width (px)", 320, 1920, 960, 10)

# ------------------- Haar Cascade -------------------
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_haar(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                     minSize=(min_size_px, min_size_px), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def draw_faces_haar(bgr_img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if draw_score:
            label = f"{w}x{h}"
            cv2.putText(bgr_img, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return bgr_img

# ------------------- MediaPipe AI Class -------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class FaceDetectionAI:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hand_signs = {
            'thumbs_up': 'Thumbs Up 👍','thumbs_down': 'Thumbs Down 👎','peace': 'Peace Sign ✌','ok': 'OK Sign 👌',
            'rock': 'Rock 🤘','fist': 'Fist ✊','open_palm': 'Open Palm ✋','pointing': 'Pointing 👉'
        }

    def detect_faces(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for d in results.detections:
                box = d.location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)
                faces.append({'bbox': (x, y, bw, bh), 'confidence': d.score[0]})
        return faces

    def analyze_eyes(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        eye_data = []
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                h, w, _ = image.shape
                left_ids = [33, 133]; right_ids = [362, 263]
                left_eye = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in left_ids]
                right_eye = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in right_ids]
                eye_data.append({'left_eye_points': left_eye,'right_eye_points': right_eye})
        return eye_data

    def detect_hand_signs(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        signs = []
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            for idx, hand in enumerate(results.multi_hand_landmarks):
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
                signs.append({'landmarks': pts, 'handedness': results.multi_handedness[idx].classification[0].label})
        return signs

    def draw_annotations(self, image, faces, eyes, hands):
        out = image.copy()
        for f in faces:
            x, y, w, h = f['bbox']
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(out, f"{f['confidence']:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        for e in eyes:
            for p in e['left_eye_points']+e['right_eye_points']:
                cv2.circle(out, p, 2, (255,0,0), -1)
        for hnd in hands:
            for p in hnd['landmarks']:
                cv2.circle(out, p, 2, (0,0,255), -1)
        return out

# ------------------- Webcam Processor -------------------
class FaceDetectProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_time = time.time()
        self.fps = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        faces = detect_faces_haar(img)
        img = draw_faces_haar(img, faces)
        if show_fps:
            now = time.time(); dt = now - self.last_time; self.last_time = now
            if dt > 0:
                self.fps = 0.9*self.fps + 0.1*(1.0/dt) if self.fps>0 else (1.0/dt)
            cv2.putText(img, f"FPS: {self.fps:.1f}", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ------------------- Main Logic -------------------
ai = FaceDetectionAI()

if mode == "📷 Image Upload":
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded:
        img = np.array(Image.open(uploaded))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = ai.detect_faces(bgr) if enable_face_detection else []
        eyes = ai.analyze_eyes(bgr) if enable_eye_analysis else []
        hands = ai.detect_hand_signs(bgr) if enable_hand_signs else []
        annotated = ai.draw_annotations(bgr, faces, eyes, hands)
        st.image(img, caption="Original", use_container_width=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analysis", use_container_width=True)

elif mode == "🎥 Webcam (Real-time)":
    webrtc_streamer(key="cam", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config,
                    video_processor_factory=FaceDetectProcessor, media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)

elif mode == "🎞 Video File":
    uploaded = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
    if uploaded and st.button("Process Video"):
        t_in = tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded.name.split(".")[-1])
        t_in.write(uploaded.read()); t_in.flush()
        cap = cv2.VideoCapture(t_in.name)
        if not cap.isOpened():
            st.error("Could not open video")
        else:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            scale = min(1.0, max_width/max(w,1)); out_w, out_h = int(w*scale), int(h*scale)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            t_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            writer = cv2.VideoWriter(t_out.name, fourcc, fps, (out_w, out_h))
            prog = st.progress(0); info = st.empty(); proc = 0
            while True:
                ret, frame = cap.read();
                if not ret: break
                if scale!=1.0: frame = cv2.resize(frame, (out_w,out_h))
                faces = detect_faces_haar(frame)
                frame = draw_faces_haar(frame, faces)
                writer.write(frame); proc += 1
                if total>0: prog.progress(min(1.0, proc/total)); info.text(f"{proc}/{total}")
            cap.release(); writer.release(); prog.empty(); info.empty()
            st.success("Done!")
            st.video(t_out.name)
            with open(t_out.name, "rb") as f:
                st.download_button("⬇ Download processed video", f, "faces_detected.mp4", "video/mp4")
