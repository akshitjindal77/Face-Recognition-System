import os, time, glob, pickle
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import joblib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# InsightFace (detector + alignment + ArcFace embeddings)
from insightface.app import FaceAnalysis

# -------------------- Config & paths --------------------
BASE_DIR  = Path(__file__).parent.resolve()
DATA_DIR  = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
DB_FILE   = DATA_DIR / "insightface_db.pkl"     # new DB format (stores all_encodings + mean)
SVM_FILE  = DATA_DIR / "svm_insightface.pkl"    # optional trained classifier
MODEL_PACK = "buffalo_l"
DET_SIZE   = (640, 640)

USERS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Utilities --------------------
def draw_label(img, text, x, y, color=(0, 255, 0)):
    cv2.putText(img, text, (x, max(20, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def load_db():
    if DB_FILE.exists():
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def list_users():
    return sorted([p.name for p in USERS_DIR.iterdir() if p.is_dir()])

def delete_user(name: str):
    udir = USERS_DIR / name
    if udir.is_dir():
        for p in udir.glob("*"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            udir.rmdir()
        except OSError:
            pass

def ensure_cam(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

@st.cache_resource(show_spinner=False)
def load_face_app(ctx_id: int):
    """Load InsightFace (cached by Streamlit). ctx_id=0 GPU, -1 CPU."""
    app = FaceAnalysis(name=MODEL_PACK)
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    return app

def cosine_sim_matrix(A, b):
    """Cosine similarity between rows of A (n,d) and a single vector b (d,)."""
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return A @ b

def build_database(app: FaceAnalysis):
    """
    For each user's saved images, compute per-image InsightFace normed embeddings,
    store all vectors + per-user mean embedding.
    """
    db = {}
    total = 0
    user_dirs = [p for p in USERS_DIR.iterdir() if p.is_dir()]
    for udir in user_dirs:
        name = udir.name
        all_vecs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            for img_path in udir.glob(ext):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                faces = app.get(img)
                if len(faces) != 1:
                    # retry on a slightly larger view (helps on tight crops)
                    h, w = img.shape[:2]
                    scale = 800 / max(h, w)
                    if scale > 1.0:
                        img_big = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
                        faces = app.get(img_big)
                        if len(faces) == 1:
                            img = img_big
                if len(faces) != 1:
                    continue
                emb = faces[0].normed_embedding
                if emb is None or emb.size == 0:
                    continue
                all_vecs.append(emb.astype(np.float32))
                total += 1
        if all_vecs:
            arr = np.vstack(all_vecs)
            mean_vec = arr.mean(axis=0)
            db[name] = {
                "all_encodings": all_vecs,
                "embedding": mean_vec,
                "count": len(all_vecs),
            }
    save_db(db)
    return db, total

def update_user_in_db(app: FaceAnalysis, user_name: str):
    """
    After registering a user, compute their embeddings and update the DB immediately.
    """
    udir = USERS_DIR / user_name
    if not udir.is_dir():
        st.warning(f"No folder found for user {user_name}")
        return

    db = load_db()
    all_vecs = []
    total = 0

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        for img_path in udir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = app.get(img)
            if len(faces) != 1:
                continue
            emb = faces[0].normed_embedding
            if emb is None or emb.size == 0:
                continue
            all_vecs.append(emb.astype(np.float32))
            total += 1

    if all_vecs:
        arr = np.vstack(all_vecs)
        mean_vec = arr.mean(axis=0)
        db[user_name] = {
            "all_encodings": all_vecs,
            "embedding": mean_vec,
            "count": len(all_vecs),
        }
        save_db(db)
        st.success(f"✅ Database updated automatically for '{user_name}' ({len(all_vecs)} images).")
    else:
        st.warning(f"No valid faces found for '{user_name}'. Try re-registering.")

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Face ID (InsightFace)", layout="centered")
st.title("Face Recognition Demo (InsightFace)")
st.caption(f"Data directory: {DATA_DIR}")
st.caption("Register users → Database updates automatically → Recognize in real time")

with st.sidebar:
    st.header("Controls")
    cam_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    use_cpu = st.checkbox("Force CPU (uncheck for GPU if available)", value=True)
    ctx_id = -1 if use_cpu else 0

    thresh = st.slider("Unknown threshold", 0.40, 0.90, 0.60, 0.01)
    shots = st.number_input("Shots per registration", 4, 60, 40, 1)
    delay = st.slider("Delay between shots (sec)", 0.05, 1.0, 0.25, 0.05)

    st.divider()
    # DB actions
    if st.button("Rebuild All (optional)"):
        face_app = load_face_app(ctx_id)
        with st.spinner("Rebuilding database..."):
            db, total = build_database(face_app)
        st.success(f"DB rebuilt: {len(db)} users, {total} images. File: {DB_FILE}")

    users = list_users()
    if st.button("Show Users"):
        if not users:
            st.info("No users yet")
        else:
            lines = []
            for u in users:
                cnt = sum(1 for _ in (USERS_DIR / u).glob("*.*"))
                lines.append(f"{u}  —  {cnt} files")
            st.info("\n".join(lines))

tab1, tab2, tab3 = st.tabs(["Register", "Database (optional)", "Recognize"])

# ------------- Tab 1: Register -------------
with tab1:
    st.subheader("Register a new user")
    colA, colB = st.columns([2, 1])
    with colA:
        name = st.text_input("User name (folder-safe)", key="reg_name", placeholder="e.g., Akshit")
    with colB:
        do_delete = st.checkbox("Delete user first (if exists)", value=False)

    start_reg = st.button("Start capture")
    ph = st.empty()
    status = st.empty()

    if start_reg:
        if not name.strip():
            st.error("Please enter a name")
        else:
            user_dir = USERS_DIR / name.strip()
            if do_delete and user_dir.exists():
                delete_user(name.strip())
                user_dir.mkdir(parents=True, exist_ok=True)
            else:
                user_dir.mkdir(parents=True, exist_ok=True)

            face_app = load_face_app(ctx_id)
            cap = ensure_cam(cam_index)
            if cap is None:
                st.error("Camera could not be opened. Try a different index or check permissions.")
            else:
                taken = 0
                last = time.time()
                st.info("Center your face in the green box; keep gentle pose/lighting changes.")
                while taken < shots:
                    ok, frame = cap.read()
                    if not ok:
                        status.error("Camera read failed")
                        break

                    faces = face_app.get(frame)
                    if faces:
                        f = faces[0]
                        x1, y1, x2, y2 = map(int, f.bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        draw_label(frame, f"Capturing {taken+1}/{shots}", x1, y1 - 10)

                        if (time.time() - last) >= delay:
                            pad = 40
                            h, w = frame.shape[:2]
                            px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
                            px2 = min(w, x2 + pad); py2 = min(h, y2 + pad)
                            face_crop = frame[py1:py2, px1:px2]
                            if face_crop.size > 0:
                                out_path = user_dir / f"{int(time.time()*1000)}.jpg"
                                cv2.imwrite(str(out_path), face_crop)
                                taken += 1
                                last = time.time()

                    draw_label(frame, "Close tab or wait to finish.", 10, 30, (0, 200, 255))
                    ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                cap.release()
                if taken > 0:
                    status.success(f"Saved {taken} images to {user_dir}. Updating database automatically...")
                    with st.spinner("Computing embeddings and updating database..."):
                        update_user_in_db(face_app, name.strip())
                else:
                    status.warning("No images captured. Database not updated.")

# ------------- Tab 2: Database (optional) -------------
with tab2:
    st.subheader("Rebuild / Inspect Database")
    st.write("Automatic updates happen after each registration. Use this only to rebuild all users.")
    if st.button("Rebuild Database Now", key="build_now"):
        face_app = load_face_app(ctx_id)
        with st.spinner("Building database..."):
            db, total = build_database(face_app)
        st.success(f"DB rebuilt: {len(db)} users, {total} images. File: {DB_FILE}")
    st.caption("Tip: Use if you've manually deleted or moved user folders.")

# ------------- Tab 3: Recognize -------------
with tab3:
    st.subheader("Live Recognition")
    col1, col2 = st.columns(2)
    with col1:
        start_rec = st.button("Start recognition")
    with col2:
        stop_rec = st.button("Stop recognition")

    if "recognizing" not in st.session_state:
        st.session_state.recognizing = False

    if start_rec:
        st.session_state.recognizing = True
    if stop_rec:
        st.session_state.recognizing = False

    out = st.empty()
    note = st.empty()

    if st.session_state.recognizing:
        db = load_db()
        if not db:
            st.error("No database found. Build it in the 'Database' tab.")
        else:
            names = list(db.keys())
            centroids = np.stack([db[n]["embedding"] for n in names], axis=0) if names else None

            clf = None
            classes = None
            if SVM_FILE.exists():
                try:
                    clf = joblib.load(SVM_FILE)
                    classes = clf.classes_
                    st.info("Using SVM model with probability thresholding.")
                except Exception as e:
                    st.warning(f"Failed to load SVM ({e}). Falling back to cosine similarity.")

            face_app = load_face_app(ctx_id)
            cap = ensure_cam(cam_index)
            if cap is None:
                st.error("Camera could not be opened. Try a different index or check permissions.")
            else:
                note.info("Press 'Stop recognition' to end.")
                while st.session_state.recognizing:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    faces = face_app.get(frame)
                    for f in faces:
                        x1, y1, x2, y2 = map(int, f.bbox)
                        emb = f.normed_embedding
                        label = "Unknown"
                        score = 0.0
                        color = (0, 0, 255)

                        if emb is not None and emb.size > 0:
                            if clf is not None:
                                probs = clf.predict_proba([emb])[0]
                                best_idx = int(np.argmax(probs))
                                score = float(probs[best_idx])
                                pred = classes[best_idx] if score >= thresh else "Unknown"
                                label = pred
                                color = (0, 255, 0) if pred != "Unknown" else (0, 0, 255)
                            elif centroids is not None and len(names) > 0:
                                sims = cosine_sim_matrix(centroids, emb)
                                best_idx = int(np.argmax(sims))
                                score = float(sims[best_idx])
                                pred = names[best_idx] if score >= thresh else "Unknown"
                                label = pred
                                color = (0, 255, 0) if pred != "Unknown" else (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        draw_label(frame, f"{label} ({score:.2f})", x1, y1 - 10, color)

                    out.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    time.sleep(0.01)

                cap.release()
                note.info("Stopped.")
