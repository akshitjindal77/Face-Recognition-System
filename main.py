#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, time, pickle, argparse, glob
import numpy as np
import joblib

from insightface.app import FaceAnalysis
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans

DATA_DIR   = os.path.join("data")
USERS_DIR  = os.path.join(DATA_DIR, "users")
DB_FILE    = os.path.join(DATA_DIR, "insightface_db.pkl")
SVM_FILE   = os.path.join(DATA_DIR, "svm_insightface.pkl")
MODEL_PACK = "buffalo_l"
DET_SIZE   = (640, 640)
os.makedirs(USERS_DIR, exist_ok=True)

def _draw_label(frame, text, x, y, color=(0,255,0)):
    cv2.putText(frame, text, (x, max(20,y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def _load_app(ctx_id=0):
    app = FaceAnalysis(name=MODEL_PACK)
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    return app

def _embed_image(app: FaceAnalysis, img: np.ndarray):
    """
    Returns a (512,) flip-averaged, L2-normalized embedding or None.
    emb = avg(embedding(img), embedding(flipped_img))
    """
    faces = app.get(img)
    if len(faces) != 1:
        return None
    emb = faces[0].normed_embedding
    if emb is None or emb.size == 0:
        return None

    img_flip = cv2.flip(img, 1)
    faces_flip = app.get(img_flip)
    if faces_flip:
        emb_flip = faces_flip[0].normed_embedding
        if emb_flip is not None and emb_flip.size > 0:
            emb = (emb + emb_flip) / 2.0

    return emb.astype(np.float32)

def register_user(name: str, shots: int = 50, delay: float = 0.25, ctx_id: int = 0):
    user_dir = os.path.join(USERS_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    app = _load_app(ctx_id=ctx_id)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    taken = 0
    print(f"[INFO] Registering '{name}'. Look at the camera. Capturing {shots} face images...")
    while taken < shots:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera frame not available.")
            break

        faces = app.get(frame)
        if faces:
            f = faces[0]
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            _draw_label(frame, f"Capturing {taken+1}/{shots}", x1, max(20, y1-10))
            pad = 20
            h, w = frame.shape[:2]
            px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
            px2 = min(w, x2 + pad); py2 = min(h, y2 + pad)
            face_crop = frame[py1:py2, px1:px2]

            if face_crop.size > 0:
                out_path = os.path.join(user_dir, f"{int(time.time()*1000)}.jpg")
                cv2.imwrite(out_path, face_crop)
                taken += 1
                time.sleep(delay)

        _draw_label(frame, "Press 'q' to cancel", 10, 30, (0,200,255))
        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {taken} images to {user_dir}")

    if taken > 0:
        print(f"[INFO] Updating database automatically for '{name}'...")
        app = _load_app(ctx_id)
        _update_single_user_in_db(app, name)
    else:
        print("[WARN] No images were captured, database not updated.")

def _face_paths():
    user_dirs = [d for d in glob.glob(os.path.join(USERS_DIR, "*")) if os.path.isdir(d)]
    for udir in user_dirs:
        name = os.path.basename(udir)
        for img_path in glob.glob(os.path.join(udir, "*.*")):
            yield name, img_path

def build_database(ctx_id: int = 0):
    app = _load_app(ctx_id=ctx_id)
    db = {}
    total_images = 0

    # collect per-image embeddings
    for name, img_path in _face_paths():
        img = cv2.imread(img_path)
        if img is None:
            continue
        emb = _embed_image(app, img)
        if emb is None:
            continue

        if name not in db:
            db[name] = {"all_encodings": [], "centroids": [], "embedding": None, "count": 0}
        db[name]["all_encodings"].append(emb)
        db[name]["count"] += 1
        total_images += 1

    # cluster per user to create multiple centroids (handles "multiple looks")
    for name, rec in db.items():
        arr = np.vstack(rec["all_encodings"])
        k = 3 if arr.shape[0] >= 30 else (2 if arr.shape[0] >= 15 else 1)
        if k == 1:
            centroids = [arr.mean(axis=0)]
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(arr)
            centroids = [arr[labels == i].mean(axis=0) for i in range(k)]

        rec["centroids"] = centroids
        rec["embedding"] = arr.mean(axis=0)
        print(f"[DB] {name}: {arr.shape[0]} images -> {len(centroids)} centroid(s)")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)
    print(f"[INFO] Database built with {len(db)} users from {total_images} images. -> {DB_FILE}")
    print("[HINT] Train the SVM for best accuracy: python main.py train-svm")

def _load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "rb") as f:
        return pickle.load(f)

def _save_db(db):
    """Helper to save the database pickle file."""
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def _update_single_user_in_db(app: FaceAnalysis, user_name: str):
    """
    After registering a user, compute embeddings for that user's folder and
    update the database immediately (without rebuilding everyone).
      - flip-averaged per-image embeddings
      - cluster per-user into centroids (1..3)
    """
    user_dir = os.path.join(USERS_DIR, user_name)
    if not os.path.isdir(user_dir):
        print(f"[WARN] No folder found for '{user_name}'")
        return

    db = _load_db()
    all_vecs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        for img_path in glob.glob(os.path.join(user_dir, ext)):
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = _embed_image(app, img)
            if emb is not None:
                all_vecs.append(emb)

    if not all_vecs:
        print(f"[WARN] No valid faces found for '{user_name}' — database not updated.")
        return

    arr = np.vstack(all_vecs)
    k = 3 if arr.shape[0] >= 30 else (2 if arr.shape[0] >= 15 else 1)
    if k == 1:
        centroids = [arr.mean(axis=0)]
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(arr)
        centroids = [arr[labels == i].mean(axis=0) for i in range(k)]

    db[user_name] = {
        "all_encodings": all_vecs,
        "centroids": centroids,
        "embedding": arr.mean(axis=0),
        "count": len(all_vecs),
    }
    _save_db(db)
    print(f"[INFO] Database updated automatically for '{user_name}' ({len(all_vecs)} images).")

def _cosine_sim_matrix(A, b):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return A @ b

def train_svm():
    db = _load_db()
    if not db:
        print("[ERROR] No database found. Run: python main.py build-db")
        return

    X, y = [], []
    for name, rec in db.items():
        for e in rec.get("all_encodings", []):
            X.append(e)
            y.append(name)
    if len(set(y)) < 2:
        print("[ERROR] Need at least 2 users to train SVM.")
        return

    X = np.array(X, dtype=np.float32)
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, cv=5)
    clf.fit(X, y)
    joblib.dump(clf, SVM_FILE)
    print(f"[INFO] Trained SVM on {len(X)} samples across {len(set(y))} users -> {SVM_FILE}")

def recognize_live(thresh_unknown: float = 0.60, show_score: bool = True, ctx_id: int = 0):
    """
    Live recognition using InsightFace embeddings.
    If an SVM model exists, use it with probability threshold for 'Unknown'.
    Otherwise, fall back to cosine similarity vs. per-user **centroids** (multi-look).
    """
    db = _load_db()
    if not db:
        print("[ERROR] No database found. Run: python main.py build-db")
        return

    clf = None
    classes = None
    if os.path.exists(SVM_FILE):
        try:
            clf = joblib.load(SVM_FILE)
            if hasattr(clf, "predict_proba"):
                classes = clf.classes_
                print(f"[INFO] Using SVM model: {SVM_FILE}")
            else:
                print("[WARN] Loaded SVM has no predict_proba; using cosine matching.")
                clf = None
        except Exception as e:
            print(f"[WARN] Failed to load SVM ({e}). Falling back to cosine matching.")

    names_flat, centroids = [], []
    for uname, rec in db.items():
        c_list = rec.get("centroids") or [rec["embedding"]]
        for c in c_list:
            names_flat.append(uname)
            centroids.append(c)
    if not centroids:
        print("[ERROR] No centroids available in DB.")
        return
    centroids = np.stack(centroids, axis=0)

    app = _load_app(ctx_id=ctx_id)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Recognizing... Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Camera frame not available.")
            break

        faces = app.get(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            emb = f.normed_embedding
            label_text = "Unknown"
            score = 0.0
            color = (0,0,255)

            if emb is not None and emb.size > 0:
                if clf is not None:
                    probs = clf.predict_proba([emb])[0]
                    best_idx = int(np.argmax(probs))
                    score = float(probs[best_idx])
                    pred = classes[best_idx] if score >= thresh_unknown else "Unknown"
                    label_text = pred
                    color = (0,255,0) if pred != "Unknown" else (0,0,255)
                else:
                    sims = _cosine_sim_matrix(centroids, emb)   # higher is better
                    best_idx = int(np.argmax(sims))
                    score = float(sims[best_idx])
                    pred = names_flat[best_idx] if score >= thresh_unknown else "Unknown"
                    label_text = pred
                    color = (0,255,0) if pred != "Unknown" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            text = f"{label_text}"
            if show_score:
                text += f" ({score:.2f})"
            _draw_label(frame, text, x1, max(20, y1-10), color)

        _draw_label(frame, "Press 'q' to quit", 10, 30, (200,200,0))
        cv2.imshow("Recognize", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def list_users():
    if not os.path.isdir(USERS_DIR):
        print("[INFO] No users yet.")
        return
    users = sorted([os.path.basename(d) for d in glob.glob(os.path.join(USERS_DIR, "*")) if os.path.isdir(d)])
    if not users:
        print("[INFO] No users yet.")
    else:
        for u in users:
            print("-", u)

def delete_user(name: str):
    udir = os.path.join(USERS_DIR, name)
    if not os.path.isdir(udir):
        print(f"[ERROR] User '{name}' not found.")
        return
    for p in glob.glob(os.path.join(udir, "*")):
        try:
            os.remove(p)
        except Exception:
            pass
    os.rmdir(udir)
    print(f"[INFO] Deleted user '{name}'. Now rebuild the DB: python main.py build-db")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Face recognition app (InsightFace under the hood)")
    sub = ap.add_subparsers(dest="cmd")

    rg = sub.add_parser("register", help="Register a user via webcam")
    rg.add_argument("--name", required=True, help="User name (folder-safe)")
    rg.add_argument("--shots", type=int, default=12, help="Number of face images to capture")
    rg.add_argument("--delay", type=float, default=0.25, help="Delay between captures (sec)")
    rg.add_argument("--cpu", action="store_true", help="Force CPU (ctx_id = -1)")

    bd = sub.add_parser("build-db", help="Build or update the embeddings database")
    bd.add_argument("--cpu", action="store_true", help="Force CPU (ctx_id = -1)")

    rc = sub.add_parser("recognize", help="Live recognition")
    rc.add_argument("--thresh", type=float, default=0.60, help="Unknown threshold (prob/cosine)")
    rc.add_argument("--no-score", action="store_true", help="Hide score numbers")
    rc.add_argument("--cpu", action="store_true", help="Force CPU (ctx_id = -1)")

    sub.add_parser("train-svm", help="Train SVM on stored embeddings")

    sub.add_parser("list", help="List registered users")

    dl = sub.add_parser("delete", help="Delete a user and their images")
    dl.add_argument("--name", required=True)

    args = ap.parse_args()
    ctx_id = -1 if getattr(args, "cpu", False) else 0

    if args.cmd == "register":
        register_user(args.name, shots=args.shots, delay=args.delay, ctx_id=ctx_id)
    elif args.cmd == "build-db":
        build_database(ctx_id=ctx_id)
    elif args.cmd == "recognize":
        recognize_live(thresh_unknown=args.thresh, show_score=not args.no_score, ctx_id=ctx_id)
    elif args.cmd == "train-svm":
        train_svm()
    elif args.cmd == "list":
        list_users()
    elif args.cmd == "delete":
        delete_user(args.name)
    else:
        ap.print_help()
