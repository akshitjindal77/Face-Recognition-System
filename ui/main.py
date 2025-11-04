# import os, cv2, time, pickle, argparse, glob
# import numpy as np
# import face_recognition

# DATA_DIR = os.path.join("data")
# USERS_DIR = os.path.join(DATA_DIR, "users")
# DB_FILE = os.path.join(DATA_DIR, "encodings.pkl")
# os.makedirs(USERS_DIR, exist_ok=True)

# def _draw_label(frame, text, x, y, color=(0,255,0)):
#     cv2.putText(frame, text, (x, max(20,y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# def register_user(name: str, shots: int = 10, delay: float = 0.25):
#     """
#     Capture several aligned face crops for a given user from webcam.
#     Press 'q' anytime to quit early.
#     """
#     user_dir = os.path.join(USERS_DIR, name)
#     os.makedirs(user_dir, exist_ok=True)
#     cap = cv2.VideoCapture(0)
#     taken = 0
#     print(f"[INFO] Registering '{name}'. Look at the camera. Capturing {shots} face images...")
#     while taken < shots:
#         ok, frame = cap.read()
#         if not ok:
#             print("[ERROR] Camera frame not available.")
#             break
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         boxes = face_recognition.face_locations(rgb, model="hog")  # fast; try 'cnn' if you have CUDA
#         if boxes:
#             (top, right, bottom, left) = boxes[0]
#             cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
#             _draw_label(frame, f"Capturing {taken+1}/{shots}", left, top-10)
#             # Save a cropped, slightly padded face region for consistency
#             pad = 20
#             h, w = frame.shape[:2]
#             y1 = max(0, top - pad); y2 = min(h, bottom + pad)
#             x1 = max(0, left - pad); x2 = min(w, right + pad)
#             face_crop = frame[y1:y2, x1:x2]
#             if face_crop.size > 0:
#                 out_path = os.path.join(user_dir, f"{int(time.time()*1000)}.jpg")
#                 cv2.imwrite(out_path, face_crop)
#                 taken += 1
#                 time.sleep(delay)
#         _draw_label(frame, "Press 'q' to cancel", 10, 30, (0,200,255))
#         cv2.imshow("Register", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"[INFO] Saved {taken} images to {user_dir}")

# def build_database():
#     """
#     Go through all users' folders, compute per-image encodings, and store an
#     average embedding per user. Supports incremental updates.
#     """
#     db = {}
#     user_dirs = [d for d in glob.glob(os.path.join(USERS_DIR, "*")) if os.path.isdir(d)]
#     total_images = 0
#     for udir in user_dirs:
#         name = os.path.basename(udir)
#         encs = []
#         for img_path in glob.glob(os.path.join(udir, "*.jpg")):
#             img = face_recognition.load_image_file(img_path)
#             boxes = face_recognition.face_locations(img, model="hog")
#             if not boxes: 
#                 continue
#             # Use the first detected face
#             enc = face_recognition.face_encodings(img, known_face_locations=[boxes[0]])
#             if enc:
#                 encs.append(enc[0])
#                 total_images += 1
#         if encs:
#             encs = np.array(encs)
#             mean_enc = encs.mean(axis=0)  # average embedding for robustness
#             db[name] = {
#                 "embedding": mean_enc,
#                 "count": len(encs)
#             }
#             print(f"[DB] {name}: {len(encs)} images -> 128-d avg embedding")
#         else:
#             print(f"[WARN] No valid face encodings for {name}.")
#     os.makedirs(DATA_DIR, exist_ok=True)
#     with open(DB_FILE, "wb") as f:
#         pickle.dump(db, f)
#     print(f"[INFO] Database built with {len(db)} users from {total_images} images. -> {DB_FILE}")

# def recognize_live(tolerance: float = 0.45, show_dist: bool = True):
#     """
#     Live recognition from webcam against stored average embeddings.
#     'tolerance' is the max allowed distance (lower is stricter).
#     """
#     if not os.path.exists(DB_FILE):
#         print("[ERROR] No database found. Run: python face_app.py build-db")
#         return
#     with open(DB_FILE, "rb") as f:
#         db = pickle.load(f)
#     if not db:
#         print("[ERROR] Database is empty. Register a user first.")
#         return

#     names = list(db.keys())
#     encs = np.stack([db[n]["embedding"] for n in names], axis=0)

#     cap = cv2.VideoCapture(0)
#     print("[INFO] Recognizing... Press 'q' to quit.")
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             print("[ERROR] Camera frame not available.")
#             break
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         boxes = face_recognition.face_locations(rgb, model="hog")
#         face_encs = face_recognition.face_encodings(rgb, boxes)
#         for (top, right, bottom, left), fenc in zip(boxes, face_encs):
#             # Compare to all user embeddings
#             dists = np.linalg.norm(encs - fenc, axis=1)
#             best_idx = int(np.argmin(dists))
#             best_dist = float(dists[best_idx])
#             label = names[best_idx] if best_dist <= tolerance else "Unknown"
#             color = (0,255,0) if label != "Unknown" else (0,0,255)
#             cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#             text = f"{label}"
#             if show_dist:
#                 text += f" ({best_dist:.2f})"
#             _draw_label(frame, text, left, top-10, color)
#         _draw_label(frame, "Press 'q' to quit", 10, 30, (200,200,0))
#         cv2.imshow("Recognize", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# def list_users():
#     if not os.path.isdir(USERS_DIR):
#         print("[INFO] No users yet.")
#         return
#     users = sorted([os.path.basename(d) for d in glob.glob(os.path.join(USERS_DIR, "*")) if os.path.isdir(d)])
#     if not users:
#         print("[INFO] No users yet.")
#     else:
#         print("[INFO] Users:")
#         for u in users:
#             print(" -", u)

# def delete_user(name: str):
#     udir = os.path.join(USERS_DIR, name)
#     if not os.path.isdir(udir):
#         print(f"[ERROR] '{name}' not found.")
#         return
#     for p in glob.glob(os.path.join(udir, "*.jpg")):
#         os.remove(p)
#     os.rmdir(udir)
#     print(f"[INFO] Deleted user '{name}'. Now rebuild the DB: python face_app.py build-db")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="Simple multi-user face recognition app")
#     sub = ap.add_subparsers(dest="cmd")

#     rg = sub.add_parser("register", help="Register a user via webcam")
#     rg.add_argument("--name", required=True, help="User name (folder-safe)")
#     rg.add_argument("--shots", type=int, default=12, help="Number of face images to capture")
#     rg.add_argument("--delay", type=float, default=0.25, help="Delay between captures (sec)")

#     bd = sub.add_parser("build-db", help="Build or update the encodings database")

#     rc = sub.add_parser("recognize", help="Live recognition")
#     rc.add_argument("--tolerance", type=float, default=0.45, help="Distance threshold; lower is stricter")
#     rc.add_argument("--no-dist", action="store_true", help="Hide distance numbers")

#     ls = sub.add_parser("list", help="List registered users")

#     dl = sub.add_parser("delete", help="Delete a user and their images")
#     dl.add_argument("--name", required=True)

#     args = ap.parse_args()

#     if args.cmd == "register":
#         register_user(args.name, shots=args.shots, delay=args.delay)
#     elif args.cmd == "build-db":
#         build_database()
#     elif args.cmd == "recognize":
#         recognize_live(tolerance=args.tolerance, show_dist=not args.no_dist)
#     elif args.cmd == "list":
#         list_users()
#     elif args.cmd == "delete":
#         delete_user(args.name)
#     else:
#         ap.print_help()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, time, pickle, argparse, glob
import numpy as np
import joblib

# --- New imports (InsightFace + SVM) ---
from insightface.app import FaceAnalysis
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# -------------------------
# Paths / constants
# -------------------------
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

def register_user(name: str, shots: int = 50, delay: float = 0.25, ctx_id: int = 0):
    user_dir = os.path.join(USERS_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    app = _load_app(ctx_id=ctx_id)
    cap = cv2.VideoCapture(0)
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
            _draw_label(frame, f"Capturing {taken+1}/{shots}", x1, y1-10)
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

    for name, img_path in _face_paths():
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        
        if len(faces) != 1:
            continue
        emb = faces[0].normed_embedding  
        if emb is None or emb.size == 0:
            continue

        if name not in db:
            db[name] = {"all_encodings": [], "embedding": None, "count": 0}
        db[name]["all_encodings"].append(emb.astype(np.float32))
        db[name]["count"] += 1
        total_images += 1

    for name, rec in db.items():
        arr = np.vstack(rec["all_encodings"])  
        rec["embedding"] = arr.mean(axis=0)    
        print(f"[DB] {name}: {arr.shape[0]} images -> 512-d mean embedding")

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
    base = LinearSVC()  # linear is all you need with ArcFace embeddings
    clf = CalibratedClassifierCV(base, cv=5)  # probability calibration
    clf.fit(X, y)
    joblib.dump(clf, SVM_FILE)
    print(f"[INFO] Trained SVM on {len(X)} samples across {len(set(y))} users -> {SVM_FILE}")

def recognize_live(thresh_unknown: float = 0.60, show_score: bool = True, ctx_id: int = 0):
    """
    Live recognition using InsightFace embeddings.
    If an SVM model exists, use it with probability threshold for 'Unknown'.
    Otherwise, fall back to cosine similarity vs. per-user means.
    """
    db = _load_db()
    if not db:
        print("[ERROR] No database found. Run: python main.py build-db")
        return

    # Try to load SVM (optional)
    clf = None
    if os.path.exists(SVM_FILE):
        try:
            clf = joblib.load(SVM_FILE)
            classes = clf.classes_
            print(f"[INFO] Using SVM model: {SVM_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to load SVM ({e}). Falling back to cosine matching.")

    # Prepare fallback centroid matrix
    names_centroid = list(db.keys())
    centroids = np.stack([db[n]["embedding"] for n in names_centroid], axis=0)  # (U,512)

    app = _load_app(ctx_id=ctx_id)
    cap = cv2.VideoCapture(0)
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
                    # cosine similarity fallback to user centroids
                    sims = _cosine_sim_matrix(centroids, emb)   # higher is better
                    best_idx = int(np.argmax(sims))
                    score = float(sims[best_idx])
                    # A typical good start threshold for cosine is around 0.60
                    pred = names_centroid[best_idx] if score >= thresh_unknown else "Unknown"
                    label_text = pred
                    color = (0,255,0) if pred != "Unknown" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            text = f"{label_text}"
            if show_score:
                text += f" ({score:.2f})"
            _draw_label(frame, text, x1, y1-10, color)

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
    # delete files
    for p in glob.glob(os.path.join(udir, "*")):
        try:
            os.remove(p)
        except Exception:
            pass
    os.rmdir(udir)
    print(f"[INFO] Deleted user '{name}'. Now rebuild the DB: python main.py build-db")

# -------------------------
# CLI wiring (same commands)
# -------------------------
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

    # Kept the same 'recognize' entry point, but now supports SVM if trained
    rc = sub.add_parser("recognize", help="Live recognition")
    rc.add_argument("--thresh", type=float, default=0.60, help="Unknown threshold (prob/cosine)")
    rc.add_argument("--no-score", action="store_true", help="Hide score numbers")
    rc.add_argument("--cpu", action="store_true", help="Force CPU (ctx_id = -1)")

    # New but optional: train an SVM (you can ignore if you want cosine-only)
    sv = sub.add_parser("train-svm", help="Train SVM on stored embeddings")

    ls = sub.add_parser("list", help="List registered users")
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
