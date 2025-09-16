import cv2
import joblib
import numpy as np
import insightface
import csv
import os
from datetime import datetime, date
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from PIL import Image, ImageEnhance
import random
import io
import time
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
import ssl
from email.message import EmailMessage

# --------------------------------
# Config
# --------------------------------
ATTENDANCE_FILE = "attendance.csv"
COOLDOWN = 2.0  # seconds
MAX_EMBEDDINGS_PER_EMPLOYEE = 60  # Updated to match app.py
DEACTIVATED_EMBEDDINGS_LIMIT = 10  # New constant from app.py
DUPLICATE_THRESHOLD = 0.95  # cosine similarity cutoff for duplicates

# Email notification config via environment variables (set these in your env)
NOTIFY_EMAIL_SENDER = os.getenv("EMAIL_SENDER")
NOTIFY_EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
NOTIFY_EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Import zone tracking functionality
try:
    from zone_tracking import log_employee_detection
    ZONE_TRACKING_ENABLED = True
    print("[INIT] Zone tracking enabled")
except ImportError as e:
    print(f"[WARN] Zone tracking disabled: {e}")
    ZONE_TRACKING_ENABLED = False
    
    def log_employee_detection(employee_name, camera_name, confidence, snapshot_path=None, is_tracked=False):
        """Fallback function when zone tracking is disabled"""
        print(f"[ZONE-FALLBACK] Would log: {employee_name} on {camera_name} (zone tracking disabled)")
        return True

# --- Global processing status for stop mechanism ---
PROCESSING_STATUS = {
    "should_stop": False,
    "is_running": False
}

def set_stop_signal():
    """Signal the webcam processing to stop"""
    global PROCESSING_STATUS
    PROCESSING_STATUS["should_stop"] = True

def reset_processing_status():
    """Reset processing status"""
    global PROCESSING_STATUS
    PROCESSING_STATUS["should_stop"] = False
    PROCESSING_STATUS["is_running"] = False

# --- Employee status store ---
EMPLOYEE_STATUS_FILE = "employee_status.pkl"
if os.path.exists(EMPLOYEE_STATUS_FILE):
    EMPLOYEE_STATUS = joblib.load(EMPLOYEE_STATUS_FILE)
else:
    EMPLOYEE_STATUS = {}

def reload_embeddings_and_models():
    """Helper function to reload embeddings and models from disk"""
    known_embs, known_labels, svm, le = np.empty((0, 512), dtype=np.float32), np.array([], dtype=np.int32), None, LabelEncoder()
    
    try:
        if os.path.exists("train_embeddings.pkl"):
            known_embs, known_labels, le = joblib.load("train_embeddings.pkl")
            known_embs = np.array(known_embs, dtype=np.float32) if len(known_embs) > 0 else np.empty((0, 512), dtype=np.float32)
            known_labels = np.array(known_labels, dtype=np.int32) if len(known_labels) > 0 else np.array([], dtype=np.int32)
        
        if os.path.exists("svm_model.pkl"):
            svm = joblib.load("svm_model.pkl")
        
        if os.path.exists("label_encoder.pkl"):
            le = joblib.load("label_encoder.pkl")
            
    except Exception as e:
        print(f"[ERROR] Failed to reload models: {e}")
        # Return defaults on error
        known_embs = np.empty((0, 512), dtype=np.float32)
        known_labels = np.array([], dtype=np.int32)
        svm = None
        le = LabelEncoder()
    
    return known_embs, known_labels, svm, le

def save_employee_status():
    """Save employee status to disk"""
    joblib.dump(EMPLOYEE_STATUS, EMPLOYEE_STATUS_FILE)

# --------------------------------
# Models load / helpers
# --------------------------------
def load_models():
    providers = ["CPUExecutionProvider"]
    face_model = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_model.prepare(ctx_id=0, det_size=(640, 640))

    if (
        os.path.exists("svm_model.pkl")
        and os.path.exists("label_encoder.pkl")
        and os.path.exists("train_embeddings.pkl")
    ):
        try:
            svm = joblib.load("svm_model.pkl")
        except Exception:
            svm = None
        le = joblib.load("label_encoder.pkl")
        known_embs, known_labels, _ = joblib.load("train_embeddings.pkl")
        known_embs = np.array(known_embs, dtype=np.float32) if len(known_embs) > 0 else np.empty((0, 512), dtype=np.float32)
        known_labels = np.array(known_labels, dtype=np.int32) if len(known_labels) > 0 else np.array([], dtype=np.int32)
    else:
        svm, le = None, LabelEncoder()
        known_embs = np.empty((0, 512), dtype=np.float32)
        known_labels = np.array([], dtype=np.int32)

    return face_model, svm, le, known_embs, known_labels

# --------------------------------
# CSV helpers
# --------------------------------
def init_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Status", "Timestamp"])

def log_attendance(name, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, status, timestamp])

# --------------------------------
# Augmentation
# --------------------------------
def augment_image(img: Image.Image, aug_count=6):
    augmented = []
    for _ in range(aug_count):
        new_img = img.rotate(random.randint(-25, 25))
        if random.random() > 0.5:
            new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Brightness(new_img)
        new_img = enhancer.enhance(random.uniform(0.7, 1.3))
        augmented.append(new_img)
    return augmented

# --------------------------------
# Duplicate check
# --------------------------------
def is_duplicate(new_emb, existing_embs):  
    """
    Return True if new_emb is too similar to any vector in existing_embs.
    """
    if existing_embs is None or len(existing_embs) == 0:
        return False
    sims = cosine_similarity([new_emb], existing_embs)[0]
    return np.max(sims) >= DUPLICATE_THRESHOLD

# --------------------------------
# Email notification utilities
# --------------------------------
def send_email_notification(name, snapshot_path, camera_id="CAM1"):
    """
    Send a simple email with the snapshot attached.
    Requires environment variables:
      NOTIFY_EMAIL_SENDER, NOTIFY_EMAIL_PASSWORD, NOTIFY_EMAIL_RECEIVER
    This function catches exceptions so it won't crash the main loop.
    """
    sender = NOTIFY_EMAIL_SENDER
    password = NOTIFY_EMAIL_PASSWORD
    receiver = NOTIFY_EMAIL_RECEIVER

    if not sender or not password or not receiver:
        print("[EMAIL] Skipping email: email config not set (set environment variables).")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = f"⚠️ Employee {name} detected on {camera_id}"
        msg["From"] = sender
        msg["To"] = receiver

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg.set_content(
            f"Employee: {name}\nCamera: {camera_id}\nTime: {timestamp}\n\nSnapshot attached."
        )

        with open(snapshot_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(snapshot_path)
        msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"[EMAIL] Sent notification for {name} (saved {snapshot_path})")
    except Exception as e:
        print(f"[EMAIL] Failed to send notification: {e}")

def save_snapshot_and_notify(frame, bbox, name, camera_id="CAM1"):
    """
    Crop face from frame, save snapshot with timestamp, and call send_email_notification.
    Non-blocking: exceptions are caught and printed.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # pad bbox slightly but keep inside image bounds
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1s = max(0, x1 - pad_x)
        y1s = max(0, y1 - pad_y)
        x2s = min(w, x2 + pad_x)
        y2s = min(h, y2 + pad_y)
        face_crop = frame[y1s:y2s, x1s:x2s]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{name}_{camera_id}_{timestamp}.jpg"
        path = os.path.join(SNAPSHOT_DIR, fname)
        cv2.imwrite(path, face_crop)
        # Send email (catch exceptions inside)
        send_email_notification(name, path, camera_id=camera_id)
    except Exception as e:
        print(f"[SNAPSHOT] Failed to save/notify snapshot: {e}")

def save_full_frame_and_notify(frame, name, camera_id="CAM1"):
    """
    Save the entire frame and send email notification for the tracked employee.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{name}_{camera_id}_{timestamp}_full.jpg"
        path = os.path.join(SNAPSHOT_DIR, fname)
        cv2.imwrite(path, frame)
        send_email_notification(name, path, camera_id=camera_id)
    except Exception as e:
        print(f"[TRACKING] Failed to save/send full-frame snapshot: {e}")

# --------------------------------
# Add employee (enrollment)
# --------------------------------
def add_employee(face_model, name, image_files, known_embs, known_labels, le):
    """
    image_files: list of image bytes
    known_embs, known_labels: current arrays
    le: LabelEncoder (mutable)
    Returns dict with keys: status, added (num embeddings added), total_embeddings, ...
    """
    new_embeddings = []
    new_labels = []
    added_count = 0

    for idx, file_bytes in enumerate(image_files, start=1):
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Image {idx}: cannot open ({e})")
            continue

        faces = face_model.get(np.array(img))
        if not faces:
            print(f"[SKIP] Image {idx}: No face detected")
            continue

        emb = faces[0].embedding
        if emb is None or emb.shape[0] != 512:
            print(f"[SKIP] Image {idx}: Invalid embedding")
            continue

        # existing embeddings for this employee (if any)
        emp_embs = np.empty((0, 512), dtype=np.float32)
        if name in le.classes_:
            encoded_label = le.transform([name])[0]
            emp_embs = known_embs[known_labels == encoded_label] if known_embs.size else np.empty((0, 512), dtype=np.float32)
            if emp_embs.shape[0] >= MAX_EMBEDDINGS_PER_EMPLOYEE:
                print(f"[SKIP] Image {idx}: Cap reached ({MAX_EMBEDDINGS_PER_EMPLOYEE})")
                continue
            if is_duplicate(emb, emp_embs):
                print(f"[SKIP] Image {idx}: Duplicate embedding")
                continue
            if emp_embs.shape[0] >= MAX_EMBEDDINGS_PER_EMPLOYEE - 5:
                print(f"[WARN] {name} already has {emp_embs.shape[0]} embeddings, close to cap")

        # Add the original image embedding
        new_embeddings.append(emb)
        new_labels.append(name)
        added_count += 1
        print(f"[ADD] Image {idx}: New embedding added")

        # Augmented embeddings
        aug_images = augment_image(img)
        for aug_idx, aug in enumerate(aug_images, start=1):
            try:
                faces_aug = face_model.get(np.array(aug))
                if faces_aug:
                    emb_aug = faces_aug[0].embedding
                    if emb_aug is None or emb_aug.shape[0] != 512:
                        continue
                    # if existing emp_embs present, check duplicate
                    if name in le.classes_ and is_duplicate(emb_aug, emp_embs):
                        print(f"[SKIP] Augmented {idx}.{aug_idx}: Duplicate embedding")
                        continue
                    new_embeddings.append(emb_aug)
                    new_labels.append(name)
                    added_count += 1
                    print(f"[ADD] Augmented {idx}.{aug_idx}: Added")
            except Exception:
                continue

    if added_count == 0:
        return {"status": "error", "message": "No valid/unique embeddings found.", "added": 0}

    # Update LabelEncoder (append if new)
    if name not in le.classes_:
        le_classes = list(le.classes_)
        le_classes.append(name)
        le.classes_ = np.array(le_classes)

    # encode new labels and append embeddings/labels to existing
    encoded_new = le.transform(new_labels)
    new_embeddings = np.array(new_embeddings, dtype=np.float32)
    known_embs_new = np.vstack([known_embs, new_embeddings]) if known_embs.size else np.array(new_embeddings, dtype=np.float32)
    known_labels_new = np.append(known_labels, encoded_new) if known_labels.size else np.array(encoded_new, dtype=np.int32)

    # Retrain SVM immediately
    # Handle the case when there's only one class (employee)
    if len(np.unique(known_labels_new)) > 1:
        svm = SVC(probability=True, kernel="linear")
        svm.fit(known_embs_new, known_labels_new)
    else:
        # For a single class, we can't train an SVM
        # We'll set svm to None and handle this case in recognition
        svm = None
        print("[INFO] Only one class detected, SVM not trained (will use cosine similarity only)")

    # Save
    joblib.dump(svm, "svm_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump([known_embs_new, known_labels_new, le], "train_embeddings.pkl")

    total_for_person = int(np.sum(known_labels_new == encoded_new[0])) if len(encoded_new) > 0 else 0
    print(f"[INFO] Added {added_count} embeddings for {name}. Total now: {total_for_person}")
    if total_for_person >= MAX_EMBEDDINGS_PER_EMPLOYEE - 5:
        print(f"[WARN] {name} nearing cap ({total_for_person}/{MAX_EMBEDDINGS_PER_EMPLOYEE})")

    # Save employee status
    EMPLOYEE_STATUS[name] = "active"
    save_employee_status()

    return {
        "status": "success",
        "employee": name,
        "added": int(added_count),
        "total_embeddings": total_for_person,
    }

# --------------------------------
# Rename employee
# --------------------------------
def rename_employee(old_name, new_name, known_embs, known_labels, le):
    """
    Rename an employee in the system.
    Returns success/error dict.
    """
    global EMPLOYEE_STATUS
    
    if old_name not in le.classes_:
        return {"status": "error", "message": f"Employee '{old_name}' not found"}
    
    if new_name in le.classes_:
        return {"status": "error", "message": f"Employee '{new_name}' already exists"}
    
    try:
        # Update label encoder classes
        classes = list(le.classes_)
        idx = classes.index(old_name)
        classes[idx] = new_name
        le.classes_ = np.array(classes)
        
        # Update employee status
        EMPLOYEE_STATUS[new_name] = EMPLOYEE_STATUS.pop(old_name, "active")
        save_employee_status()
        
        # Save updated models
        joblib.dump(le, "label_encoder.pkl")
        joblib.dump((known_embs, known_labels, le), "train_embeddings.pkl")
        
        print(f"[INFO] Renamed employee '{old_name}' to '{new_name}'")
        return {"status": "success", "message": f"Renamed '{old_name}' to '{new_name}'"}
    except Exception as e:
        print(f"[ERROR] Failed to rename employee: {e}")
        return {"status": "error", "message": f"Failed to rename employee: {str(e)}"}

# --------------------------------
# Deactivate employee (reduce to limited embeddings)
# --------------------------------
def deactivate_employee(name, known_embs, known_labels, le):
    """
    Deactivate an employee by keeping only DEACTIVATED_EMBEDDINGS_LIMIT embeddings.
    Returns success/error dict.
    """
    global EMPLOYEE_STATUS
    
    if name not in le.classes_:
        return {"status": "error", "message": "Employee not found"}
    
    try:
        encoded_label = le.transform([name])[0]
        mask = known_labels == encoded_label
        employee_embs = known_embs[mask]
        
        # Keep only limited embeddings for deactivated employee
        if employee_embs.shape[0] > DEACTIVATED_EMBEDDINGS_LIMIT:
            employee_embs = employee_embs[:DEACTIVATED_EMBEDDINGS_LIMIT]
        
        # Rebuild arrays
        new_embs, new_labels = [], []
        for idx, emb in enumerate(known_embs):
            if known_labels[idx] == encoded_label:
                continue
            new_embs.append(emb)
            new_labels.append(known_labels[idx])
        
        # Add limited embeddings back
        new_embs.extend(employee_embs)
        new_labels.extend([encoded_label] * employee_embs.shape[0])
        
        known_embs_new = np.array(new_embs, dtype=np.float32)
        known_labels_new = np.array(new_labels, dtype=np.int32)
        
        # Save updated data
        joblib.dump((known_embs_new, known_labels_new, le), "train_embeddings.pkl")
        
        # Update employee status
        EMPLOYEE_STATUS[name] = "inactive"
        save_employee_status()
        
        print(f"[INFO] Employee '{name}' deactivated (limited to {DEACTIVATED_EMBEDDINGS_LIMIT} embeddings)")
        return {"status": "success", "message": f"Employee '{name}' deactivated."}
    except Exception as e:
        print(f"[ERROR] Failed to deactivate employee: {e}")
        return {"status": "error", "message": f"Failed to deactivate employee: {str(e)}"}

# --------------------------------
# Activate employee
# --------------------------------
def activate_employee(name, le):
    """
    Reactivate an employee by changing their status to active.
    Returns success/error dict.
    """
    global EMPLOYEE_STATUS
    
    if name not in le.classes_:
        return {"status": "error", "message": "Employee not found"}
    
    EMPLOYEE_STATUS[name] = "active"
    save_employee_status()
    
    print(f"[INFO] Employee '{name}' reactivated")
    return {"status": "success", "message": f"Employee '{name}' reactivated."}

# --------------------------------
# Employee tracking functions
# --------------------------------
def track_employee(name, le):
    """
    Set an employee to be tracked with notifications.
    Returns success/error dict.
    """
    global TRACK_EMPLOYEE_NAME
    
    if name not in le.classes_:
        return {"status": "error", "message": f"Employee '{name}' not found"}
    
    TRACK_EMPLOYEE_NAME = name
    print(f"[INFO] Tracking enabled for '{name}'")
    return {"status": "success", "message": f"Tracking enabled for '{name}'"}

def clear_tracked_employee():
    """
    Clear the currently tracked employee.
    Returns success dict.
    """
    global TRACK_EMPLOYEE_NAME
    
    TRACK_EMPLOYEE_NAME = None
    print(f"[INFO] Tracking cleared")
    return {"status": "success", "message": "Tracking cleared."}

# --------------------------------
# Camera management functions
# --------------------------------
CAMERA_SOURCES_FILE = "camera_sources.pkl"
if os.path.exists(CAMERA_SOURCES_FILE):
    CAMERA_SOURCES = joblib.load(CAMERA_SOURCES_FILE)
else:
    CAMERA_SOURCES = {}  # dict {name: url}

# Add default webcam if not already present
if "webcam" not in CAMERA_SOURCES:
    CAMERA_SOURCES["webcam"] = 0  # Default laptop webcam
    joblib.dump(CAMERA_SOURCES, CAMERA_SOURCES_FILE)
    print("[INFO] Added default 'webcam' camera (index 0)")

def save_camera_sources():
    joblib.dump(CAMERA_SOURCES, CAMERA_SOURCES_FILE)

def add_camera(name, url):
    """
    Add a new camera source.
    Returns success/error dict.
    """
    global CAMERA_SOURCES
    
    if name in CAMERA_SOURCES:
        return {"status": "error", "message": "Camera name already exists"}
    
    try:
        # Basic validation for camera URL/index
        if url.isdigit():
            url = int(url)  # Convert to integer for camera index
        
        CAMERA_SOURCES[name] = url
        save_camera_sources()
        
        print(f"[INFO] Added camera '{name}' with URL '{url}'")
        return {"status": "success", "cameras": CAMERA_SOURCES}
    except Exception as e:
        print(f"[ERROR] Failed to add camera: {e}")
        return {"status": "error", "message": f"Failed to add camera: {str(e)}"}

def remove_camera(name):
    """
    Remove a camera source.
    Returns success/error dict.
    """
    global CAMERA_SOURCES
    
    if name not in CAMERA_SOURCES:
        return {"status": "error", "message": "Camera not found"}
    
    try:
        CAMERA_SOURCES.pop(name)
        save_camera_sources()
        
        print(f"[INFO] Removed camera '{name}'")
        return {"status": "success", "cameras": CAMERA_SOURCES}
    except Exception as e:
        print(f"[ERROR] Failed to remove camera: {e}")
        return {"status": "error", "message": f"Failed to remove camera: {str(e)}"}

def list_cameras():
    """
    List all available cameras.
    Returns camera dict.
    """
    return {"cameras": CAMERA_SOURCES, "count": len(CAMERA_SOURCES)}

# --------------------------------
# Delete employee (hard delete)
# --------------------------------
def delete_employee(name, known_embs, known_labels, le):
    """
    Completely remove an employee's embeddings and label from the encoder.
    Returns success/error dict.
    """
    if name not in le.classes_:
        return {"status": "error", "message": "Employee not found"}

    # get encoded label id
    encoded_label = le.transform([name])[0]

    # remove all embeddings for this label
    mask = known_labels != encoded_label
    known_embs_new = known_embs[mask] if known_embs.size else np.empty((0, 512), dtype=np.float32)
    known_labels_new = known_labels[mask] if known_labels.size else np.array([], dtype=np.int32)

    # remove name from label encoder classes
    remaining = [c for c in le.classes_ if c != name]
    le.classes_ = np.array(remaining)

    # save embeddings + labels + label encoder
    joblib.dump([known_embs_new, known_labels_new, le], "train_embeddings.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # retrain svm if any embeddings left, otherwise remove file
    if known_embs_new.size and known_embs_new.shape[0] > 0:
        svm = SVC(probability=True, kernel="linear")
        svm.fit(known_embs_new, known_labels_new)
        joblib.dump(svm, "svm_model.pkl")
    else:
        if os.path.exists("svm_model.pkl"):
            try:
                os.remove("svm_model.pkl")
            except Exception:
                pass
        svm = None

    print(f"[INFO] Deleted all embeddings for {name}")
    return {"status": "success", "employee": name, "message": "Deleted employee"}

# --------------------------------
# Status / list helpers
# --------------------------------
def get_employee_status(le, known_labels):
    """
    Get employee status including embedding counts and active/inactive status.
    Returns dict with employee info.
    """
    global EMPLOYEE_STATUS
    
    status = {}
    for name in le.classes_:
        encoded_label = le.transform([name])[0]
        count = int(np.sum(known_labels == encoded_label)) if known_labels.size else 0
        emp_status = EMPLOYEE_STATUS.get(name, "active")
        status[name] = {
            "embeddings_count": count,
            "status": emp_status
        }
    return status

def get_single_employee_status(name, le, known_labels):
    """
    Get status for a single employee.
    Returns dict with employee info or error.
    """
    global EMPLOYEE_STATUS
    
    if name not in le.classes_:
        return {"status": "error", "message": "Employee not found"}
    
    encoded_label = le.transform([name])[0]
    count = int(np.sum(known_labels == encoded_label))
    emp_status = EMPLOYEE_STATUS.get(name, "active")
    
    return {
        "employee": name,
        "embeddings_count": count,
        "status": emp_status
    }

def list_employees(le):
    """
    List all employees (active only by default).
    Returns list of active employees.
    """
    global EMPLOYEE_STATUS
    
    employees = list(le.classes_)
    active = [emp for emp in employees if EMPLOYEE_STATUS.get(emp, "active") == "active"]
    return active

def list_all_employees(le):
    """
    List all employees with their status.
    Returns list of dicts with name and status.
    """
    global EMPLOYEE_STATUS
    
    employees = list(le.classes_)
    return [{"name": emp, "status": EMPLOYEE_STATUS.get(emp, "active")} for emp in employees]

# --------------------------------
# Webcam recognition & auto-learning
# --------------------------------
def process_webcam(face_model, svm, le, known_embs, known_labels, camera_name=None, camera_id="CAM1", unknown_thresh=0.50, show_display=False):
    """
    Main webcam loop. If a recognized person is detected with conf >= 0.85,
    it will try to add a new embedding (if not duplicate and under cap),
    retrain immediately, save models, and also save a snapshot + send email notification.
    Includes tracking for a specific employee with full-frame snapshot and email notification.
    
    Args:
        camera_name: Name of camera from CAMERA_SOURCES (e.g., "cafeteria", "office")
                    If None or not found, defaults to camera index 0
        camera_id: Display name for logging/notifications (defaults to "CAM1")
        show_display: Whether to show OpenCV window (False for API/headless mode)
    """
    global PROCESSING_STATUS
    
    # Set processing status
    PROCESSING_STATUS["is_running"] = True
    PROCESSING_STATUS["should_stop"] = False
    
    try:
        # Determine camera source
        camera_source = 0  # Default fallback
        
        if camera_name and camera_name in CAMERA_SOURCES:
            camera_source = CAMERA_SOURCES[camera_name]
            if not camera_id or camera_id == "CAM1":  # Update camera_id to match camera name
                camera_id = camera_name.upper()
            print(f"[INFO] Using named camera '{camera_name}': {camera_source}")
        else:
            if camera_name:
                print(f"[WARN] Camera '{camera_name}' not found in CAMERA_SOURCES, using default (index 0)")
            else:
                print("[INFO] Using default camera (index 0)")
        
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            print(f"⚠️ Could not open camera: {camera_source}")
            return

        seen_counts, last_seen, in_frame = {}, {}, set()
        today = date.today()

        if show_display:
            print(f"[INFO] Starting webcam processing with display window. Press 'q' to quit or call set_stop_signal() to stop.")
        else:
            print(f"[INFO] Starting webcam processing in headless mode (API). Call set_stop_signal() to stop.")
        
        while not PROCESSING_STATUS["should_stop"]:
            if date.today() != today:
                seen_counts.clear()
                last_seen.clear()
                in_frame.clear()
                today = date.today()

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera")
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_model.get(frame_rgb)
                if len(faces) > 0:
                    print(f"[DEBUG] Detected {len(faces)} face(s) in frame")
            except Exception as e:
                print(f"[ERROR] Face detection failed: {e}")
                continue
                
            current_time = time.time()
            current_seen = set()

            for f in faces:
                try:
                    x1, y1, x2, y2 = f.bbox.astype(int)
                    emb = f.embedding
                    if emb is None or emb.shape[0] != 512:
                        continue

                    label, conf = "unknown", 0.0

                    # If we have stored embeddings, compute similarity and SVM fallback
                    if known_embs.size and known_embs.shape[0] > 0:
                        emb_norm = emb / np.linalg.norm(emb)
                        known_norms = known_embs / np.linalg.norm(known_embs, axis=1, keepdims=True)
                        sims = np.dot(known_norms, emb_norm)
                        best_idx = int(np.argmax(sims))
                        best_sim = float(np.max(sims))

                        if best_sim >= unknown_thresh and svm:
                            pred_idx = svm.predict([emb])[0]
                            label = le.inverse_transform([pred_idx])[0]
                            try:
                                conf = float(np.max(svm.predict_proba([emb])[0]))
                            except Exception:
                                conf = best_sim
                            print(f"[RECOGNITION] Detected: {label} (confidence: {conf:.3f})")
                        else:
                            label, conf = "unknown", best_sim
                            if best_sim > 0.3:  # Only log if somewhat similar
                                print(f"[RECOGNITION] Unknown face (best similarity: {best_sim:.3f})")
                    else:
                        print(f"[RECOGNITION] No trained models available - face detected as unknown")

                    current_seen.add(label)

                    # Attendance logging (toggle)
                    if label not in in_frame:
                        last_time = last_seen.get(label, 0)
                        if current_time - last_time > COOLDOWN:
                            count = seen_counts.get(label, 0) + 1
                            seen_counts[label] = count
                            last_seen[label] = current_time
                            status = "inside office" if count % 2 == 1 else "not inside office"
                            log_attendance(label, status)

                    # Auto-embed new samples (self-learning) + immediate retrain + notify
                    if label != "unknown" and conf >= 0.85:
                        encoded_label = le.transform([label])[0]
                        emp_embs = known_embs[known_labels == encoded_label] if known_embs.size else np.empty((0, 512), dtype=np.float32)
                        current_count = emp_embs.shape[0] if emp_embs.size else 0

                        # --- TRACKING FEATURE ---
                        if TRACK_EMPLOYEE_NAME and label == TRACK_EMPLOYEE_NAME:
                            track_key = label + "_track"
                            last_time = last_seen.get(track_key, 0)
                            if current_time - last_time > COOLDOWN:  # Prevent multiple emails
                                last_seen[track_key] = current_time
                                print(f"[TRACK] {label} detected. Sending full-frame snapshot.")
                                save_full_frame_and_notify(frame, label, camera_id=camera_id)
                                log_attendance(label, "entered office")  # Log entry automatically
                        
                        # Zone-based tracking for all employees
                        if ZONE_TRACKING_ENABLED and camera_id:
                            # Save snapshot for zone tracking
                            snapshot_path = None
                            try:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                fname = f"{label}_{camera_id}_{timestamp}.jpg"
                                snapshot_path = os.path.join(SNAPSHOT_DIR, fname)
                                
                                # Crop face for snapshot
                                h, w = frame.shape[:2]
                                pad_x = int((x2 - x1) * 0.1)
                                pad_y = int((y2 - y1) * 0.1)
                                x1s = max(0, x1 - pad_x)
                                y1s = max(0, y1 - pad_y)
                                x2s = min(w, x2 + pad_x)
                                y2s = min(h, y2 + pad_y)
                                face_crop = frame[y1s:y2s, x1s:x2s]
                                cv2.imwrite(snapshot_path, face_crop)
                            except Exception as e:
                                print(f"[SNAPSHOT ERROR] Failed to save snapshot: {e}")
                                snapshot_path = None
                            
                            # Log to zone tracking system
                            is_tracked = (TRACK_EMPLOYEE_NAME == label)
                            log_employee_detection(
                                employee_name=label,
                                camera_name=camera_id.lower(),  # Convert to lowercase to match zone config
                                confidence=conf,
                                snapshot_path=snapshot_path,
                                is_tracked=is_tracked
                            )

                        if current_count < MAX_EMBEDDINGS_PER_EMPLOYEE:
                            if not is_duplicate(emb, emp_embs):
                                # Append new embedding
                                if known_embs.size:
                                    known_embs = np.vstack([known_embs, emb])
                                else:
                                    known_embs = np.array([emb], dtype=np.float32)
                                known_labels = np.append(known_labels, encoded_label) if known_labels.size else np.array([encoded_label], dtype=np.int32)

                                # Retrain SVM
                                try:
                                    svm = SVC(probability=True, kernel="linear")
                                    svm.fit(known_embs, known_labels)

                                    # Persist models
                                    joblib.dump([known_embs, known_labels, le], "train_embeddings.pkl")
                                    joblib.dump(svm, "svm_model.pkl")

                                    total = current_count + 1
                                    print(f"[AUTO-LEARN] Added new embedding for {label}, total now {total}")
                                    if total >= MAX_EMBEDDINGS_PER_EMPLOYEE - 5:
                                        print(f"[WARN] {label} nearing embedding cap ({MAX_EMBEDDINGS_PER_EMPLOYEE})")

                                    # Save snapshot and send notification (non-blocking)
                                    save_snapshot_and_notify(frame, (x1, y1, x2, y2), label, camera_id=camera_id)
                                        
                                except Exception as e:
                                    print(f"[ERROR] Failed to retrain/save models: {e}")
                            else:
                                print(f"[SKIP] Duplicate embedding for {label} (webcam)")
                        else:
                            print(f"[SKIP] {label} reached cap ({MAX_EMBEDDINGS_PER_EMPLOYEE})")

                    # Draw bounding box + label on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process face: {e}")
                    continue

            in_frame = current_seen
            
            # Display frame only if show_display is True (for standalone mode)
            if show_display:
                try:
                    cv2.imshow("Webcam Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] User pressed 'q' to quit")
                        break
                except Exception as e:
                    print(f"[ERROR] Display error: {e}")
                    show_display = False  # Disable display on error
            else:
                # Small delay to prevent CPU overload in headless mode
                time.sleep(0.03)  # ~30 FPS equivalent

    except Exception as e:
        print(f"[ERROR] Webcam processing failed: {e}")
    finally:
        # Cleanup
        PROCESSING_STATUS["is_running"] = False
        try:
            cap.release()
            if show_display:  # Only destroy windows if we created them
                cv2.destroyAllWindows()
        except:
            pass
        print("[INFO] Webcam processing stopped and cleanup completed")

   # ------------------------------
# Main execution (standalone mode)
# ------------------------------
if __name__ == "__main__":
    import sys

    print("[INFO] Starting Face Recognition System...")

    # Load models and embeddings
    face_model, svm, le, known_embs, known_labels = load_models()
    print(f"[INFO] Loaded {len(le.classes_)} employees from embeddings.")

    # Ensure attendance CSV exists
    init_csv()

    # Display employee status
    status = get_employee_status(le, known_labels)
    print("[INFO] Current employee embeddings count:")
    for emp, count in status.items():
        print(f"  {emp}: {count}")

    # Display tracking info
    if TRACK_EMPLOYEE_NAME:
        print(f"[INFO] Tracking employee: {TRACK_EMPLOYEE_NAME}")

    # Display available cameras
    if CAMERA_SOURCES:
        print("[INFO] Available cameras:")
        for name, source in CAMERA_SOURCES.items():
            print(f"  {name}: {source}")
    else:
        print("[INFO] No named cameras configured, using default camera (index 0)")

    # Check for camera name argument
    camera_name = None
    if len(sys.argv) > 1:
        camera_name = sys.argv[1]
        print(f"[INFO] Using camera: {camera_name}")

    # Start webcam loop
    try:
        process_webcam(face_model, svm, le, known_embs, known_labels, 
                      camera_name=camera_name, camera_id="CAM1", show_display=True)
    except KeyboardInterrupt:
        print("\n[INFO] Webcam stopped by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Webcam loop failed: {e}")
        sys.exit(1)

    print("[INFO] Face Recognition System terminated.")

# source /Users/ameermortaza/Desktop/Face-Detection-System/.venv/bin/activate
# to activate the virtual environment
