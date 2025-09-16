# app.py - FastAPI backend for webcam employee recognition
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import joblib
import numpy as np
import os
import cv2
import threading
import time
from datetime import datetime, date, timedelta
from webcam_testing_backend import (
    load_models,
    add_employee,
    get_employee_status,
    get_single_employee_status,
    list_employees,
    list_all_employees,
    rename_employee,
    deactivate_employee,
    activate_employee,
    track_employee,
    clear_tracked_employee,
    add_camera,
    remove_camera,
    list_cameras,
    save_employee_status,
    process_webcam,
    set_stop_signal,
    reset_processing_status,
    PROCESSING_STATUS,
)
from auto_sync_service import start_auto_sync, stop_auto_sync, force_sync

app = FastAPI(title="Webcam Employee Attendance API")

# --- Config ---
MAX_EMBEDDINGS_PER_EMPLOYEE = 60
DEACTIVATED_EMBEDDINGS_LIMIT = 10

# Load models at startup
face_model, svm, le, known_embs, known_labels = load_models()

# Start automatic database synchronization service
print("[STARTUP] Starting automatic database synchronization...")
start_auto_sync()
print("[STARTUP] Auto-sync service started successfully")

# Global processing status tracking
processing_status = {
    "is_running": False,
    "camera_name": None,
    "start_time": None,
    "thread": None
}

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "face-detection-api"
        }
    )

# ---------------- Employee Endpoints ----------------

@app.post("/employee/add")
async def add_employee_endpoint(name: str = Form(...), files: list[UploadFile] = File(...)):
    global known_embs, known_labels, svm, le

    # Check if employee already exists and get current embedding count
    if name in le.classes_:
        try:
            encoded_label = le.transform([name])[0]
            current_count = int(np.sum(known_labels == encoded_label))
        except Exception as e:
            print(f"[WARN] Error checking existing employee count: {e}")
            current_count = 0
    else:
        current_count = 0

    if current_count >= MAX_EMBEDDINGS_PER_EMPLOYEE:
        return {"status": "error", "message": f"Embedding cap reached ({MAX_EMBEDDINGS_PER_EMPLOYEE})."}

    image_bytes = [await file.read() for file in files]
    result = add_employee(face_model, name, image_bytes, known_embs, known_labels, le)

    # Check if add_employee was successful
    if result.get("status") != "success":
        # Return the error result directly
        return result

    # Reload models after adding employee
    if os.path.exists("train_embeddings.pkl"):
        try:
            known_embs, known_labels, _ = joblib.load("train_embeddings.pkl")
            known_embs = np.array(known_embs, dtype=np.float32)
            known_labels = np.array(known_labels, dtype=np.int32)
        except Exception as e:
            print(f"[ERROR] Failed to reload train_embeddings.pkl: {e}")
    if os.path.exists("svm_model.pkl"):
        try:
            svm = joblib.load("svm_model.pkl")
        except Exception as e:
            print(f"[ERROR] Failed to reload svm_model.pkl: {e}")
    if os.path.exists("label_encoder.pkl"):
        try:
            le = joblib.load("label_encoder.pkl")
        except Exception as e:
            print(f"[ERROR] Failed to reload label_encoder.pkl: {e}")

    # **IMPORTANT: Also add employee to MySQL database**
    if result.get("status") == "success":
        try:
            from zone_tracking import zone_tracker
            employee_id = zone_tracker.add_employee_if_not_exists(name)
            if employee_id:
                print(f"[DB] Employee '{name}' automatically added to database with ID: {employee_id}")
            else:
                print(f"[DB WARNING] Failed to add employee '{name}' to database")
        except Exception as e:
            print(f"[DB ERROR] Failed to sync employee '{name}' to database: {e}")
            # Don't fail the entire request if database sync fails
        
        # Also trigger auto-sync to ensure everything is synchronized
        try:
            force_sync()
        except Exception as e:
            print(f"[AUTO-SYNC] Failed to trigger auto-sync: {e}")

    # Get updated count after adding employee
    total_count = 0
    if name in le.classes_:
        try:
            encoded_label = le.transform([name])[0]
            total_count = int(np.sum(known_labels == encoded_label))
        except Exception as e:
            print(f"[WARN] Error getting updated employee count: {e}")
            total_count = result.get("total_embeddings", 0)
    else:
        # If the employee was not added to the label encoder, use the result value
        total_count = result.get("total_embeddings", 0)

    return {
        "employee": name,
        "added": result.get("added", 0) if "added" in result else result.get("added_embeddings", 0),
        "total_embeddings": total_count,
        "status": "active",
        "database_synced": True  # Indicate that employee was synced to database
    }

@app.post("/employee/rename")
async def rename_employee_endpoint(old_name: str = Form(...), new_name: str = Form(...)):
    global known_embs, known_labels, le
    
    result = rename_employee(old_name, new_name, known_embs, known_labels, le)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404 if "not found" in result["message"] else 400, detail=result["message"])
    
    # Reload models after rename
    if os.path.exists("train_embeddings.pkl"):
        known_embs, known_labels, _ = joblib.load("train_embeddings.pkl")
        known_embs = np.array(known_embs, dtype=np.float32)
        known_labels = np.array(known_labels, dtype=np.int32)
    if os.path.exists("label_encoder.pkl"):
        le = joblib.load("label_encoder.pkl")
    
    return result

@app.post("/employee/deactivate")
async def deactivate_employee_endpoint(name: str = Form(...)):
    global known_embs, known_labels, le
    
    result = deactivate_employee(name, known_embs, known_labels, le)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    # Reload models after deactivation
    if os.path.exists("train_embeddings.pkl"):
        known_embs, known_labels, _ = joblib.load("train_embeddings.pkl")
        known_embs = np.array(known_embs, dtype=np.float32)
        known_labels = np.array(known_labels, dtype=np.int32)
    
    return result

@app.post("/employee/activate")
async def activate_employee_endpoint(name: str = Form(...)):
    result = activate_employee(name, le)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.post("/employee/track")
async def track_employee_endpoint(name: str = Form(...)):
    result = track_employee(name, le)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.post("/employee/track/clear")
async def clear_tracked_employee_endpoint():
    return clear_tracked_employee()

@app.get("/employee/status")
async def employee_status_endpoint(name: str = None):
    if name:
        return get_single_employee_status(name, le, known_labels)
    else:
        return get_employee_status(le, known_labels)

@app.get("/employee/list")
async def employee_list_endpoint():
    employees = list_employees(le)
    return {"employees": employees, "count": len(employees)}

@app.get("/employee/list/all")
async def employee_list_all_endpoint():
    return list_all_employees(le)

@app.post("/employee/sync-to-database")
async def sync_employees_to_database():
    """Sync all face recognition employees to MySQL database"""
    try:
        from zone_tracking import zone_tracker
        
        # Get all employees from face recognition system
        face_recognition_employees = list(le.classes_) if hasattr(le, 'classes_') else []
        
        if not face_recognition_employees:
            return {
                "status": "success",
                "message": "No employees found in face recognition system",
                "synced_count": 0,
                "total_employees": 0
            }
        
        synced_count = 0
        failed_employees = []
        
        for employee_name in face_recognition_employees:
            try:
                employee_id = zone_tracker.add_employee_if_not_exists(employee_name)
                if employee_id:
                    synced_count += 1
                    print(f"[SYNC] Employee '{employee_name}' synced to database with ID: {employee_id}")
                else:
                    failed_employees.append(employee_name)
                    print(f"[SYNC ERROR] Failed to sync employee '{employee_name}' to database")
            except Exception as e:
                failed_employees.append(employee_name)
                print(f"[SYNC ERROR] Exception syncing '{employee_name}': {e}")
        
        return {
            "status": "success" if synced_count > 0 else "partial",
            "message": f"Synchronized {synced_count}/{len(face_recognition_employees)} employees to database",
            "synced_count": synced_count,
            "total_employees": len(face_recognition_employees),
            "failed_employees": failed_employees,
            "note": "All existing face recognition employees have been synced to MySQL database"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync employees to database: {str(e)}")

@app.post("/employee/auto-sync/force")
async def force_auto_sync():
    """Force an immediate automatic synchronization"""
    try:
        force_sync()
        return {
            "status": "success",
            "message": "Forced synchronization completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Force sync failed: {str(e)}")

@app.get("/employee/auto-sync/status")
async def auto_sync_status():
    """Get the status of the automatic synchronization service"""
    from auto_sync_service import auto_sync_service
    return {
        "auto_sync_enabled": auto_sync_service.running,
        "check_interval_seconds": auto_sync_service.check_interval,
        "last_employee_count": len(auto_sync_service.last_employee_set),
        "monitored_files": auto_sync_service.model_files
    }



# ---------------- Camera Endpoints ----------------

@app.post("/camera/add")
async def add_camera_endpoint(name: str = Form(...), url: str = Form(...)):
    return add_camera(name, url)

@app.delete("/camera/remove")
async def remove_camera_endpoint(name: str = Form(...)):
    result = remove_camera(name)
    
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    
    return result

@app.get("/camera/list")
async def list_cameras_endpoint():
    from webcam_testing_backend import CAMERA_SOURCES
    return {
        "cameras": CAMERA_SOURCES, 
        "count": len(CAMERA_SOURCES),
        "available_names": list(CAMERA_SOURCES.keys()),
        "note": "Use camera names with /camera/process and /camera/stream endpoints"
    }

def gen_frames_with_recognition(url, face_model, svm, le, known_embs, known_labels):
    """Generate video frames with face recognition and bounding boxes for streaming"""
    print(f"[STREAM] Starting video stream with face recognition for camera: {url}")
    cap = None
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[STREAM] ERROR: Could not open camera {url}")
            return
        
        print(f"[STREAM] Camera {url} opened successfully with face recognition")
        frame_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                print(f"[STREAM] Failed to read frame {frame_count} from camera {url}")
                break
                
            frame_count += 1
            
            # Perform face recognition on the frame
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_model.get(frame_rgb)
                
                # Process each detected face
                for f in faces:
                    try:
                        x1, y1, x2, y2 = f.bbox.astype(int)
                        emb = f.embedding
                        if emb is None or emb.shape[0] != 512:
                            continue

                        label, conf = "unknown", 0.0
                        unknown_thresh = 0.50

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
                            else:
                                label, conf = "unknown", best_sim

                        # Draw bounding box and label on frame
                        color = (0, 255, 0) if label != "unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with confidence
                        label_text = f"{label} ({conf:.2f})"
                        font_scale = 0.7
                        thickness = 2
                        
                        # Get text size to create background
                        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                        
                        # Draw text
                        cv2.putText(
                            frame,
                            label_text,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                        )
                        
                        if frame_count % 30 == 0:  # Log every 30 frames
                            print(f"[STREAM] Detected: {label} (confidence: {conf:.3f})")
                            
                    except Exception as e:
                        print(f"[STREAM] Error processing face: {e}")
                        continue
                        
            except Exception as e:
                print(f"[STREAM] Face detection failed: {e}")
            
            # Encode frame as JPEG with recognition results
            try:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception as e:
                print(f"[STREAM] Error encoding frame: {e}")
                break
                
    except Exception as e:
        print(f"[STREAM] Streaming error: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()


def gen_frames_simple(url):
    """Generate simple video frames for streaming - NO face recognition"""
    print(f"[STREAM] Starting simple video stream for camera: {url}")
    cap = None
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[STREAM] ERROR: Could not open camera {url}")
            return
        
        print(f"[STREAM] Camera {url} opened successfully")
        frame_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                print(f"[STREAM] Failed to read frame {frame_count} from camera {url}")
                break
                
            frame_count += 1
            if frame_count % 100 == 0:  # Log every 100 frames to reduce spam
                print(f"[STREAM] Streaming frame {frame_count} from camera {url}")
            
            # Encode frame as JPEG
            try:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            except Exception as e:
                print(f"[STREAM] Error encoding frame: {e}")
                break
                
    except Exception as e:
        print(f"[STREAM] Streaming error: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        print(f"[STREAM] Camera {url} simple stream ended")

@app.get("/camera/stream/{name}")
async def stream_camera(name: str):
    """Stream video from a named camera (simple stream without recognition)"""
    from webcam_testing_backend import CAMERA_SOURCES
    
    print(f"[API] Simple stream request for camera: {name}")
    
    if name not in CAMERA_SOURCES:
        available_cameras = list(CAMERA_SOURCES.keys())
        error_msg = f"Camera '{name}' not found. Available cameras: {available_cameras}"
        print(f"[API] {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    camera_source = CAMERA_SOURCES[name]
    print(f"[API] Simple streaming camera '{name}' from source: {camera_source}")
    
    try:
        return StreamingResponse(
            gen_frames_simple(camera_source), 
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        print(f"[API] Streaming error for camera '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@app.get("/camera/stream/{name}/recognition")
async def stream_camera_with_recognition(name: str):
    """Stream video from a named camera WITH face recognition and bounding boxes"""
    from webcam_testing_backend import CAMERA_SOURCES
    global face_model, svm, le, known_embs, known_labels
    
    print(f"[API] Recognition stream request for camera: {name}")
    
    if name not in CAMERA_SOURCES:
        available_cameras = list(CAMERA_SOURCES.keys())
        error_msg = f"Camera '{name}' not found. Available cameras: {available_cameras}"
        print(f"[API] {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)
    
    camera_source = CAMERA_SOURCES[name]
    print(f"[API] Recognition streaming camera '{name}' from source: {camera_source}")
    
    try:
        return StreamingResponse(
            gen_frames_with_recognition(camera_source, face_model, svm, le, known_embs, known_labels), 
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        print(f"[API] Recognition streaming error for camera '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Recognition streaming failed: {str(e)}")

def run_webcam_processing(camera_name=None):
    """Background function to run webcam processing"""
    global face_model, svm, le, known_embs, known_labels, processing_status
    
    try:
        print(f"[INFO] Starting webcam processing with camera: {camera_name or 'default'}")
        processing_status["is_running"] = True
        processing_status["camera_name"] = camera_name
        processing_status["start_time"] = time.time()
        
        # Call the actual webcam processing function
        process_webcam(
            face_model=face_model,
            svm=svm, 
            le=le,
            known_embs=known_embs,
            known_labels=known_labels,
            camera_name=camera_name,
            camera_id=camera_name.upper() if camera_name else "CAM1",
            show_display=False  # No display in API mode
        )
    except Exception as e:
        print(f"[ERROR] Webcam processing failed: {e}")
    finally:
        processing_status["is_running"] = False
        processing_status["camera_name"] = None
        processing_status["start_time"] = None
        processing_status["thread"] = None
        print("[INFO] Webcam processing stopped")

@app.post("/camera/process")
async def process_camera_endpoint(camera_name: str = Form(None)):
    """Start webcam processing with optional camera name"""
    global processing_status
    
    # Check if already running using the backend status
    if PROCESSING_STATUS["is_running"]:
        return {
            "status": "error",
            "message": f"Processing already running on camera: {processing_status.get('camera_name', 'unknown')}",
            "current_camera": processing_status.get("camera_name"),
            "running_since": processing_status.get("start_time")
        }
    
    # Validate camera if specified
    from webcam_testing_backend import CAMERA_SOURCES
    if camera_name:
        if camera_name not in CAMERA_SOURCES:
            available_cameras = list(CAMERA_SOURCES.keys())
            raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found. Available cameras: {available_cameras}")
    else:
        # If no camera_name provided, use default 'webcam'
        camera_name = "webcam"
    
    # Reset any previous stop signals
    reset_processing_status()
    
    # Start processing in background thread
    thread = threading.Thread(target=run_webcam_processing, args=(camera_name,), daemon=True)
    processing_status["thread"] = thread
    thread.start()
    
    camera_source = CAMERA_SOURCES.get(camera_name, 0) if camera_name else 0
    
    return {
        "status": "success",
        "message": f"Webcam processing started with camera: {camera_name or 'default'}",
        "camera_name": camera_name,
        "camera_source": camera_source,
        "note": "Face recognition is now running. Use /camera/stop endpoint to stop processing"
    }

@app.post("/camera/stop")
async def stop_camera_processing():
    """Stop webcam processing"""
    global processing_status
    
    if not PROCESSING_STATUS["is_running"]:
        return {
            "status": "error",
            "message": "No webcam processing is currently running"
        }
    
    # Send stop signal to the processing loop
    set_stop_signal()
    
    # Wait a bit for graceful shutdown
    time.sleep(1)
    
    return {
        "status": "success",
        "message": "Stop signal sent. Processing will stop gracefully.",
        "note": "The webcam processing has been signaled to stop"
    }

@app.get("/camera/status")
async def get_camera_processing_status():
    """Get current webcam processing status"""
    global processing_status
    
    if PROCESSING_STATUS["is_running"]:
        uptime = time.time() - processing_status.get("start_time", time.time())
        return {
            "status": "running",
            "camera_name": processing_status.get("camera_name"),
            "uptime_seconds": round(uptime, 2),
            "start_time": processing_status.get("start_time"),
            "should_stop": PROCESSING_STATUS["should_stop"]
        }
    else:
        return {
            "status": "stopped",
            "camera_name": None,
            "uptime_seconds": 0,
            "should_stop": False
        }

# ---------------- Dashboard & Statistics Endpoints ----------------

@app.get("/dashboard/statistics")
async def get_dashboard_statistics():
    """Get comprehensive dashboard statistics for the system"""
    from zone_tracking import zone_tracker
    from datetime import datetime, date as date_obj, timedelta
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        today = date_obj.today()
        yesterday = today - timedelta(days=1)
        this_week_start = today - timedelta(days=today.weekday())
        this_month_start = today.replace(day=1)
        
        # System Overview
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status = 'active'")
        total_active_employees = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status = 'inactive'")
        total_inactive_employees = cursor.fetchone()[0]
        
        # Today's Activity
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_name) 
            FROM employee_zone_events 
            WHERE DATE(timestamp) = %s AND zone_name = 'office' AND action_type = 'entry'
        """, (today,))
        employees_present_today = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM employee_zone_events 
            WHERE DATE(timestamp) = %s
        """, (today,))
        total_events_today = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_name) 
            FROM employee_zone_events 
            WHERE DATE(timestamp) = %s AND zone_name = 'cafeteria'
        """, (today,))
        cafeteria_visits_today = cursor.fetchone()[0]
        
        # Working Hours Statistics
        cursor.execute("""
            SELECT 
                AVG(working_hours_minutes) as avg_working_hours,
                AVG(total_cafeteria_minutes) as avg_break_time,
                COUNT(*) as employees_with_data
            FROM daily_attendance_summary 
            WHERE date = %s
        """, (today,))
        working_hours_stats = cursor.fetchone()
        avg_working_hours = working_hours_stats[0] / 60 if working_hours_stats[0] else 0
        avg_break_time = working_hours_stats[1] / 60 if working_hours_stats[1] else 0
        employees_with_data = working_hours_stats[2]
        
        # Weekly Comparison
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_name) 
            FROM employee_zone_events 
            WHERE DATE(timestamp) >= %s AND zone_name = 'office' AND action_type = 'entry'
        """, (this_week_start,))
        unique_employees_this_week = cursor.fetchone()[0]
        
        # Recent Activity (last 10 events)
        cursor.execute("""
            SELECT employee_name, camera_name, zone_name, action_type, timestamp
            FROM employee_zone_events 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent_events = cursor.fetchall()
        
        recent_activity = []
        for event in recent_events:
            employee, camera, zone, action, timestamp = event
            recent_activity.append({
                "employee": employee,
                "camera": camera,
                "zone": zone,
                "action": action,
                "timestamp": timestamp.isoformat(),
                "time": timestamp.strftime("%H:%M:%S"),
                "date": timestamp.strftime("%Y-%m-%d")
            })
        
        # Overstay Alerts
        cursor.execute("""
            SELECT COUNT(*) 
            FROM cafeteria_overstay_alerts 
            WHERE DATE(alert_sent_time) = %s
        """, (today,))
        overstay_alerts_today = cursor.fetchone()[0]
        
        # Top Active Employees Today
        cursor.execute("""
            SELECT employee_name, COUNT(*) as event_count
            FROM employee_zone_events 
            WHERE DATE(timestamp) = %s
            GROUP BY employee_name 
            ORDER BY event_count DESC 
            LIMIT 5
        """, (today,))
        top_active_employees = cursor.fetchall()
        
        top_employees = []
        for emp_data in top_active_employees:
            employee, count = emp_data
            top_employees.append({
                "employee": employee,
                "events_count": count
            })
        
        # Zone Activity Distribution
        cursor.execute("""
            SELECT zone_name, action_type, COUNT(*) as count
            FROM employee_zone_events 
            WHERE DATE(timestamp) = %s
            GROUP BY zone_name, action_type
            ORDER BY zone_name, action_type
        """, (today,))
        zone_activity = cursor.fetchall()
        
        zone_stats = {}
        for zone_data in zone_activity:
            zone, action, count = zone_data
            if zone not in zone_stats:
                zone_stats[zone] = {}
            zone_stats[zone][action] = count
        
        return {
            "dashboard_data": {
                "system_overview": {
                    "total_active_employees": total_active_employees,
                    "total_inactive_employees": total_inactive_employees,
                    "total_employees": total_active_employees + total_inactive_employees
                },
                "todays_activity": {
                    "employees_present": employees_present_today,
                    "total_events": total_events_today,
                    "cafeteria_visits": cafeteria_visits_today,
                    "overstay_alerts": overstay_alerts_today,
                    "date": str(today)
                },
                "working_hours_summary": {
                    "average_working_hours": round(avg_working_hours, 2),
                    "average_break_hours": round(avg_break_time, 2),
                    "employees_with_data": employees_with_data,
                    "productivity_rate": round((avg_working_hours / (avg_working_hours + avg_break_time) * 100) if (avg_working_hours + avg_break_time) > 0 else 0, 1)
                },
                "weekly_overview": {
                    "unique_employees_this_week": unique_employees_this_week,
                    "week_start": str(this_week_start)
                },
                "top_active_employees_today": top_employees,
                "zone_activity_today": zone_stats,
                "recent_activity": recent_activity
            },
            "generated_at": datetime.now().isoformat(),
            "date": str(today)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard statistics: {str(e)}")
    finally:
        conn.close()

@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get a quick summary of key metrics"""
    from zone_tracking import zone_tracker
    from datetime import date as date_obj
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        today = date_obj.today()
        
        # Quick stats
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status = 'active'")
        active_employees = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_name) 
            FROM employee_zone_events 
            WHERE DATE(timestamp) = %s AND zone_name = 'office'
        """, (today,))
        employees_today = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT AVG(working_hours_minutes) 
            FROM daily_attendance_summary 
            WHERE date = %s
        """, (today,))
        avg_hours_result = cursor.fetchone()[0]
        avg_working_hours = round(avg_hours_result / 60, 1) if avg_hours_result else 0
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM cafeteria_overstay_alerts 
            WHERE DATE(alert_sent_time) = %s
        """, (today,))
        alerts_today = cursor.fetchone()[0]
        
        return {
            "summary": {
                "active_employees": active_employees,
                "employees_present_today": employees_today,
                "average_working_hours_today": avg_working_hours,
                "overstay_alerts_today": alerts_today,
                "attendance_rate": round((employees_today / active_employees * 100) if active_employees > 0 else 0, 1)
            },
            "date": str(today)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
    finally:
        conn.close()

@app.post("/dashboard/refresh")
async def refresh_dashboard_data():
    """Force refresh dashboard data by clearing any potential caching and forcing fresh queries"""
    from zone_tracking import zone_tracker
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        # Force auto-sync to ensure database is in sync with face recognition system
        try:
            force_sync()
        except Exception as e:
            print(f"[DASHBOARD] Warning: Auto-sync failed during refresh: {e}")
        
        return {
            "status": "success",
            "message": "Dashboard data refresh initiated successfully",
            "note": "Dashboard endpoints will now return fresh data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh dashboard data: {str(e)}")
    finally:
        conn.close()

@app.post("/dashboard/cleanup")
async def cleanup_dashboard_data(days_to_keep: int = 7):
    """Clean up old dashboard data, keeping only the specified number of days"""
    from zone_tracking import zone_tracker
    from datetime import datetime, timedelta
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old employee_zone_events
        cursor.execute("""
            DELETE FROM employee_zone_events 
            WHERE timestamp < %s
        """, (cutoff_date,))
        deleted_events = cursor.rowcount
        
        # Clean up old daily_attendance_summary
        cursor.execute("""
            DELETE FROM daily_attendance_summary 
            WHERE date < %s
        """, (cutoff_date.date(),))
        deleted_summaries = cursor.rowcount
        
        # Clean up old cafeteria_overstay_alerts
        cursor.execute("""
            DELETE FROM cafeteria_overstay_alerts 
            WHERE alert_sent_time < %s
        """, (cutoff_date,))
        deleted_alerts = cursor.rowcount
        
        conn.commit()
        
        # Force auto-sync to ensure database is in sync with face recognition system
        try:
            force_sync()
        except Exception as e:
            print(f"[DASHBOARD] Warning: Auto-sync failed during cleanup: {e}")
        
        return {
            "status": "success",
            "message": f"Dashboard data cleanup completed successfully",
            "deleted_records": {
                "employee_zone_events": deleted_events,
                "daily_attendance_summaries": deleted_summaries,
                "cafeteria_overstay_alerts": deleted_alerts
            },
            "cutoff_date": cutoff_date.isoformat(),
            "note": "Old data has been removed. Dashboard endpoints will now return fresh data."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup dashboard data: {str(e)}")
    finally:
        conn.close()

@app.post("/dashboard/sync-employees")
async def sync_employees_to_dashboard():
    """Synchronize employee data between face recognition system and database, then refresh dashboard"""
    global known_embs, known_labels, le
    
    try:
        # Get current employees from face recognition system
        face_recognition_employees = list(le.classes_) if hasattr(le, 'classes_') else []
        
        if not face_recognition_employees:
            return {
                "status": "warning",
                "message": "No employees found in face recognition system",
                "synced_count": 0,
                "total_employees": 0
            }
        
        from zone_tracking import zone_tracker
        conn = zone_tracker.get_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            # Get all employees from database
            cursor.execute("SELECT id, name, status FROM employees")
            db_employees = {row[1]: {'id': row[0], 'status': row[2]} for row in cursor.fetchall()}
            
            # Track changes
            added_count = 0
            updated_count = 0
            removed_count = 0
            
            # Create a set of face recognition employees for efficient lookup
            face_set = set(face_recognition_employees)
            
            # 1. Add or update employees from face recognition system
            for employee_name in face_recognition_employees:
                if employee_name not in db_employees:
                    # Add new employee
                    cursor.execute("""
                        INSERT INTO employees (name, status) 
                        VALUES (%s, 'active')
                    """, (employee_name,))
                    added_count += 1
                    print(f"[SYNC] Added employee '{employee_name}' to database")
                elif db_employees[employee_name]['status'] != 'active':
                    # Reactivate employee
                    cursor.execute("""
                        UPDATE employees 
                        SET status = 'active' 
                        WHERE name = %s
                    """, (employee_name,))
                    updated_count += 1
                    print(f"[SYNC] Reactivated employee '{employee_name}' in database")
            
            # 2. Remove employees that are in database but not in face recognition system
            for db_employee_name, db_employee_data in db_employees.items():
                if db_employee_name not in face_set:
                    # Completely remove employee from database
                    employee_id = db_employee_data['id']
                    cursor.execute("DELETE FROM employee_zone_events WHERE employee_id = %s", (employee_id,))
                    cursor.execute("DELETE FROM daily_attendance_summary WHERE employee_id = %s", (employee_id,))
                    cursor.execute("DELETE FROM cafeteria_overstay_alerts WHERE employee_id = %s", (employee_id,))
                    cursor.execute("DELETE FROM employees WHERE id = %s", (employee_id,))
                    removed_count += 1
                    print(f"[SYNC] Removed employee '{db_employee_name}' from database")
            
            conn.commit()
            
            # Force auto-sync to ensure everything is synchronized
            try:
                force_sync()
            except Exception as e:
                print(f"[SYNC] Warning: Auto-sync failed: {e}")
            
            return {
                "status": "success",
                "message": f"Employee synchronization completed.",
                "changes": {
                    "added": added_count,
                    "updated": updated_count,
                    "removed": removed_count
                },
                "face_recognition_employees": len(face_recognition_employees),
                "database_employees_before": len(db_employees),
                "database_employees_after": len(face_recognition_employees),
                "note": "Database is now fully synchronized with face recognition system."
            }
            
        finally:
            conn.close()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync employees: {str(e)}")

# ---------------- Zone Management Endpoints ----------------

@app.post("/camera/setup/zones")
async def setup_zone_cameras():
    """Set up all 4 zone cameras with default configurations"""
    from webcam_testing_backend import CAMERA_SOURCES
    
    # Default zone camera configurations
    zone_cameras = {
        'office_entry': 0,      # Default webcam for office entry
        'office_exit': 0,       # Default webcam for office exit  
        'cafeteria_entry': 0,   # Default webcam for cafeteria entry
        'cafeteria_exit': 0     # Default webcam for cafeteria exit
    }
    
    success_count = 0
    results = []
    
    for camera_name, camera_url in zone_cameras.items():
        try:
            result = add_camera(camera_name, camera_url)
            if result.get("status") == "success":
                success_count += 1
            results.append({"camera": camera_name, "result": result})
        except Exception as e:
            results.append({"camera": camera_name, "result": {"status": "error", "message": str(e)}})
    
    return {
        "status": "success" if success_count == len(zone_cameras) else "partial",
        "message": f"Zone camera setup completed. {success_count}/{len(zone_cameras)} cameras configured successfully.",
        "cameras_configured": success_count,
        "total_cameras": len(zone_cameras),
        "details": results,
        "note": "All zone cameras are set to default webcam (0). Update individual cameras as needed."
    }

@app.get("/zone/cameras/status")
async def get_zone_cameras_status():
    """Check status of all zone cameras"""
    from webcam_testing_backend import CAMERA_SOURCES
    from zone_tracking import ZONE_CONFIG
    
    zone_camera_status = []
    available_zones = list(ZONE_CONFIG.keys())
    
    for zone_name in available_zones:
        if zone_name in CAMERA_SOURCES:
            camera_url = CAMERA_SOURCES[zone_name]
            status = "configured"
        else:
            camera_url = None
            status = "not_configured"
        
        zone_info = ZONE_CONFIG[zone_name]
        zone_camera_status.append({
            "zone_name": zone_name,
            "zone": zone_info['zone'],
            "action": zone_info['action'],
            "camera_url": camera_url,
            "status": status
        })
    
    configured_count = sum(1 for camera in zone_camera_status if camera["status"] == "configured")
    
    return {
        "zone_cameras": zone_camera_status,
        "total_zones": len(available_zones),
        "configured_zones": configured_count,
        "all_configured": configured_count == len(available_zones),
        "available_zone_names": available_zones
    }

# ---------------- Zone Reports & Analytics Endpoints ----------------

@app.get("/zone/report/{employee_name}")
async def get_zone_report(employee_name: str, date: str = None):
    """Get daily zone activity report for an employee"""
    from zone_tracking import zone_tracker
    from datetime import datetime, date as date_obj
    
    try:
        if date:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            target_date = date_obj.today()
        
        # Get daily zone report (this function seems to be missing from zone_tracking.py)
        # Let's implement it here using the zone_tracker connection
        conn = zone_tracker.get_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        try:
            cursor = conn.cursor()
            
            # Get all zone events for the employee on the specified date
            cursor.execute("""
                SELECT camera_name, zone_name, action_type, timestamp, confidence, snapshot_path
                FROM employee_zone_events 
                WHERE employee_name = %s AND DATE(timestamp) = %s
                ORDER BY timestamp
            """, (employee_name, target_date))
            
            events = cursor.fetchall()
            
            # Process events into a more readable format
            zone_events = []
            for event in events:
                camera, zone, action, timestamp, confidence, snapshot = event
                zone_events.append({
                    "camera_name": camera,
                    "zone_name": zone,
                    "action_type": action,
                    "timestamp": timestamp.isoformat(),
                    "time": timestamp.strftime("%H:%M:%S"),
                    "confidence": float(confidence),
                    "snapshot_path": snapshot
                })
            
            # Calculate metrics
            office_entries = [e for e in zone_events if e['zone_name'] == 'office' and e['action_type'] == 'entry']
            office_exits = [e for e in zone_events if e['zone_name'] == 'office' and e['action_type'] == 'exit']
            cafeteria_entries = [e for e in zone_events if e['zone_name'] == 'cafeteria' and e['action_type'] == 'entry']
            cafeteria_exits = [e for e in zone_events if e['zone_name'] == 'cafeteria' and e['action_type'] == 'exit']
            
            first_entry = office_entries[0] if office_entries else None
            last_exit = office_exits[-1] if office_exits else None
            
            return {
                "employee": employee_name,
                "date": str(target_date),
                "events": zone_events,
                "summary": {
                    "total_events": len(zone_events),
                    "office_entries": len(office_entries),
                    "office_exits": len(office_exits),
                    "cafeteria_visits": len(cafeteria_entries),
                    "cafeteria_exits": len(cafeteria_exits),
                    "first_office_entry": first_entry['time'] if first_entry else None,
                    "last_office_exit": last_exit['time'] if last_exit else None
                }
            }
            
        finally:
            conn.close()
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get zone report: {str(e)}")

@app.get("/attendance/calculate/{employee_name}")
async def calculate_attendance(employee_name: str, date: str = None):
    """Get comprehensive daily attendance summary with working hours"""
    from zone_tracking import zone_tracker
    from datetime import datetime, date as date_obj
    
    try:
        if date:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            target_date = date_obj.today()
        
        summary = zone_tracker.calculate_daily_attendance_summary(employee_name, target_date)
        
        if not summary:
            raise HTTPException(status_code=404, detail=f"No attendance data found for {employee_name} on {target_date}")
        
        # Convert datetime objects to strings for JSON serialization
        if summary.get('first_office_entry'):
            summary['first_office_entry'] = summary['first_office_entry'].isoformat()
        if summary.get('last_office_exit'):
            summary['last_office_exit'] = summary['last_office_exit'].isoformat()
        
        # Convert session times to strings
        for session in summary.get('office_sessions', []):
            if session.get('entry'):
                session['entry'] = session['entry'].isoformat()
            if session.get('exit'):
                session['exit'] = session['exit'].isoformat()
        
        for session in summary.get('cafeteria_sessions', []):
            if session.get('entry'):
                session['entry'] = session['entry'].isoformat()
            if session.get('exit'):
                session['exit'] = session['exit'].isoformat()
        
        return summary
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate attendance: {str(e)}")

@app.get("/attendance/weekly/{employee_name}")
async def get_weekly_attendance(employee_name: str, start_date: str = None):
    """Get weekly attendance summary for an employee"""
    from zone_tracking import zone_tracker
    from datetime import datetime, date as date_obj, timedelta
    
    try:
        if start_date:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            # Get Monday of current week
            today = date_obj.today()
            start_date_obj = today - timedelta(days=today.weekday())
        
        weekly_summary = zone_tracker.get_weekly_summary(employee_name, start_date_obj)
        
        if not weekly_summary:
            raise HTTPException(status_code=404, detail=f"No weekly data found for {employee_name}")
        
        # Convert datetime objects to strings for JSON serialization
        for daily in weekly_summary.get('daily_summaries', []):
            if daily.get('first_office_entry'):
                daily['first_office_entry'] = daily['first_office_entry'].isoformat()
            if daily.get('last_office_exit'):
                daily['last_office_exit'] = daily['last_office_exit'].isoformat()
            
            # Convert session times
            for session in daily.get('office_sessions', []):
                if session.get('entry'):
                    session['entry'] = session['entry'].isoformat()
                if session.get('exit'):
                    session['exit'] = session['exit'].isoformat()
            
            for session in daily.get('cafeteria_sessions', []):
                if session.get('entry'):
                    session['entry'] = session['entry'].isoformat()
                if session.get('exit'):
                    session['exit'] = session['exit'].isoformat()
        
        return weekly_summary
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get weekly attendance: {str(e)}")

@app.get("/zone/events/recent")
async def get_recent_zone_events(limit: int = 50):
    """Get recent zone events across all cameras"""
    from zone_tracking import zone_tracker
    
    if limit > 200:
        limit = 200  # Prevent excessive queries
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT employee_name, camera_name, zone_name, action_type, 
                   timestamp, confidence, snapshot_path
            FROM employee_zone_events 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (limit,))
        
        events = cursor.fetchall()
        
        recent_events = []
        for event in events:
            employee, camera, zone, action, timestamp, confidence, snapshot = event
            recent_events.append({
                "employee_name": employee,
                "camera_name": camera,
                "zone_name": zone,
                "action_type": action,
                "timestamp": timestamp.isoformat(),
                "time": timestamp.strftime("%H:%M:%S"),
                "date": timestamp.strftime("%Y-%m-%d"),
                "confidence": float(confidence),
                "snapshot_path": snapshot
            })
        
        return {
            "recent_events": recent_events,
            "count": len(recent_events),
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent events: {str(e)}")
    finally:
        conn.close()

@app.get("/zone/overstays/alerts")
async def get_overstay_alerts(days: int = 7):
    """Get cafeteria overstay alerts for the specified number of days"""
    from zone_tracking import zone_tracker
    from datetime import datetime, timedelta
    
    if days > 30:
        days = 30  # Prevent excessive queries
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        
        # Get alerts from the last N days
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT employee_name, entry_time, alert_sent_time, overstay_duration_minutes
            FROM cafeteria_overstay_alerts 
            WHERE alert_sent_time >= %s
            ORDER BY alert_sent_time DESC
        """, (since_date,))
        
        alerts = cursor.fetchall()
        
        overstay_alerts = []
        for alert in alerts:
            employee, entry_time, alert_time, duration = alert
            overstay_alerts.append({
                "employee_name": employee,
                "entry_time": entry_time.isoformat(),
                "alert_sent_time": alert_time.isoformat(),
                "overstay_duration_minutes": duration,
                "overstay_duration_hours": round(duration / 60, 2),
                "date": alert_time.strftime("%Y-%m-%d")
            })
        
        # Get summary statistics
        total_alerts = len(overstay_alerts)
        unique_employees = len(set(alert["employee_name"] for alert in overstay_alerts))
        avg_duration = sum(alert["overstay_duration_minutes"] for alert in overstay_alerts) / total_alerts if total_alerts > 0 else 0
        
        return {
            "overstay_alerts": overstay_alerts,
            "summary": {
                "total_alerts": total_alerts,
                "unique_employees": unique_employees,
                "average_overstay_minutes": round(avg_duration, 1),
                "average_overstay_hours": round(avg_duration / 60, 2),
                "days_covered": days
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get overstay alerts: {str(e)}")
    finally:
        conn.close()

# ---------------- Database Management Endpoints ----------------

@app.post("/database/migrate")
async def run_database_migration():
    """Run database migration to add working hours support"""
    try:
        # Import and run the migration function
        from migrate_working_hours import migrate_database
        
        success = migrate_database()
        
        if success:
            return {
                "status": "success",
                "message": "Database migration completed successfully",
                "note": "Working hours calculations are now available"
            }
        else:
            raise HTTPException(status_code=500, detail="Database migration failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration error: {str(e)}")

@app.get("/database/verify")
async def verify_database():
    """Verify database connection and table structure"""
    from zone_tracking import zone_tracker
    
    conn = zone_tracker.get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = conn.cursor()
        
        # Check required tables
        required_tables = ['employees', 'employee_zone_events', 'daily_attendance_summary', 'cafeteria_overstay_alerts']
        table_status = {}
        
        for table in required_tables:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            result = cursor.fetchone()
            table_status[table] = "exists" if result else "missing"
        
        # Get basic statistics
        cursor.execute("SELECT COUNT(*) FROM employees")
        employee_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM employee_zone_events WHERE DATE(timestamp) = CURDATE()")
        todays_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM daily_attendance_summary WHERE date = CURDATE()")
        todays_summaries = cursor.fetchone()[0]
        
        all_tables_exist = all(status == "exists" for status in table_status.values())
        
        return {
            "database_status": "healthy" if all_tables_exist else "issues_found",
            "tables": table_status,
            "statistics": {
                "total_employees": employee_count,
                "todays_events": todays_events,
                "todays_summaries": todays_summaries
            },
            "all_tables_exist": all_tables_exist
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database verification failed: {str(e)}")
    finally:
        conn.close()

# ---------------- Zone Detection Logging Endpoint ----------------

@app.post("/zone/detect")
async def log_zone_detection(employee_name: str = Form(...), camera_name: str = Form(...), confidence: float = Form(...), snapshot_path: str = Form(None)):
    """Manually log a zone detection event (for testing or external integration)"""
    from zone_tracking import log_employee_detection
    from webcam_testing_backend import TRACK_EMPLOYEE_NAME
    
    # Check if employee is being tracked for notifications
    is_tracked = (TRACK_EMPLOYEE_NAME == employee_name)
    
    success = log_employee_detection(employee_name, camera_name, confidence, snapshot_path, is_tracked)
    
    if success:
        return {
            "status": "success",
            "message": f"Zone detection logged: {employee_name} on {camera_name}",
            "employee": employee_name,
            "camera": camera_name,
            "confidence": confidence,
            "tracked": is_tracked,
            "snapshot": snapshot_path
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to log zone detection")

# ---------------- Root ----------------

@app.get("/")
async def root():
    return {
        "message": "Webcam Employee Attendance API is running",
        "features": [
            "Employee Management",
            "Camera Management", 
            "Face Recognition Streaming",
            "Zone-Based Attendance Tracking",
            "Working Hours Calculation",
            "Email Notifications",
            "Overstay Monitoring"
        ],
        "api_docs": "/docs",
        "zone_cameras": "/zone/cameras/status",
        "setup_zones": "/camera/setup/zones"
    }
#to run this code use the command: uvicorn app:app --reload
#to make it available for all the devices in the network use: uvicorn app:app --host 0.0.0.0 --port 8000
#for collegues to access it use using my IP address followed by the port number (http://192.168.0.106:8000/docs)
#this makes it useable for all the devices in the network
#the most optimal way would be that that the backend is hosted on a server and the frontend is hosted on a web server
# or we can host the backend on a cloud service like AWS, GCP, Azure etc.
#that way the laptop would not act as a server and would not need to be on all the time
#I will be adding role based access system in the next update which will only allow the admin to access the options, and the employees would only be able to
#zzga xklt dqjc vdrt (app password)
#export EMAIL_SENDER="mortazaameer8@gmail.com"
#export EMAIL_PASSWORD="zzga xklt dqjc vdrt"  # Gmail App Password
#export EMAIL_RECEIVER="localhostlogin22@gmail.com"
#cd /Users/ameermortaza/Desktop/Face-Detection-System && export MYSQL_HOST=localhost && export MYSQL_PORT=3306 && export MYSQL_DATABASE=face_recognition_attendance && export MYSQL_USER=root && export MYSQL_PASSWORD=Mortaza@348 && .venv/bin/python -m uvicorn app:app --reload
