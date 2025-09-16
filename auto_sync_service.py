#!/usr/bin/env python3
"""
Automatic Database Synchronization Service
Monitors face recognition system for changes and automatically syncs to database
"""

import os
import time
import threading
import joblib
import pymysql
from datetime import datetime
from pathlib import Path

class AutoSyncService:
    def __init__(self, check_interval=30):  # Check every 30 seconds
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        
        # Database configuration - standardized to use environment variables
        self.db_config = {
            'host': os.getenv('MYSQL_HOST', 'localhost'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'database': os.getenv('MYSQL_DATABASE', 'face_recognition_attendance'),
            'user': os.getenv('MYSQL_USER', 'root'),
            'password': os.getenv('MYSQL_PASSWORD', 'Mortaza@348'),
            'charset': 'utf8mb4',
            'autocommit': True,
            'connect_timeout': 10,
            'read_timeout': 10,
            'write_timeout': 10
        }
        
        # File monitoring
        self.model_files = [
            'label_encoder.pkl',
            'train_embeddings.pkl',
            'svm_model.pkl'
        ]
        
        self.last_modified_times = {}
        self.last_employee_set = set()
        
        # Initialize tracking
        self._update_file_timestamps()
        self._update_employee_set()
        
    def _update_file_timestamps(self):
        """Update the last modified timestamps for model files"""
        for file_path in self.model_files:
            if os.path.exists(file_path):
                self.last_modified_times[file_path] = os.path.getmtime(file_path)
            else:
                self.last_modified_times[file_path] = 0
                
    def _update_employee_set(self):
        """Update the current set of employees from face recognition system"""
        try:
            if os.path.exists('label_encoder.pkl'):
                le = joblib.load('label_encoder.pkl')
                self.last_employee_set = set(le.classes_) if hasattr(le, 'classes_') else set()
            else:
                self.last_employee_set = set()
        except Exception as e:
            print(f"[AUTO-SYNC] Error loading employees: {e}")
            self.last_employee_set = set()
    
    def _files_changed(self):
        """Check if any model files have been modified"""
        for file_path in self.model_files:
            if os.path.exists(file_path):
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > self.last_modified_times.get(file_path, 0):
                    return True
        return False
    
    def _employees_changed(self):
        """Check if the employee set has changed"""
        try:
            if os.path.exists('label_encoder.pkl'):
                le = joblib.load('label_encoder.pkl')
                current_employees = set(le.classes_) if hasattr(le, 'classes_') else set()
                return current_employees != self.last_employee_set
        except Exception:
            return False
        return False
    
    def _get_face_recognition_employees(self):
        """Get current employees from face recognition system"""
        try:
            le = joblib.load('label_encoder.pkl')
            return list(le.classes_) if hasattr(le, 'classes_') else []
        except Exception as e:
            print(f"[AUTO-SYNC] Error loading face recognition data: {e}")
            return []
    
    def _get_database_employees(self):
        """Get current employees from database"""
        try:
            connection = pymysql.connect(**self.db_config)
            with connection.cursor() as cursor:
                cursor.execute("SELECT name FROM employees WHERE status = 'active'")
                result = cursor.fetchall()
                return [row[0] for row in result]
        except Exception as e:
            print(f"[AUTO-SYNC] Error connecting to database: {e}")
            return []
        finally:
            if 'connection' in locals():
                connection.close()
    
    def _sync_employees_to_database(self, employees):
        """Sync employees to database (add missing ones)"""
        if not employees:
            return 0
            
        try:
            connection = pymysql.connect(**self.db_config)
            synced_count = 0
            
            with connection.cursor() as cursor:
                for employee_name in employees:
                    try:
                        # Check if employee already exists
                        cursor.execute(
                            "SELECT id FROM employees WHERE name = %s AND status = 'active'",
                            (employee_name,)
                        )
                        if cursor.fetchone():
                            continue  # Employee already exists
                        
                        # Add new employee
                        cursor.execute(
                            "INSERT INTO employees (name, status, created_at) VALUES (%s, %s, %s)",
                            (employee_name, 'active', datetime.now())
                        )
                        synced_count += 1
                        print(f"[AUTO-SYNC] Added employee: {employee_name}")
                        
                    except Exception as e:
                        print(f"[AUTO-SYNC] Error adding employee {employee_name}: {e}")
                        continue
                
                connection.commit()
                
            return synced_count
            
        except Exception as e:
            print(f"[AUTO-SYNC] Database sync error: {e}")
            return 0
        finally:
            if 'connection' in locals():
                connection.close()
    
    def _full_sync_employees(self, face_employees):
        """Fully synchronize employees between face recognition system and database"""
        if not face_employees:
            return {"added": 0, "updated": 0, "removed": 0}
        
        try:
            connection = pymysql.connect(**self.db_config)
            
            with connection.cursor() as cursor:
                # Get all employees from database
                cursor.execute("SELECT id, name, status FROM employees")
                db_employees = {row[1]: {'id': row[0], 'status': row[2]} for row in cursor.fetchall()}
                
                # Track changes
                added_count = 0
                updated_count = 0
                removed_count = 0
                
                # Create a set of face recognition employees for efficient lookup
                face_set = set(face_employees)
                
                # 1. Add or update employees from face recognition system
                for employee_name in face_employees:
                    if employee_name not in db_employees:
                        # Add new employee
                        cursor.execute("""
                            INSERT INTO employees (name, status) 
                            VALUES (%s, 'active')
                        """, (employee_name,))
                        added_count += 1
                        print(f"[AUTO-SYNC] Added employee '{employee_name}' to database")
                    elif db_employees[employee_name]['status'] != 'active':
                        # Reactivate employee
                        cursor.execute("""
                            UPDATE employees 
                            SET status = 'active' 
                            WHERE name = %s
                        """, (employee_name,))
                        updated_count += 1
                        print(f"[AUTO-SYNC] Reactivated employee '{employee_name}' in database")
                
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
                        print(f"[AUTO-SYNC] Removed employee '{db_employee_name}' from database")
            
            connection.commit()
            
            return {
                "added": added_count,
                "updated": updated_count,
                "removed": removed_count
            }
            
        except Exception as e:
            print(f"[AUTO-SYNC] Full sync error: {e}")
            return {"added": 0, "updated": 0, "removed": 0}
        finally:
            if 'connection' in locals():
                connection.close()
    
    def _check_and_sync(self):
        """Check for changes and sync if needed"""
        try:
            # Check if files or employees have changed
            if self._files_changed() or self._employees_changed():
                print("[AUTO-SYNC] Changes detected, syncing...")
                
                # Update tracking
                self._update_file_timestamps()
                self._update_employee_set()
                
                # Get current employees from face recognition system
                face_employees = self._get_face_recognition_employees()
                
                # Perform full sync
                result = self._full_sync_employees(face_employees)
                print(f"[AUTO-SYNC] Sync completed - Added: {result['added']}, Updated: {result['updated']}, Removed: {result['removed']}")
                
        except Exception as e:
            print(f"[AUTO-SYNC] Error during sync check: {e}")
    
    def start(self):
        """Start the auto-sync service"""
        if self.running:
            print("[AUTO-SYNC] Service already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("[AUTO-SYNC] Service started")
    
    def stop(self):
        """Stop the auto-sync service"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("[AUTO-SYNC] Service stopped")
    
    def _run(self):
        """Main service loop"""
        while self.running:
            try:
                self._check_and_sync()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"[AUTO-SYNC] Error in main loop: {e}")
                time.sleep(self.check_interval)
    
    def force_sync(self):
        """Force an immediate sync"""
        try:
            face_employees = self._get_face_recognition_employees()
            result = self._full_sync_employees(face_employees)
            print(f"[AUTO-SYNC] Forced sync completed - Added: {result['added']}, Updated: {result['updated']}, Removed: {result['removed']}")
            return result
        except Exception as e:
            print(f"[AUTO-SYNC] Error during forced sync: {e}")
            return {"added": 0, "updated": 0, "removed": 0}

# Global instance
auto_sync_service = AutoSyncService()

def start_auto_sync():
    """Start the auto-sync service"""
    auto_sync_service.start()

def stop_auto_sync():
    """Stop the auto-sync service"""
    auto_sync_service.stop()

def force_sync():
    """Force an immediate sync"""
    return auto_sync_service.force_sync()

if __name__ == "__main__":
    # For testing the service directly
    try:
        start_auto_sync()
        print("Auto-sync service is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_auto_sync()