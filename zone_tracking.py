# zone_tracking.py - Zone-based attendance tracking with MySQL integration
import pymysql
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
import ssl
from email.message import EmailMessage

# Database configuration from environment variables
# Updated to match the environment variables in .env files
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'database': os.getenv('MYSQL_DATABASE', 'face_recognition_attendance'),
    'user': os.getenv('MYSQL_USER', 'root'),  # Changed from 'app_user' to match default
    'password': os.getenv('MYSQL_PASSWORD', 'Mortaza@348'),  # Changed from 'app_password' to match default
    'charset': 'utf8mb4',
    'autocommit': True,
    'connect_timeout': 10,
    'read_timeout': 10,
    'write_timeout': 10
}

# Zone mapping configuration
ZONE_CONFIG = {
    'office_entry': {'zone': 'office', 'action': 'entry'},
    'office_exit': {'zone': 'office', 'action': 'exit'},
    'cafeteria_entry': {'zone': 'cafeteria', 'action': 'entry'},
    'cafeteria_exit': {'zone': 'cafeteria', 'action': 'exit'}
}

# Anti-duplicate configuration (in seconds)
COOLDOWN_PERIOD = 5  # 5-second cooldown between detections on same camera
CAFETERIA_OVERSTAY_THRESHOLD = 30 * 60  # 30 minutes in seconds

# Email configuration
NOTIFY_EMAIL_SENDER = os.getenv("EMAIL_SENDER")
NOTIFY_EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
NOTIFY_EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Global tracking variables
employee_last_seen = {}  # {(employee_id, camera_name): timestamp}
cafeteria_entries = {}   # {employee_id: entry_timestamp}
overstay_alerts_sent = set()  # Set of employee_ids who already received overstay alert today

class ZoneTracker:
    """Zone-based attendance tracking system"""
    
    def __init__(self):
        self.db_config = DB_CONFIG
        self.init_database()
        self.start_overstay_monitor()
    
    def get_connection(self):
        """Get database connection"""
        try:
            connection = pymysql.connect(**self.db_config)
            return connection
        except Exception as e:
            print(f"[DB ERROR] Failed to connect to database: {e}")
            return None
    
    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        if not conn:
            print("[ERROR] Cannot initialize database - no connection")
            return
        
        try:
            cursor = conn.cursor()
            
            # Create employees table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL UNIQUE,
                    status ENUM('active', 'inactive') DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_name (name),
                    INDEX idx_status (status)
                )
            """)
            
            # Create employee_zone_events table for zone-based tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employee_zone_events (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id INT,
                    employee_name VARCHAR(100) NOT NULL,
                    camera_name VARCHAR(50) NOT NULL,
                    zone_name VARCHAR(50) NOT NULL,
                    action_type ENUM('entry', 'exit') NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence FLOAT DEFAULT 0.0,
                    snapshot_path VARCHAR(255),
                    INDEX idx_employee (employee_id),
                    INDEX idx_name (employee_name),
                    INDEX idx_camera (camera_name),
                    INDEX idx_zone (zone_name),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_zone_action (zone_name, action_type),
                    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE SET NULL
                )
            """)
            
            # Create daily attendance summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_attendance_summary (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id INT,
                    employee_name VARCHAR(100) NOT NULL,
                    date DATE NOT NULL,
                    first_office_entry TIME,
                    last_office_exit TIME,
                    total_office_minutes INT DEFAULT 0,
                    total_cafeteria_minutes INT DEFAULT 0,
                    working_hours_minutes INT DEFAULT 0,
                    cafeteria_visits_count INT DEFAULT 0,
                    longest_work_session_minutes INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_employee_date (employee_id, date),
                    INDEX idx_employee (employee_id),
                    INDEX idx_date (date),
                    INDEX idx_working_hours (working_hours_minutes),
                    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
                )
            """)
            
            # Create cafeteria_overstay_alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cafeteria_overstay_alerts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id INT,
                    employee_name VARCHAR(100) NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    alert_sent_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overstay_duration_minutes INT NOT NULL,
                    INDEX idx_employee (employee_id),
                    INDEX idx_entry_time (entry_time),
                    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
                )
            """)
            
            print("[DB] Database schema initialized successfully")
            
        except Exception as e:
            print(f"[DB ERROR] Failed to initialize database schema: {e}")
        finally:
            conn.close()
    
    def add_employee_if_not_exists(self, employee_name):
        """Add employee to database if they don't exist"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT IGNORE INTO employees (name, status) 
                VALUES (%s, 'active')
            """, (employee_name,))
            
            # Get employee ID
            cursor.execute("SELECT id FROM employees WHERE name = %s", (employee_name,))
            result = cursor.fetchone()
            employee_id = result[0] if result else None
            
            return employee_id
            
        except Exception as e:
            print(f"[DB ERROR] Failed to add employee: {e}")
            return None
        finally:
            conn.close()
    
    def should_process_detection(self, employee_name, camera_name):
        """Check if detection should be processed (anti-duplicate logic)"""
        key = (employee_name, camera_name)
        current_time = time.time()
        
        if key in employee_last_seen:
            time_since_last = current_time - employee_last_seen[key]
            if time_since_last < COOLDOWN_PERIOD:
                return False
        
        employee_last_seen[key] = current_time
        return True
    
    def log_zone_event(self, employee_name, camera_name, confidence, snapshot_path=None):
        """Log a zone event to the database"""
        if not self.should_process_detection(employee_name, camera_name):
            print(f"[ZONE] Skipping duplicate detection: {employee_name} on {camera_name}")
            return False
        
        # Get zone and action from camera name
        if camera_name not in ZONE_CONFIG:
            print(f"[ZONE ERROR] Unknown camera: {camera_name}")
            return False
        
        zone_info = ZONE_CONFIG[camera_name]
        zone_name = zone_info['zone']
        action_type = zone_info['action']
        
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            # Ensure employee exists
            employee_id = self.add_employee_if_not_exists(employee_name)
            if not employee_id:
                print(f"[ZONE ERROR] Could not create/find employee: {employee_name}")
                return False
            
            cursor = conn.cursor()
            
            # Insert zone event
            cursor.execute("""
                INSERT INTO employee_zone_events 
                (employee_id, employee_name, camera_name, zone_name, action_type, confidence, snapshot_path) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (employee_id, employee_name, camera_name, zone_name, action_type, confidence, snapshot_path))
            
            print(f"[ZONE] Logged: {employee_name} {action_type} {zone_name} via {camera_name} (conf: {confidence:.2f})")
            
            # Handle cafeteria tracking for overstay monitoring
            if zone_name == 'cafeteria':
                if action_type == 'entry':
                    cafeteria_entries[employee_id] = time.time()
                    # Reset overstay alert for new entry
                    today = datetime.now().date()
                    overstay_key = f"{employee_id}_{today}"
                    overstay_alerts_sent.discard(overstay_key)
                elif action_type == 'exit':
                    cafeteria_entries.pop(employee_id, None)
                    # Reset overstay alert when they exit
                    today = datetime.now().date()
                    overstay_key = f"{employee_id}_{today}"
                    overstay_alerts_sent.discard(overstay_key)
            
            return True
            
        except Exception as e:
            print(f"[ZONE ERROR] Failed to log zone event: {e}")
            return False
        finally:
            conn.close()
    
    def calculate_daily_attendance_summary(self, employee_name, date=None):
        """Calculate comprehensive daily attendance summary for an employee"""
        if date is None:
            date = datetime.now().date()
        
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            
            # Get all events for the day
            cursor.execute("""
                SELECT zone_name, action_type, timestamp
                FROM employee_zone_events 
                WHERE employee_name = %s AND DATE(timestamp) = %s
                ORDER BY timestamp
            """, (employee_name, date))
            
            events = cursor.fetchall()
            
            if not events:
                return {
                    'employee': employee_name,
                    'date': str(date),
                    'first_office_entry': None,
                    'last_office_exit': None,
                    'total_office_minutes': 0,
                    'total_cafeteria_minutes': 0,
                    'cafeteria_visits_count': 0,
                    'longest_work_session_minutes': 0,
                    'office_sessions': [],
                    'cafeteria_sessions': []
                }
            
            # Separate office and cafeteria events
            office_events = [(e[1], e[2]) for e in events if e[0] == 'office']  # (action, timestamp)
            cafeteria_events = [(e[1], e[2]) for e in events if e[0] == 'cafeteria']
            
            # Calculate office sessions
            office_sessions = []
            current_entry = None
            total_office_minutes = 0
            longest_session = 0
            
            for action, timestamp in office_events:
                if action == 'entry':
                    current_entry = timestamp
                elif action == 'exit' and current_entry:
                    duration_minutes = int((timestamp - current_entry).total_seconds() // 60)
                    office_sessions.append({
                        'entry': current_entry,
                        'exit': timestamp,
                        'duration_minutes': duration_minutes
                    })
                    total_office_minutes += duration_minutes
                    longest_session = max(longest_session, duration_minutes)
                    current_entry = None
            
            # Handle case where employee is still in office (no exit recorded)
            if current_entry:
                # Calculate duration until end of day or current time
                end_time = min(datetime.now(), datetime.combine(date, datetime.max.time()))
                if current_entry.date() == date:
                    duration_minutes = int((end_time - current_entry).total_seconds() // 60)
                    office_sessions.append({
                        'entry': current_entry,
                        'exit': None,  # Still in office
                        'duration_minutes': duration_minutes
                    })
                    total_office_minutes += duration_minutes
                    longest_session = max(longest_session, duration_minutes)
            
            # Calculate cafeteria sessions
            cafeteria_sessions = []
            current_caf_entry = None
            total_cafeteria_minutes = 0
            cafeteria_visits = 0
            
            for action, timestamp in cafeteria_events:
                if action == 'entry':
                    current_caf_entry = timestamp
                    cafeteria_visits += 1
                elif action == 'exit' and current_caf_entry:
                    duration_minutes = int((timestamp - current_caf_entry).total_seconds() // 60)
                    cafeteria_sessions.append({
                        'entry': current_caf_entry,
                        'exit': timestamp,
                        'duration_minutes': duration_minutes
                    })
                    total_cafeteria_minutes += duration_minutes
                    current_caf_entry = None
            
            # Handle case where employee is still in cafeteria
            if current_caf_entry:
                end_time = min(datetime.now(), datetime.combine(date, datetime.max.time()))
                if current_caf_entry.date() == date:
                    duration_minutes = int((end_time - current_caf_entry).total_seconds() // 60)
                    cafeteria_sessions.append({
                        'entry': current_caf_entry,
                        'exit': None,  # Still in cafeteria
                        'duration_minutes': duration_minutes
                    })
                    total_cafeteria_minutes += duration_minutes
            
            # Get first office entry and last office exit
            first_office_entry = office_sessions[0]['entry'] if office_sessions else None
            last_office_exit = None
            for session in reversed(office_sessions):
                if session['exit'] is not None:
                    last_office_exit = session['exit']
                    break
            
            # Update or insert daily summary
            employee_id = self.add_employee_if_not_exists(employee_name)
            if employee_id:
                working_hours_minutes = max(0, total_office_minutes - total_cafeteria_minutes)
                
                cursor.execute("""
                    INSERT INTO daily_attendance_summary 
                    (employee_id, employee_name, date, first_office_entry, last_office_exit, 
                     total_office_minutes, total_cafeteria_minutes, working_hours_minutes,
                     cafeteria_visits_count, longest_work_session_minutes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        first_office_entry = VALUES(first_office_entry),
                        last_office_exit = VALUES(last_office_exit),
                        total_office_minutes = VALUES(total_office_minutes),
                        total_cafeteria_minutes = VALUES(total_cafeteria_minutes),
                        working_hours_minutes = VALUES(working_hours_minutes),
                        cafeteria_visits_count = VALUES(cafeteria_visits_count),
                        longest_work_session_minutes = VALUES(longest_work_session_minutes),
                        updated_at = CURRENT_TIMESTAMP
                """, (employee_id, employee_name, date, 
                      first_office_entry.time() if first_office_entry else None,
                      last_office_exit.time() if last_office_exit else None,
                      total_office_minutes, total_cafeteria_minutes, working_hours_minutes,
                      cafeteria_visits, longest_session))
            
            return {
                'employee': employee_name,
                'date': str(date),
                'first_office_entry': first_office_entry,
                'last_office_exit': last_office_exit,
                'total_office_minutes': total_office_minutes,
                'total_cafeteria_minutes': total_cafeteria_minutes,
                'working_hours_minutes': max(0, total_office_minutes - total_cafeteria_minutes),  # Actual working time
                'cafeteria_visits_count': cafeteria_visits,
                'longest_work_session_minutes': longest_session,
                'office_sessions': office_sessions,
                'cafeteria_sessions': cafeteria_sessions,
                'metrics': {
                    'total_office_hours': round(total_office_minutes / 60, 2),
                    'total_cafeteria_hours': round(total_cafeteria_minutes / 60, 2),
                    'actual_working_hours': round(max(0, total_office_minutes - total_cafeteria_minutes) / 60, 2),
                    'break_percentage': round((total_cafeteria_minutes / total_office_minutes * 100) if total_office_minutes > 0 else 0, 1)
                }
            }
            
        except Exception as e:
            print(f"[CALC ERROR] Failed to calculate attendance summary: {e}")
            return None
        finally:
            conn.close()
    
    def get_weekly_summary(self, employee_name, start_date=None):
        """Get weekly attendance summary for an employee"""
        if start_date is None:
            # Get Monday of current week
            today = datetime.now().date()
            start_date = today - timedelta(days=today.weekday())
        
        weekly_data = []
        total_office_hours = 0
        total_working_hours = 0
        total_cafeteria_visits = 0
        
        for i in range(7):
            day = start_date + timedelta(days=i)
            daily_summary = self.calculate_daily_attendance_summary(employee_name, day)
            
            if daily_summary:
                weekly_data.append(daily_summary)
                total_office_hours += daily_summary['total_office_minutes'] / 60
                total_working_hours += daily_summary.get('working_hours_minutes', 0) / 60
                total_cafeteria_visits += daily_summary['cafeteria_visits_count']
        
        return {
            'employee': employee_name,
            'week_start': str(start_date),
            'week_end': str(start_date + timedelta(days=6)),
            'daily_summaries': weekly_data,
            'weekly_totals': {
                'total_office_hours': round(total_office_hours, 2),
                'total_working_hours': round(total_working_hours, 2),
                'total_break_hours': round(total_office_hours - total_working_hours, 2),
                'total_cafeteria_visits': total_cafeteria_visits,
                'average_office_hours_per_day': round(total_office_hours / 7, 2),
                'average_working_hours_per_day': round(total_working_hours / 7, 2),
                'weekly_productivity_percentage': round((total_working_hours / total_office_hours * 100) if total_office_hours > 0 else 0, 1)
            }
        }

    def get_daily_zone_report(self, employee_name, date=None):
        """Get daily zone activity report for an employee"""
        if date is None:
            date = datetime.now().date()
        
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            cursor.execute("""
                SELECT 
                    camera_name, zone_name, action_type, timestamp, confidence
                FROM employee_zone_events 
                WHERE employee_name = %s AND DATE(timestamp) = %s
                ORDER BY timestamp
            """, (employee_name, date))
            
            events = cursor.fetchall()
            
            # Calculate metrics
            office_entries = [e for e in events if e['zone_name'] == 'office' and e['action_type'] == 'entry']
            office_exits = [e for e in events if e['zone_name'] == 'office' and e['action_type'] == 'exit']
            cafeteria_entries = [e for e in events if e['zone_name'] == 'cafeteria' and e['action_type'] == 'entry']
            
            first_entry = office_entries[0]['timestamp'] if office_entries else None
            last_exit = office_exits[-1]['timestamp'] if office_exits else None
            
            return {
                'employee': employee_name,
                'date': str(date),
                'events': events,
                'first_office_entry': first_entry,
                'last_office_exit': last_exit,
                'office_entries_count': len(office_entries),
                'office_exits_count': len(office_exits),
                'cafeteria_visits_count': len(cafeteria_entries)
            }
            
        except Exception as e:
            print(f"[DB ERROR] Failed to get daily report: {e}")
            return None
        finally:
            conn.close()
    
    def send_overstay_alert(self, employee_name, entry_time, duration_minutes):
        """Send overstay alert email"""
        if not NOTIFY_EMAIL_SENDER or not NOTIFY_EMAIL_PASSWORD or not NOTIFY_EMAIL_RECEIVER:
            print("[EMAIL] Overstay alert skipped: email config not set")
            return
        
        try:
            msg = EmailMessage()
            msg["Subject"] = f"🚨 CAFETERIA OVERSTAY ALERT - {employee_name}"
            msg["From"] = NOTIFY_EMAIL_SENDER
            msg["To"] = NOTIFY_EMAIL_RECEIVER
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")
            
            msg.set_content(f"""
⚠️ CAFETERIA OVERSTAY ALERT ⚠️

Employee: {employee_name}
Entry Time: {entry_time_str}
Duration: {duration_minutes} minutes
Alert Time: {current_time}

The employee has been in the cafeteria for more than {CAFETERIA_OVERSTAY_THRESHOLD // 60} minutes 
and has not exited yet.

Please check on the employee's status.
            """)
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                smtp.login(NOTIFY_EMAIL_SENDER, NOTIFY_EMAIL_PASSWORD)
                smtp.send_message(msg)
            
            print(f"[EMAIL] Sent overstay alert for {employee_name} ({duration_minutes} minutes)")
            
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send overstay alert: {e}")
    
    def send_detection_alert(self, employee_name, camera_name, snapshot_path):
        """Send detection alert email for tracked employees"""
        if not NOTIFY_EMAIL_SENDER or not NOTIFY_EMAIL_PASSWORD or not NOTIFY_EMAIL_RECEIVER:
            print("[EMAIL] Detection alert skipped: email config not set")
            return
        
        try:
            msg = EmailMessage()
            msg["Subject"] = f"👀 Employee Detection - {employee_name} on {camera_name}"
            msg["From"] = NOTIFY_EMAIL_SENDER
            msg["To"] = NOTIFY_EMAIL_RECEIVER
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            zone_info = ZONE_CONFIG.get(camera_name, {'zone': 'unknown', 'action': 'unknown'})
            
            msg.set_content(f"""
Employee Detection Alert

Employee: {employee_name}
Camera: {camera_name}
Zone: {zone_info['zone']}
Action: {zone_info['action']}
Time: {timestamp}

Snapshot attached.
            """)
            
            # Attach snapshot if available
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(snapshot_path)
                msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
                smtp.login(NOTIFY_EMAIL_SENDER, NOTIFY_EMAIL_PASSWORD)
                smtp.send_message(msg)
            
            print(f"[EMAIL] Sent detection alert for {employee_name} on {camera_name}")
            
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send detection alert: {e}")
    
    def check_cafeteria_overstays(self):
        """Check for employees who have overstayed in cafeteria"""
        current_time = time.time()
        today = datetime.now().date()
        
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            for employee_id, entry_time in cafeteria_entries.items():
                duration = current_time - entry_time
                
                if duration >= CAFETERIA_OVERSTAY_THRESHOLD:
                    # Check if we already sent alert today
                    overstay_key = f"{employee_id}_{today}"
                    if overstay_key in overstay_alerts_sent:
                        continue
                    
                    # Get employee name
                    cursor.execute("SELECT name FROM employees WHERE id = %s", (employee_id,))
                    result = cursor.fetchone()
                    if not result:
                        continue
                    
                    employee_name = result[0]
                    duration_minutes = int(duration // 60)
                    entry_datetime = datetime.fromtimestamp(entry_time)
                    
                    # Send alert
                    self.send_overstay_alert(employee_name, entry_datetime, duration_minutes)
                    
                    # Log alert to database
                    cursor.execute("""
                        INSERT INTO cafeteria_overstay_alerts 
                        (employee_id, employee_name, entry_time, overstay_duration_minutes)
                        VALUES (%s, %s, %s, %s)
                    """, (employee_id, employee_name, entry_datetime, duration_minutes))
                    
                    # Mark alert as sent
                    overstay_alerts_sent.add(overstay_key)
                    
                    print(f"[OVERSTAY] Alert sent for {employee_name} ({duration_minutes} minutes)")
        
        except Exception as e:
            print(f"[OVERSTAY ERROR] Failed to check overstays: {e}")
        finally:
            conn.close()
    
    def start_overstay_monitor(self):
        """Start background thread to monitor cafeteria overstays"""
        def monitor_loop():
            while True:
                try:
                    self.check_cafeteria_overstays()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    print(f"[MONITOR ERROR] Overstay monitor error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("[MONITOR] Started cafeteria overstay monitoring")

# Global zone tracker instance
zone_tracker = ZoneTracker()

def log_employee_detection(employee_name, camera_name, confidence, snapshot_path=None, is_tracked=False):
    """Main function to log employee detection in zone-based system"""
    success = zone_tracker.log_zone_event(employee_name, camera_name, confidence, snapshot_path)
    
    # Send email alerts if needed
    if success and is_tracked:
        zone_tracker.send_detection_alert(employee_name, camera_name, snapshot_path)
    
    return success

def get_employee_zone_report(employee_name, date=None):
    """Get zone activity report for employee"""
    return zone_tracker.get_daily_zone_report(employee_name, date)
