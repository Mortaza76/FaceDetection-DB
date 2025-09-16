#!/usr/bin/env python3
"""
Database Migration Script for Working Hours Support
"""

import pymysql
import os

def migrate_database():
    """Run database migration to add working hours support"""
    # Standardized database configuration using environment variables
    db_config = {
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
    
    try:
        connection = pymysql.connect(**db_config)
        
        with connection.cursor() as cursor:
            # Check if daily_attendance_summary table exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = 'daily_attendance_summary'
            """, (db_config['database'],))
            
            result = cursor.fetchone()
            table_exists = result[0] > 0
            
            if not table_exists:
                # Create daily_attendance_summary table
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
                print("[MIGRATION] Created daily_attendance_summary table")
            
            # Check if cafeteria_overstay_alerts table exists
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = 'cafeteria_overstay_alerts'
            """, (db_config['database'],))
            
            result = cursor.fetchone()
            alerts_table_exists = result[0] > 0
            
            if not alerts_table_exists:
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
                print("[MIGRATION] Created cafeteria_overstay_alerts table")
        
        connection.commit()
        print("[MIGRATION] Database migration completed successfully")
        return True
        
    except Exception as e:
        print(f"[MIGRATION] Database migration failed: {e}")
        return False
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    migrate_database()