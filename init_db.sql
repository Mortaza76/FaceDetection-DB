-- Initialize Face Recognition Attendance Database
-- This script creates all necessary tables for the Face Detection System

USE face_recognition_attendance;

-- Create employees table
CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    status ENUM('active', 'inactive') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_status (status)
);

-- Create employee_zone_events table for zone-based tracking
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
);

-- Create daily attendance summary table
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
);

-- Create cafeteria_overstay_alerts table
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
);

-- Insert sample data for testing (optional)
INSERT IGNORE INTO employees (name, status) VALUES 
('John Doe', 'active'),
('Jane Smith', 'active'),
('Bob Johnson', 'active');

-- Create indexes for better performance
CREATE INDEX idx_events_employee_date ON employee_zone_events (employee_name, DATE(timestamp));
CREATE INDEX idx_events_zone_date ON employee_zone_events (zone_name, DATE(timestamp));
CREATE INDEX idx_summary_date_range ON daily_attendance_summary (date, employee_id);

COMMIT;