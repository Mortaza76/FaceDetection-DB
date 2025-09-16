# Face Detection Attendance System

A comprehensive face detection and attendance tracking system with MySQL database integration.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Quick Deployment](#quick-deployment)
- [Deployment Options](#deployment-options)
- [Environment Configuration](#environment-configuration)
- [Database Setup](#database-setup)
- [Camera System Features](#camera-system-features)
- [API Access](#api-access)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## Overview

This system provides real-time face detection and recognition capabilities with integrated attendance tracking. It uses computer vision to identify employees and track their movements through designated zones to calculate working hours and monitor attendance.

The system was developed by Ameer Mortaza as a complete solution for employee attendance tracking using facial recognition technology.

## Key Features

- **Real-time Face Detection and Recognition**: Uses advanced computer vision to identify employees
- **Zone-based Attendance Tracking**: Monitors employee movements through office and cafeteria zones
- **Working Hours Calculation**: Automatically calculates productive working hours
- **Cafeteria Overstay Monitoring**: Alerts when employees stay too long in cafeteria
- **Multi-camera Support**: Supports multiple camera feeds simultaneously
- **Email Notifications**: Sends alerts for tracked employees and overstay events
- **Comprehensive Dashboard**: Detailed attendance statistics and reports
- **Database Integration**: Persistent storage of all attendance data
- **RESTful API**: Complete API for integration with other systems
- **Docker Containerization**: Easy deployment and scaling

## System Architecture

The system follows a microservices architecture with two main components:

1. **Face Detection Service** (`face-detection`): FastAPI application (Docker image: `mortaza77/database:amd64`)
2. **Database Service** (`mysql`): MySQL database for storing employee data and attendance records

Both services are containerized using Docker and orchestrated with Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- At least 2GB RAM available
- AMD64 architecture (system is optimized for this architecture)

## Quick Deployment

1. Clone or download the repository:
   ```bash
   git clone <repository-url>
   cd Clean
   ```

2. Start all services together:
   ```bash
   docker-compose up -d
   ```

3. Check service status:
   ```bash
   docker-compose ps
   ```

4. Access the application:
   - API: http://localhost:8001
   - API Documentation: http://localhost:8001/docs
   - Health Check: http://localhost:8001/health

## Deployment Options

### Recommended Approach: Docker Compose (Full System)

This deploys both the face detection service and MySQL database with proper networking:

```bash
# Deploy both services with proper configuration
docker-compose up -d

# Check logs
docker-compose logs -f face-detection
docker-compose logs -f mysql
```

### Manual Docker Deployment (Advanced)

If you need to deploy services manually, ensure both are running:

```bash
# Start MySQL database
docker run -d \
  --name mysql-db \
  -e MYSQL_ROOT_PASSWORD=Mortaza@348 \
  -e MYSQL_DATABASE=face_recognition_attendance \
  -p 3306:3306 \
  -v mysql_data:/var/lib/mysql \
  mysql:8.0

# Start face detection service (link to MySQL)
docker run -d \
  --name face-detection \
  --link mysql-db:mysql \
  -p 8001:8001 \
  -e MYSQL_HOST=mysql \
  -e MYSQL_PORT=3306 \
  -e MYSQL_DATABASE=face_recognition_attendance \
  -e MYSQL_USER=root \
  -e MYSQL_PASSWORD=Mortaza@348 \
  mortaza77/database:amd64
```

## Environment Configuration

The system requires environment variables for database connection. These are automatically set in docker-compose.yml:

| Variable | Description | Value |
|----------|-------------|-------|
| `MYSQL_HOST` | Database host | `mysql` |
| `MYSQL_PORT` | Database port | `3306` |
| `MYSQL_DATABASE` | Database name | `face_recognition_attendance` |
| `MYSQL_USER` | Database user | `root` |
| `MYSQL_PASSWORD` | Database password | `Mortaza@348` |

## Database Setup

The database is automatically initialized with the required schema when the MySQL container starts for the first time using the `init_db.sql` script.

### Key Tables:
1. `employees` - Employee information
2. `employee_zone_events` - Zone entry/exit tracking
3. `daily_attendance_summary` - Daily attendance calculations
4. `cafeteria_overstay_alerts` - Cafeteria monitoring

## Camera System Features

### Multi-Camera Support
The system supports multiple camera feeds simultaneously:
- Office Entry Camera
- Office Exit Camera
- Cafeteria Entry Camera
- Cafeteria Exit Camera

### Real-time Face Recognition Streaming
- Live video streaming with face detection overlays
- Confidence scoring for recognition accuracy
- Bounding boxes around detected faces
- Real-time employee identification

### Zone-based Tracking
- Automatic zone detection based on camera assignment
- Entry/exit tracking for office and cafeteria zones
- Anti-duplicate detection logic to prevent false positives
- Session-based tracking for accurate time calculations

### Attendance Calculation
- Working hours calculation (office time minus break time)
- Cafeteria visit tracking and duration monitoring
- Overstay detection in cafeteria (configurable threshold)
- Daily and weekly attendance summaries

### Notification System
- Email alerts for tracked employees
- Cafeteria overstay notifications
- Snapshot attachments for verification
- Configurable notification settings

### Webcam Processing
- Background processing threads for continuous monitoring
- Configurable camera sources (USB, IP cameras, etc.)
- Processing status monitoring and control
- Graceful start/stop mechanisms

## API Access

Once deployed, access the system at:
- Main API: http://localhost:8001
- Documentation: http://localhost:8001/docs
- Health Check: http://localhost:8001/health

### Employee Management
- `POST /employee/add` - Register new employee with face data
- `POST /employee/rename` - Rename existing employee
- `POST /employee/deactivate` - Deactivate employee
- `POST /employee/activate` - Reactivate employee
- `GET /employee/list` - List all employees
- `GET /employee/status` - Get employee status

### Camera Management
- `POST /camera/add` - Add new camera source
- `DELETE /camera/remove` - Remove camera source
- `GET /camera/list` - List all camera sources
- `POST /camera/process` - Start webcam processing
- `POST /camera/stop` - Stop webcam processing
- `GET /camera/status` - Get processing status

### Streaming Endpoints
- `GET /camera/stream/{name}` - Simple video stream
- `GET /camera/stream/{name}/recognition` - Stream with face recognition

### Attendance and Reporting
- `GET /dashboard/statistics` - Comprehensive dashboard statistics
- `GET /dashboard/summary` - Quick summary metrics
- `GET /zone/report/{employee_name}` - Detailed zone activity report
- `GET /attendance/calculate/{employee_name}` - Daily attendance calculation
- `GET /attendance/weekly/{employee_name}` - Weekly attendance summary

### System Management
- `GET /health` - System health status
- `POST /database/migrate` - Run database migrations
- `GET /database/verify` - Verify database connection
- `POST /employee/sync-to-database` - Sync employees to database

## Troubleshooting

### Database Connection Failed

**Most Common Issue**: Deploying only the face detection service without MySQL

**Solution**: Always deploy both services together using docker-compose:

```bash
# Check if both services are running
docker-compose ps

# If only face-detection is running, start the full system:
docker-compose up -d  # This starts both services with proper networking
```

### Service Status Check

```bash
# View all running containers
docker ps

# Check specific service logs
docker logs <container-name>

# Check face detection service logs for database errors
docker-compose logs face-detection | grep -i "database\|error\|mysql"
```

### Resource Issues

If services fail to start:
```bash
# Check available resources
docker info

# Restart services with more resources
docker-compose down
docker-compose up -d
```

## Project Structure

```
.
├── app.py                    # Main FastAPI application
├── zone_tracking.py          # Database integration and zone tracking
├── webcam_testing_backend.py # Face recognition processing
├── auto_sync_service.py      # Database synchronization
├── Dockerfile                # Application container definition
├── docker-compose.yml        # Service orchestration (IMPORTANT!)
├── init_db.sql               # Database schema initialization
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Important Notes for Deployment

1. **Always use docker-compose.yml**: This file properly configures both services with the correct networking and environment variables.

2. **Database Connection**: The face detection service connects to the MySQL service using the hostname `mysql` (defined in docker-compose.yml).

3. **Environment Variables**: All required database connection details are set in the docker-compose.yml file.

4. **Data Persistence**: Database data is stored in a Docker volume (`mysql_data`) to persist between restarts.

## Support

For deployment issues, ensure you're using the docker-compose approach as it properly handles:
- Service networking
- Environment variable configuration
- Data persistence
- Health checks

Contact the development team if you continue to experience issues.