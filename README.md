# Face Detection System - AMD64 Deployment

## Prerequisites
- Docker
- Docker Compose

## Deployment Instructions

1. Create a .env file from the example:
   ```bash
   cp .env.example .env
   ```
   Customize the values as needed.

2. Build and start the services:
   ```bash
   docker-compose up -d --build
   ```

3. Check the status of services:
   ```bash
   docker-compose ps
   ```

4. Access the API:
   - API: http://localhost:8001
   - Documentation: http://localhost:8001/docs
   - Health Check: http://localhost:8001/health

## Environment Variables
- MYSQL_HOST: Database host (default: mysql)
- MYSQL_PORT: Database port (default: 3306)
- MYSQL_DATABASE: Database name (default: face_recognition_attendance)
- MYSQL_USER: Database user (default: root)
- MYSQL_PASSWORD: Database password (default: Mortaza@348)

## Useful Commands

### View logs:
```bash
docker-compose logs -f face-detection
docker-compose logs -f mysql
```

### Stop services:
```bash
docker-compose down
```

### Backup database:
```bash
docker-compose exec mysql mysqldump -u root -pMortaza@348 face_recognition_attendance > backup.sql
```

### Restore database:
```bash
docker-compose exec -T mysql mysql -u root -pMortaza@348 face_recognition_attendance < backup.sql
```

## Notes
- All .pkl files are architecture-independent and will work on AMD64
- The system has been verified for AMD64 compatibility
- Database schema is compatible with MySQL on AMD64
- Health checks are implemented for both services
- Resource limits are set for better stability
- Automatic database synchronization is enabled