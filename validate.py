#!/usr/bin/env python3
"""
Validation Script for Face Detection System Deployment
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "app.py",
        "zone_tracking.py",
        "auto_sync_service.py",
        "webcam_testing_backend.py",
        "init_db.sql",
        "README.md",
        ".env.example",
        ".dockerignore",
        "start.sh",
        "stop.sh",
        "logs.sh"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files are present")
        return True

def check_permissions():
    """Check if scripts have executable permissions"""
    scripts = ["start.sh", "stop.sh", "logs.sh"]
    no_exec = []
    
    for script in scripts:
        if os.path.exists(script) and not os.access(script, os.X_OK):
            no_exec.append(script)
    
    if no_exec:
        print("❌ Scripts without executable permissions:")
        for script in no_exec:
            print(f"  - {script}")
        return False
    else:
        print("✅ All scripts have executable permissions")
        return True

def check_dockerfile():
    """Check Dockerfile for common issues"""
    if not os.path.exists("Dockerfile"):
        print("❌ Dockerfile not found")
        return False
    
    with open("Dockerfile", "r") as f:
        content = f.read()
    
    issues = []
    if "platform=linux/amd64" not in content:
        issues.append("Missing platform specification for AMD64")
    
    if "HEALTHCHECK" not in content:
        issues.append("Missing health check")
    
    if issues:
        print("❌ Dockerfile issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Dockerfile is properly configured")
        return True

def check_docker_compose():
    """Check docker-compose.yml for common issues"""
    if not os.path.exists("docker-compose.yml"):
        print("❌ docker-compose.yml not found")
        return False
    
    with open("docker-compose.yml", "r") as f:
        content = f.read()
    
    issues = []
    if "platform: linux/amd64" not in content:
        issues.append("Missing platform specification for AMD64 services")
    
    if "healthcheck" not in content:
        issues.append("Missing health checks")
    
    if issues:
        print("❌ docker-compose.yml issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ docker-compose.yml is properly configured")
        return True

def check_env_example():
    """Check .env.example for required variables"""
    if not os.path.exists(".env.example"):
        print("❌ .env.example not found")
        return False
    
    with open(".env.example", "r") as f:
        content = f.read()
    
    required_vars = [
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if var not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables in .env.example:")
        for var in missing_vars:
            print(f"  - {var}")
        return False
    else:
        print("✅ .env.example contains all required variables")
        return True

def main():
    """Main validation function"""
    print("Validating Face Detection System Deployment for AMD64...")
    print("=" * 60)
    
    checks = [
        check_files,
        check_permissions,
        check_dockerfile,
        check_docker_compose,
        check_env_example
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All validations passed! The deployment is ready for AMD64.")
        print("\nNext steps:")
        print("1. Ensure Docker is running")
        print("2. Run './start.sh' to start the services")
        print("3. Check logs with './logs.sh' if needed")
        return 0
    else:
        print("❌ Some validations failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())