@echo off
REM Docker Demo Script for Video Recording (Windows)
REM This script automates the Docker demo setup and execution

echo ==========================================
echo   Docker Demo Setup for Video Recording
echo ==========================================
echo.

REM Check if Docker is installed
echo [*] Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [X] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)
echo [OK] Docker is installed

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [X] Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)
echo [OK] Docker Compose is installed

REM Check for Groq API key
echo [*] Checking for Groq API key...
if "%GROQ_API_KEY%"=="" (
    echo [!] GROQ_API_KEY is not set!
    echo.
    echo Please set your Groq API key:
    echo   set GROQ_API_KEY=your-groq-api-key-here
    echo.
    echo Get your API key from: https://console.groq.com
    echo.
    set /p continue="Do you want to continue without API key? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
    echo [!] Continuing without API key (limited functionality)
) else (
    echo [OK] GROQ_API_KEY is set
)

REM Create output directory
echo [*] Creating output directory...
if not exist demo_outputs mkdir demo_outputs
echo [OK] Output directory created

REM Check if container is already running
echo [*] Checking for existing container...
docker ps -a | findstr streaming-avatar-api >nul 2>&1
if not errorlevel 1 (
    echo [!] Container already exists
    set /p remove="Do you want to remove it and start fresh? (y/n): "
    if /i "%remove%"=="y" (
        echo [*] Stopping and removing existing container...
        docker-compose down -v
        echo [OK] Container removed
    )
)

REM Build Docker image
echo [*] Building Docker image (this may take 5-10 minutes)...
echo.
echo [!] This is a good time to:
echo   - Prepare your screen recording software
echo   - Review the video script in DOCKER_VIDEO_GUIDE.md
echo   - Set up your terminal (font size, colors)
echo.

docker-compose build

if errorlevel 1 (
    echo [X] Docker build failed
    pause
    exit /b 1
)

echo [OK] Docker image built successfully

REM Start container
echo [*] Starting container...
docker-compose up -d

if errorlevel 1 (
    echo [X] Failed to start container
    pause
    exit /b 1
)

echo [OK] Container started

REM Wait for server to be ready
echo [*] Waiting for server to start (this may take 30-60 seconds)...
echo.

set max_attempts=30
set attempt=0

:wait_loop
if %attempt% geq %max_attempts% goto wait_failed

REM Try to connect to health endpoint
curl -s http://localhost:7860/health >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [OK] Server is ready!
    goto wait_success
)

echo|set /p=.
timeout /t 2 /nobreak >nul
set /a attempt+=1
goto wait_loop

:wait_failed
echo.
echo [X] Server failed to start. Check logs with: docker-compose logs
pause
exit /b 1

:wait_success

REM Verify server health
echo [*] Verifying server health...
curl -s http://localhost:7860/health
echo.

echo.
echo ==========================================
echo   [OK] READY TO RECORD!
echo ==========================================
echo.
echo Your Docker environment is ready for the demo video.
echo.
echo Next Steps:
echo.
echo 1. Start your screen recording software
echo 2. Open DOCKER_VIDEO_GUIDE.md for the script
echo 3. Run the demo:
echo.
echo    docker exec -it streaming-avatar-api python demo_full_pipeline.py
echo.
echo 4. Or run tests without Groq:
echo.
echo    docker exec -it streaming-avatar-api python test_without_groq.py
echo.
echo Output files will be in: .\demo_outputs\
echo.
echo View logs: docker-compose logs -f
echo Stop server: docker-compose down
echo.
echo Good luck with your recording!
echo.
pause
