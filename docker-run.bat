@echo off
REM Quick script to build and run the Docker container on Windows

echo ==================================
echo Building Streaming Avatar API
echo ==================================
echo.

REM Check if GROQ_API_KEY is set
if "%GROQ_API_KEY%"=="" (
    echo WARNING: GROQ_API_KEY not set
    echo    Some features will not work without it
    echo    Set it with: set GROQ_API_KEY=your-key
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Build the Docker image
echo Building Docker image...
docker build -t streaming-avatar-api .

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo Build successful!
echo.

REM Stop and remove existing container if it exists
echo Stopping existing container (if any)...
docker stop streaming-avatar-api 2>nul
docker rm streaming-avatar-api 2>nul

REM Run the container
echo Starting container...
docker run -d ^
    --name streaming-avatar-api ^
    -p 7860:7860 ^
    -e GROQ_API_KEY=%GROQ_API_KEY% ^
    -v "%cd%\generated_audio:/app/generated_audio" ^
    --restart unless-stopped ^
    streaming-avatar-api

if %ERRORLEVEL% neq 0 (
    echo Failed to start container!
    exit /b 1
)

echo.
echo ==================================
echo Container started successfully!
echo ==================================
echo.
echo API is running at: http://localhost:7860
echo.
echo Useful commands:
echo   View logs:    docker logs -f streaming-avatar-api
echo   Stop:         docker stop streaming-avatar-api
echo   Restart:      docker restart streaming-avatar-api
echo   Remove:       docker rm -f streaming-avatar-api
echo.
echo Test endpoints:
echo   Health:       curl http://localhost:7860/health
echo   Docs:         http://localhost:7860/docs
echo.

REM Wait and check if container is running
timeout /t 5 /nobreak >nul
docker ps | findstr streaming-avatar-api >nul
if %ERRORLEVEL% equ 0 (
    echo Container is running!
    echo.
    echo Checking health...
    timeout /t 10 /nobreak >nul
    curl -s http://localhost:7860/health
) else (
    echo Container stopped unexpectedly!
    echo Check logs with: docker logs streaming-avatar-api
)
