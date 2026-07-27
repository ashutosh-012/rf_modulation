@echo off
echo ===================================================
echo     RF Modulation Classification Automation
echo ===================================================

echo.
echo [1/3] Preparing Data...
python scripts\prepare_data.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/3] Training Model (Basic CNN)...
python scripts\train.py model=cnn training.epochs=1
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/3] Exporting to ONNX and TensorRT...
python scripts\export_onnx.py model=cnn
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ===================================================
echo     Pipeline Completed Successfully!
echo ===================================================
pause
