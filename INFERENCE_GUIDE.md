# Face Anti-Spoofing Inference Guide

## Overview
This guide explains how to use the `inference.py` script for real-time face anti-spoofing detection using your trained model.

## Prerequisites

### Required Dependencies
Make sure you have the following installed:
```bash
pip install opencv-python torch torchvision pillow numpy
```

### Trained Model
You need a trained model file (`.pth`) from running `GD-FAS.py`. The model file is typically saved in:
```
results/<log_name>/<protocol>_best.pth
```

## Usage

### Basic Usage
```bash
python inference.py --model_path results/test/O_C_I_to_M_best.pth
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | **Required** | Path to trained model file (.pth) |
| `--backbone` | str | `clip` | Model architecture: `clip` or `resnet18` |
| `--threshold` | float | `0.5` | Classification threshold (0.0-1.0) |
| `--camera_id` | int | `0` | Camera device ID (0 for default camera) |
| `--display_size` | int | `640` | Display window width in pixels |

### Examples

#### Using CLIP backbone (default)
```bash
python inference.py --model_path results/my_model/O_C_I_to_M_best.pth
```

#### Using ResNet18 backbone
```bash
python inference.py --model_path results/resnet_model/model_best.pth --backbone resnet18
```

#### Adjust threshold for stricter detection
```bash
python inference.py --model_path results/test/O_C_I_to_M_best.pth --threshold 0.7
```

#### Use external webcam
```bash
python inference.py --model_path results/test/O_C_I_to_M_best.pth --camera_id 1
```

#### Larger display window
```bash
python inference.py --model_path results/test/O_C_I_to_M_best.pth --display_size 1024
```

## Controls

While the inference is running:
- **`q` or `ESC`** - Quit the application
- **`s`** - Save the current frame as `capture_N.jpg`

## Output Display

The video feed shows:
1. **Status Label**: "REAL FACE" (green) or "SPOOF DETECTED" (red)
2. **Confidence Score**: Confidence percentage of the prediction
3. **Probabilities**: Individual probabilities for both classes
4. **Colored Border**: Green for real, red for spoof

## Understanding Predictions

### Threshold
- Default threshold is **0.5**
- If `P(real) > threshold`, prediction is "REAL"
- If `P(real) ≤ threshold`, prediction is "SPOOF"

### Adjusting Threshold
- **Lower threshold (e.g., 0.3)**: More sensitive to real faces (fewer false rejections, more false acceptances)
- **Higher threshold (e.g., 0.7)**: Stricter spoof detection (fewer false acceptances, more false rejections)

## Troubleshooting

### Camera Issues

**Problem**: "Could not open camera"
```bash
# Try different camera IDs
python inference.py --model_path MODEL.pth --camera_id 0
python inference.py --model_path MODEL.pth --camera_id 1
python inference.py --model_path MODEL.pth --camera_id 2
```

**Problem**: Camera permission denied
- Check Windows camera privacy settings
- Ensure no other application is using the camera

### Model Loading Issues

**Problem**: "Model file not found"
- Verify the model path exists
- Use absolute path if needed:
```bash
python inference.py --model_path "e:\Capstone project\GD-FAS\results\test\O_C_I_to_M_best.pth"
```

**Problem**: "Error loading model"
- Ensure model was trained with compatible PyTorch version
- Check that the correct backbone is specified

### Performance Issues

**Problem**: Slow inference
- CPU inference is slower than GPU
- Close other applications to free resources
- Reduce display size:
```bash
python inference.py --model_path MODEL.pth --display_size 480
```

## Technical Details

### Image Preprocessing
- Input frames are resized to **224x224** pixels
- Normalized using ImageNet statistics:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

### Model Output
- Output: 2-class logits `[spoof_logit, real_logit]`
- Softmax applied to get probabilities
- Class 0: Spoof/Attack
- Class 1: Real/Live

### CPU Optimization
- Model runs on CPU with `torch.no_grad()`
- Automatic device mapping during model loading
- Efficient batch size of 1 for real-time processing

## Advanced Usage

### Batch Processing Video Files
For processing video files instead of webcam, modify the script or use this approach:
```python
# Replace cv2.VideoCapture(0) with:
cap = cv2.VideoCapture('path/to/video.mp4')
```

### Integration with Face Detection
For better results, integrate with face detection:
```python
import cv2

# Add face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# In processing loop:
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Process each detected face
for (x, y, w, h) in faces:
    face_roi = frame[y:y+h, x:x+w]
    # Run inference on face_roi
```

## Performance Metrics

Expected performance (on typical laptop CPU):
- **Inference time**: 50-200ms per frame
- **Frame rate**: 5-20 FPS
- **Memory usage**: ~500MB-2GB depending on model

## Safety Features

The script includes:
- ✅ Comprehensive error handling
- ✅ Graceful camera initialization
- ✅ Safe model loading with CPU mapping
- ✅ Keyboard interrupt handling
- ✅ Resource cleanup on exit
- ✅ Input validation

## Citation

If you use this implementation, please cite the original paper:
```
[Add paper citation here]
```

## Support

For issues or questions:
1. Check that all dependencies are installed
2. Verify model file exists and is accessible
3. Test with default parameters first
4. Check camera permissions and availability
