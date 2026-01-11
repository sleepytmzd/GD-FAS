"""
Real-time Face Anti-Spoofing Inference
Performs liveness detection on webcam feed using trained model
"""

import argparse
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Face Anti-Spoofing Inference')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model (.pth file)')
    parser.add_argument('--backbone', type=str, default='clip',
                        choices=['clip', 'resnet18'],
                        help='Model backbone architecture')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--display_size', type=int, default=640,
                        help='Display window size (default: 640)')
    return parser.parse_args()

def load_model(model_path, device='cpu'):
    """
    Load the trained model safely
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load model to CPU regardless of how it was trained
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.to(device)
        
        print("Model loaded successfully!")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def get_transform():
    """
    Define image transformations matching training pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def preprocess_frame(frame, transform):
    """
    Preprocess video frame for model input
    
    Args:
        frame: BGR image from OpenCV
        transform: torchvision transforms
    
    Returns:
        Preprocessed tensor ready for model
    """
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        tensor = transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def predict(model, frame_tensor, device, threshold=0.5):
    """
    Make prediction on preprocessed frame
    
    Args:
        model: Loaded model
        frame_tensor: Preprocessed frame tensor
        device: torch device
        threshold: Classification threshold
    
    Returns:
        is_real (bool): True if real face, False if spoof
        confidence (float): Confidence score [0-1]
        probabilities (tuple): (prob_spoof, prob_real)
    """
    try:
        with torch.no_grad():
            frame_tensor = frame_tensor.to(device)
            
            # Forward pass
            logits = model(frame_tensor)
            
            # Handle different output formats
            if isinstance(logits, tuple):
                # If model returns multiple outputs, use the last one
                logits = logits[-1]
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get probabilities for each class
            prob_spoof = probs[0, 0].item()
            prob_real = probs[0, 1].item()
            
            # Determine prediction
            is_real = prob_real > threshold
            confidence = prob_real if is_real else prob_spoof
            
            return is_real, confidence, (prob_spoof, prob_real)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0, (0.0, 0.0)

def draw_results(frame, is_real, confidence, prob_spoof, prob_real):
    """
    Draw prediction results on frame
    
    Args:
        frame: Original frame
        is_real: Boolean indicating if face is real
        confidence: Confidence score
        prob_spoof: Spoof probability
        prob_real: Real probability
    
    Returns:
        Frame with annotations
    """
    height, width = frame.shape[:2]
    
    # Determine color and label
    if is_real:
        color = (0, 255, 0)  # Green for real
        label = "REAL FACE"
    else:
        color = (0, 0, 255)  # Red for spoof
        label = "SPOOF DETECTED"
    
    # Draw background rectangle for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw main prediction label
    cv2.putText(frame, label, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Draw confidence
    conf_text = f"Confidence: {confidence:.2%}"
    cv2.putText(frame, conf_text, (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw probabilities
    prob_text = f"Real: {prob_real:.2%} | Spoof: {prob_spoof:.2%}"
    cv2.putText(frame, prob_text, (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw border
    border_thickness = 5
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, border_thickness)
    
    return frame

def main():
    args = parse_args()
    
    # Set device to CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Get transform
    transform = get_transform()
    
    # Initialize camera
    print(f"Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        print("Please check:")
        print("1. Camera is connected")
        print("2. Camera permissions are granted")
        print("3. Camera is not being used by another application")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Face Anti-Spoofing System - Real-time Detection")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {device}")
    print("\nControls:")
    print("  'q' or 'ESC' - Quit")
    print("  's' - Save current frame")
    print("="*60 + "\n")
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Resize frame for display
            display_frame = cv2.resize(frame, (args.display_size, 
                                              int(args.display_size * frame.shape[0] / frame.shape[1])))
            
            # Preprocess frame
            frame_tensor = preprocess_frame(frame, transform)
            
            if frame_tensor is not None:
                # Make prediction
                is_real, confidence, (prob_spoof, prob_real) = predict(
                    model, frame_tensor, device, args.threshold
                )
                
                # Draw results
                if is_real is not None:
                    display_frame = draw_results(
                        display_frame, is_real, confidence, prob_spoof, prob_real
                    )
            else:
                # Draw error message
                cv2.putText(display_frame, "Processing Error", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Face Anti-Spoofing Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nShutting down...")
                break
            elif key == ord('s'):  # Save frame
                saved_count += 1
                filename = f"capture_{saved_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_count}")
        print("Done!")

if __name__ == '__main__':
    main()
