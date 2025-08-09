from keras.models import load_model
import cv2
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def test_camera():
    """Test camera access and return available camera indices"""
    available_cameras = []
    for i in range(5):  # Test first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def run_classification(camera_index=0):
    """Run real-time classification with specified camera"""
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return False
    
    print(f"Successfully opened camera {camera_index}")
    print("Press ESC to quit, SPACE to save current frame")
    
    try:
        while True:
            ret, image = camera.read()
            if not ret or image is None:
                print("Failed to grab frame")
                break
            
            # Resize the image
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Show the image
            cv2.imshow("Plant Disease Classification", image_resized)
            
            # Prepare image for prediction
            image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1
            
            # Predict
            prediction = model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]
            
            # Display results
            print(f"\rClass: {class_name} | Confidence: {np.round(confidence_score * 100, 1)}%", end="")
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 32:  # SPACE key
                # Save current frame
                filename = f"capture_{class_name.replace(' ', '_')}_{int(confidence_score*100)}.jpg"
                cv2.imwrite(filename, image)
                print(f"\nSaved frame as {filename}")
                
    except Exception as e:
        print(f"Error during classification: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    print("Testing camera access...")
    available_cameras = test_camera()
    
    if not available_cameras:
        print("No cameras found! Please check:")
        print("1. Camera is connected and not in use by another application")
        print("2. Camera permissions are granted")
        print("3. Try restarting your computer")
    else:
        print(f"Available cameras: {available_cameras}")
        print("Starting classification...")
        
        # Try the first available camera
        success = run_classification(available_cameras[0])
        
        if not success:
            print("Failed to start camera. Trying alternative methods...")
            # Try different camera backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
                camera = cv2.VideoCapture(0, backend)
                if camera.isOpened():
                    print(f"Using backend: {backend}")
                    camera.release()
                    run_classification(0)
                    break
