import cv2
import cv2.aruco as aruco
import numpy as np
import urllib.request
from ultralytics import YOLO
import torch
import time

# --- GENERAL CONFIGURATION ---
# If your esp32 camera is running on a different IP address, change it here
url = 'http://192.168.0.105/cam-hi.jpg'

conf_threshold = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- WORKSPACE CONFIGURATION ---
# ArUco marker IDs for the 4 corners:
# 10 -> Top Left, 20 -> Top Right, 30 -> Bottom Right, 40 -> Bottom Left
ARUCO_IDS = {10: 0, 20: 1, 40: 2, 30: 3}

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DETECTOR = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

# --- MODEL INITIALIZATION ---
print(f"Using device: {device}")

try:
    model = YOLO('yolov8x.pt')
    class_names = model.names
    print("YOLO model loaded. Detectable classes:", class_names)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

def detect_workspace(frame):
    """
    Detects the 4 ArUco markers that define the workspace.
    Returns an array with the 4 specific corner coordinates if all are found,
    otherwise returns None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    
    if ids is not None and len(ids) >= 4:
        workspace_corners = np.zeros((4, 2), dtype=np.float32)
        found_markers = 0
        
        for i, marker_id in enumerate(ids):
            if marker_id[0] in ARUCO_IDS:
                marker_index = ARUCO_IDS[marker_id[0]]
                corner_points = corners[i][0]
                
                if marker_id[0] == 10:  # Top left
                    corner_idx = np.argmin(corner_points[:, 0] + corner_points[:, 1])
                elif marker_id[0] == 20:  # Top right
                    corner_idx = np.argmax(corner_points[:, 0] - corner_points[:, 1])
                elif marker_id[0] == 40:  # Bottom right
                    corner_idx = np.argmax(corner_points[:, 0] + corner_points[:, 1])
                elif marker_id[0] == 30:  # Bottom left
                    corner_idx = np.argmin(corner_points[:, 0] - corner_points[:, 1])
                
                workspace_corners[marker_index] = corner_points[corner_idx]
                found_markers += 1
        
        if found_markers == 4:
            aruco.drawDetectedMarkers(frame, corners, ids)
            return workspace_corners
            
    return None

def main():
    """
    Main program function
    """
    print("Starting combined detection system (ArUco + YOLO)...")
    print("Press 'q' to exit")

    try:
        while True:
            time.sleep(1)
            
            # --- CAMERA CAPTURE ---
            try:
                img_resp = urllib.request.urlopen(url, timeout=5)
                img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_array, -1)
                if frame is None:
                    print("Could not get image from camera")
                    continue
            except Exception as e:
                print(f"Error getting image: {e}")
                continue
            
            # --- WORKSPACE DETECTION ---
            workspace_corners = detect_workspace(frame)

            # --- OBJECT DETECTION AND FILTERING ---
            if workspace_corners is not None:
                print(workspace_corners)
                cv2.polylines(frame, [workspace_corners.astype(int)], True, (255, 0, 0), 2)

                results = model(frame, conf=conf_threshold, device=device)
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        print((center_x, center_y))
                        
                        if cv2.pointPolygonTest(workspace_corners, (center_x, center_y), False) > 0:
                            class_name = class_names.get(class_id, f"Class_{class_id}")
                            label = f'{class_name}: {confidence:.2f}'
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            coord_label = f"({center_x}, {center_y})"
                            cv2.putText(frame, coord_label, (center_x + 15, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- DISPLAY PREPARATION ---
            screen_width = 640
            screen_height = 640
            
            height, width = frame.shape[:2]
            scale_x = screen_width / width
            scale_y = screen_height / height
            scale = min(scale_x, scale_y, 1.0)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow('Camera View', resized_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        print("Program finished")

if __name__ == "__main__":
    main()