# Ref: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/
# Extended for dual camera setup with head tilt detection

import cv2
import time
import math as m
import mediapipe as mp
import argparse
import threading
import numpy as np
import signal
import sys
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

def findDistance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        Distance between the two points.
    """
    dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    """
    Calculate the angle between two points with respect to the y-axis.

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        Angle in degrees.
    """
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi) * theta
    return degree

def findHorizontalTiltAngle(x1, y1, x2, y2):
    """
    Calculate head tilt angle from horizontal axis.
    
    Args:
        x1, y1: Coordinates of left eye
        x2, y2: Coordinates of right eye
    
    Returns:
        Angle in degrees from horizontal
    """
    angle = m.atan2(y2 - y1, x2 - x1) * (180 / m.pi)
    return abs(angle)

def findCentroid(x1, y1, x2, y2):
    """
    Calculate the centroid of two points

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        Centroid coordiantes
    """
    x_c = (x1 + x2) / 2
    y_c = (y1 + y2) / 2
    return int(x_c), int(y_c)

def test_camera(index):
    """Test if a camera is available at the given index"""
    try:
        cap = cv2.VideoCapture(index)
    except Exception as e:
        return False
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        return ret
    return False

def list_available_cameras():
    """List all available camera indices"""
    available = []
    for i in range(10):  # Test first 10 indices
        if test_camera(i):
            available.append(i)
    return available

def parse_arguments():
    parser = argparse.ArgumentParser(description='Dual Camera Posture Monitor with MediaPipe')
    parser.add_argument('--video-side', type=int, default=0, help='Camera index for side view (default: 0)')
    parser.add_argument('--video-front', type=int, default=1, help='Camera index for front view (default: 1)')
    parser.add_argument('--offset-threshold', type=int, default=100, help='Threshold value for shoulder alignment.')
    parser.add_argument('--neck-angle-threshold', type=int, default=15, help='Threshold value for neck inclination angle.')
    parser.add_argument('--torso-angle-threshold', type=int, default=5, help='Threshold value for torso inclination angle.')
    parser.add_argument('--head-tilt-threshold', type=int, default=2, help='Threshold value for head tilt angle (degrees).')
    parser.add_argument('--shoulder-tilt-threshold', type=int, default=2, help='Threshold value for shoulder tilt angle (degrees).')
    parser.add_argument('--time-threshold', type=int, default=10, help='Time threshold for triggering a posture alert.')
    parser.add_argument('--single-camera', action='store_true', help='Use only side view camera (no head tilt detection)')
    return parser.parse_args()

class PostureMonitor:
    def __init__(self, offset_threshold, neck_angle_threshold, torso_angle_threshold, 
                 head_tilt_threshold, shoulder_tilt_threshold, time_threshold):
        self.offset_threshold = offset_threshold
        self.neck_angle_threshold = neck_angle_threshold
        self.torso_angle_threshold = torso_angle_threshold
        self.head_tilt_threshold = head_tilt_threshold
        self.shoulder_tilt_threshold = shoulder_tilt_threshold
        self.time_threshold = time_threshold
        
        # Shared state
        self.good_frames = 0
        self.bad_frames = 0
        self.head_tilt_bad = False
        self.posture_bad = False
        self.lock = threading.Lock()
        self.running = True
        self.fps = 30
        self.warning_triggered = False
        self.last_warning_time = 0
        
        # Colors
        self.blue = (255, 127, 0)
        self.red = (50, 50, 255)
        self.green = (127, 255, 0)
        self.light_green = (127, 233, 100)
        self.white = (255, 255, 255)
        self.orange = (0, 165, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def send_warning(self):
        """Send a warning alert to the user"""
        current_time = time.time()
        # Prevent spamming warnings (minimum 5 seconds between warnings)
        if current_time - self.last_warning_time < 5:
            return
        
        self.last_warning_time = current_time
        print("\n" + "="*60)
        print("⚠️  POSTURE ALERT: Bad posture detected!")
        print("="*60 + "\n")
                
        print("\a")  # Terminal bell

    def stop(self):
        """Stop all monitoring threads"""
        print("\nStopping posture monitor...")
        self.running = False

    def process_side_view(self, camera_index):
        """Process side view camera for posture detection"""
        print(f"Starting side view camera {camera_index}...")
        mp_pose_instance = mp.solutions.pose
        pose = mp_pose_instance.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open side view camera {camera_index}")
            self.running = False
            return
        
        # Try to read a frame to ensure camera is working
        ret, test_frame = cap.read()
        if not ret:
            print(f"ERROR: Could not read from side view camera {camera_index}")
            cap.release()
            self.running = False
            return
        
        print(f"Side view camera {camera_index} opened successfully")
        frame_count = 0
        
        try:
            while self.running:
                success, image = cap.read()
                if not success:
                    print(f"Side view: Failed to read frame {frame_count}")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or fps is None:
                    fps = 30
                
                with self.lock:
                    self.fps = fps
                
                h, w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                keypoints = pose.process(image_rgb)
                
                lm = keypoints.pose_landmarks
                if lm is None:
                    cv2.putText(image, 'No pose detected - Stand in side view', (10, 30), 
                               self.font, 0.6, self.red, 2)
                    cv2.putText(image, f'Frame: {frame_count}', (10, h - 10), 
                               self.font, 0.5, self.white, 1)
                    cv2.imshow('Side View - Posture', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                    continue
                
                lmPose = mp_pose_instance.PoseLandmark
                
                # Get landmarks
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                shldr_c_x, shldr_c_y = findCentroid(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
                
                r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
                r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)
                l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
                l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
                ear_c_x, ear_c_y = findCentroid(r_ear_x, r_ear_y, l_ear_x, l_ear_y)
                
                r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
                r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)
                l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
                hip_c_x, hip_c_y = findCentroid(r_hip_x, r_hip_y, l_hip_x, l_hip_y)
                
                # Calculate metrics
                offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
                neck_inclination = findAngle(shldr_c_x, shldr_c_y, ear_c_x, ear_c_y)
                torso_inclination = findAngle(hip_c_x, hip_c_y, shldr_c_x, shldr_c_y)
                
                # Draw landmarks
                cv2.circle(image, (r_shldr_x, r_shldr_y), 7, self.white, 2)
                cv2.circle(image, (l_shldr_x, l_shldr_y), 7, self.white, 2)
                cv2.circle(image, (r_ear_x, r_ear_y), 7, self.white, 2)
                cv2.circle(image, (l_ear_x, l_ear_y), 7, self.white, 2)
                cv2.circle(image, (r_hip_x, r_hip_y), 7, self.white, 2)
                cv2.circle(image, (l_hip_x, l_hip_y), 7, self.white, 2)
                cv2.circle(image, (ear_c_x, ear_c_y), 7, self.red, -1)
                cv2.circle(image, (shldr_c_x, shldr_c_y), 7, self.green, -1)
                cv2.circle(image, (hip_c_x, hip_c_y), 7, self.blue, -1)
                
                # Shoulder alignment
                if offset < self.offset_threshold:
                    cv2.putText(image, str(int(offset)) + ' Aligned', 
                               (w - 180, 30), self.font, 0.6, self.green, 2)
                else:
                    cv2.putText(image, str(int(offset)) + ' Not Aligned', 
                               (w - 180, 30), self.font, 0.6, self.red, 2)
                
                # Determine posture
                angle_text_neck = 'Neck: ' + str(int(neck_inclination))
                angle_text_torso = 'Torso: ' + str(int(torso_inclination))
                
                current_posture_good = (neck_inclination < self.neck_angle_threshold and 
                                       torso_inclination < self.torso_angle_threshold)
                
                with self.lock:
                    self.posture_bad = not current_posture_good
                    
                    if current_posture_good and not self.head_tilt_bad:
                        if self.bad_frames > 0:
                            self.bad_frames -= 1
                        self.good_frames += 1
                    else:
                        self.good_frames = 0
                        self.bad_frames += 1
                    
                    good_time = (1 / fps) * self.good_frames
                    bad_time = (1 / fps) * self.bad_frames
                
                color = self.light_green if current_posture_good else self.red
                cv2.putText(image, angle_text_neck, (10, 30), self.font, 0.6, color, 2)
                cv2.putText(image, angle_text_torso, (10, 60), self.font, 0.6, color, 2)
                cv2.putText(image, str(int(neck_inclination)), (shldr_c_x + 10, shldr_c_y), 
                           self.font, 0.9, color, 2)
                cv2.putText(image, str(int(torso_inclination)), (hip_c_x + 10, hip_c_y), 
                           self.font, 0.9, color, 2)
                
                # Draw lines
                cv2.line(image, (r_shldr_x, r_shldr_y), (shldr_c_x, shldr_c_y), color, 2)
                cv2.line(image, (l_shldr_x, l_shldr_y), (shldr_c_x, shldr_c_y), color, 2)
                cv2.line(image, (r_hip_x, r_hip_y), (hip_c_x, hip_c_y), color, 2)
                cv2.line(image, (hip_c_x, hip_c_y), (l_hip_x, l_hip_y), color, 2)
                cv2.line(image, (hip_c_x, hip_c_y), (shldr_c_x, shldr_c_y), color, 2)
                cv2.line(image, (ear_c_x, ear_c_y), (shldr_c_x, shldr_c_y), color, 2)
                
                # Display time
                if good_time > 0:
                    time_string = 'Good: ' + str(round(good_time, 1)) + 's'
                    cv2.putText(image, time_string, (10, h - 20), self.font, 0.9, self.green, 2)
                else:
                    time_string = 'Bad: ' + str(round(bad_time, 1)) + 's'
                    cv2.putText(image, time_string, (10, h - 20), self.font, 0.9, self.red, 2)
                    
                    # Visual warning on screen
                    if bad_time > self.time_threshold:
                        cv2.putText(image, '⚠️ POSTURE ALERT! ⚠️', (w//2 - 150, h//2), 
                                   self.font, 1.0, self.orange, 3)
                
                # Alert
                if bad_time > self.time_threshold:
                    with self.lock:
                            # Send warning in a separate thread to avoid blocking
                            self.send_warning()
                            self.bad_frames = 0

                
                cv2.imshow('Side View - Posture', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
        finally:
            print("Side view camera closing...")
            cap.release()
            cv2.destroyWindow('Side View - Posture')

    def process_front_view(self, camera_index):
        """Process front view camera for head tilt and shoulder alignment detection"""
        print(f"Starting front view camera {camera_index}...")
        mp_pose_instance = mp.solutions.pose
        pose = mp_pose_instance.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open front view camera {camera_index}")
            return
        
        # Try to read a frame
        ret, test_frame = cap.read()
        if not ret:
            print(f"ERROR: Could not read from front view camera {camera_index}")
            cap.release()
            return
        
        print(f"Front view camera {camera_index} opened successfully")
        
        frame_count = 0
        
        try:
            while self.running:
                success, image = cap.read()
                if not success:
                    print(f"Front view: Failed to read frame {frame_count}")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or fps is None:
                    with self.lock:
                        fps = self.fps
                
                h, w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                keypoints = pose.process(image_rgb)
                
                lm = keypoints.pose_landmarks
                if lm is None:
                    cv2.putText(image, 'No pose detected - Face camera', (10, 30), 
                               self.font, 0.6, self.red, 2)
                    cv2.putText(image, f'Frame: {frame_count}', (10, h - 10), 
                               self.font, 0.5, self.white, 1)
                    cv2.imshow('Front View - Head & Shoulders', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                    continue
                
                lmPose = mp_pose_instance.PoseLandmark
                
                # Get eye landmarks
                l_eye_x = int(lm.landmark[lmPose.LEFT_EYE].x * w)
                l_eye_y = int(lm.landmark[lmPose.LEFT_EYE].y * h)
                r_eye_x = int(lm.landmark[lmPose.RIGHT_EYE].x * w)
                r_eye_y = int(lm.landmark[lmPose.RIGHT_EYE].y * h)
                
                # Get shoulder landmarks
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                
                # Calculate head tilt (eye alignment)
                head_tilt = findHorizontalTiltAngle(r_eye_x, r_eye_y, l_eye_x, l_eye_y)
                
                # Calculate shoulder tilt angle
                shoulder_tilt = findHorizontalTiltAngle(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
                
                # Draw eye markers
                cv2.circle(image, (l_eye_x, l_eye_y), 5, self.white, -1)
                cv2.circle(image, (r_eye_x, r_eye_y), 5, self.white, -1)
                
                # Draw shoulder markers
                cv2.circle(image, (l_shldr_x, l_shldr_y), 7, self.white, 2)
                cv2.circle(image, (r_shldr_x, r_shldr_y), 7, self.white, 2)
                
                # Determine if alignment is good
                head_aligned = head_tilt < self.head_tilt_threshold
                shoulders_aligned = shoulder_tilt < self.shoulder_tilt_threshold
                both_aligned = head_aligned and shoulders_aligned
                
                # Draw lines
                eye_color = self.green if head_aligned else self.red
                shoulder_color = self.green if shoulders_aligned else self.red
                
                cv2.line(image, (l_eye_x, l_eye_y), (r_eye_x, r_eye_y), eye_color, 2)
                cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), shoulder_color, 3)
                
                # Update shared state
                with self.lock:
                    self.head_tilt_bad = not both_aligned
                    
                    if not self.head_tilt_bad and not self.posture_bad:
                        if self.bad_frames > 0:
                            self.bad_frames -= 1
                        self.good_frames += 1
                    else:
                        self.good_frames = 0
                        self.bad_frames += 1
                
                # Display info
                head_text = f'Head Tilt: {head_tilt:.1f}°'
                shoulder_text = f'Shoulder Tilt: {shoulder_tilt:.1f}°'
                
                head_color = self.light_green if head_aligned else self.red
                shoulder_color_text = self.light_green if shoulders_aligned else self.red
                
                cv2.putText(image, head_text, (10, 30), self.font, 0.6, head_color, 2)
                cv2.putText(image, shoulder_text, (10, 60), self.font, 0.6, shoulder_color_text, 2)
                
                # Overall status
                if both_aligned:
                    cv2.putText(image, 'Aligned', (10, 90), self.font, 0.7, self.light_green, 2)
                else:
                    if not head_aligned:
                        cv2.putText(image, 'Head Tilted!', (10, 90), self.font, 0.7, self.red, 2)
                    if not shoulders_aligned:
                        cv2.putText(image, 'Shoulders Uneven!', (10, 120), self.font, 0.7, self.red, 2)
                
                cv2.putText(image, f'Frame: {frame_count}', (10, h - 10), 
                           self.font, 0.5, self.white, 1)
                
                cv2.imshow('Front View - Head & Shoulders', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
        finally:
            print("Front view camera closing...")
            cap.release()
            cv2.destroyWindow('Front View - Head & Shoulders')

def main(video_side=0, video_front=1, offset_threshold=100, neck_angle_threshold=15, 
         torso_angle_threshold=5, head_tilt_threshold=4, shoulder_tilt_threshold=4,
         time_threshold=10, single_camera=False):
    
    print("\n" + "="*60)
    print("DUAL CAMERA POSTURE MONITOR")
    print("="*60)
    
    # Check available cameras
    print("\nDetecting available cameras...")
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("ERROR: No cameras detected!")
        return
    
    print(f"Available cameras: {available_cameras}")
    
    if video_side not in available_cameras:
        print(f"ERROR: Side view camera {video_side} not available!")
        print(f"Please use one of: {available_cameras}")
        return
    
    use_dual_camera = not single_camera and video_front in available_cameras
    
    if not single_camera and video_front not in available_cameras:
        print(f"WARNING: Front view camera {video_front} not available!")
        print("Running in SINGLE CAMERA mode (side view only)")
        use_dual_camera = False
    
    monitor = PostureMonitor(offset_threshold, neck_angle_threshold, torso_angle_threshold,
                            head_tilt_threshold, shoulder_tilt_threshold, time_threshold)
    
    # Set up signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal (Ctrl+C)")
        monitor.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nStarting monitoring...")
    print("Press 'q' in any window or Ctrl+C to exit\n")
    
    try:
        if use_dual_camera:
            # Create threads for each camera
            side_thread = threading.Thread(target=monitor.process_side_view, args=(video_side,))
            front_thread = threading.Thread(target=monitor.process_front_view, args=(video_front,))
            
            # Start threads
            side_thread.start()
            front_thread.start()
            
            # Wait for threads to complete
            side_thread.join()
            front_thread.join()
        else:
            # Single camera mode
            monitor.process_side_view(video_side)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        monitor.stop()
    finally:
        print("\nShutting down...")
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    args = parse_arguments()
    
    print("\nConfiguration:")
    print(f"  Side View Camera: {args.video_side}")
    print(f"  Front View Camera: {args.video_front}")
    print(f"  Single Camera Mode: {args.single_camera}")
    print(f"  Offset Threshold: {args.offset_threshold}")
    print(f"  Neck Angle Threshold: {args.neck_angle_threshold}")
    print(f"  Torso Angle Threshold: {args.torso_angle_threshold}")
    print(f"  Head Tilt Threshold: {args.head_tilt_threshold}")
    print(f"  Shoulder Tilt Threshold: {args.shoulder_tilt_threshold}")
    print(f"  Time Threshold: {args.time_threshold}s")
    
    main(video_side=args.video_side,
         video_front=args.video_front,
         offset_threshold=args.offset_threshold,
         neck_angle_threshold=args.neck_angle_threshold,
         torso_angle_threshold=args.torso_angle_threshold,
         head_tilt_threshold=args.head_tilt_threshold,
         shoulder_tilt_threshold=args.shoulder_tilt_threshold,
         time_threshold=args.time_threshold,
         single_camera=args.single_camera)