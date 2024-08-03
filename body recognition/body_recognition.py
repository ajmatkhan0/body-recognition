import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hand = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hand.Hands()
mp_drawing = mp.solutions.drawing_utils

# Global variables
cap = None
running = False

# Function to get pose keypoints
def get_pose_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results

# Function to get hand keypoints
def get_hand_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results

# Function to start the video capture and processing
def start_video():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    process_video()

# Function to resume the video capture
def resume_video():
    global running
    running = True
    process_video()

# Function to process the video frames
def process_video():
    global cap, running
    if running and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize the frame to 600x400
            frame = cv2.resize(frame, (600, 400))

            # Get pose and hand keypoints
            pose_results = get_pose_keypoints(frame)
            hand_results = get_hand_keypoints(frame)

            # Draw pose landmarks for multiple people
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw hand landmarks for multiple hands
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hand.HAND_CONNECTIONS)

            # Create a mask for the detected pose
            mask = np.zeros(frame.shape, dtype=np.uint8)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(mask, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert the mask to grayscale and find contours
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original frame
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Green color for the contour

            # Convert the frame to a format suitable for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            display_frame.imgtk = imgtk
            display_frame.configure(image=imgtk)
            
        display_frame.after(10, process_video)

# Function to stop the video capture
def stop_video():
    global running
    running = False

# Initialize Tkinter window
root = tk.Tk()
root.title("Yoga Pose Detection")

# Create a frame to display the video
display_frame = Label(root)
display_frame.pack()

# Create Start, Resume, and Stop buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

button_width = 10  # Set a fixed button width

start_button = tk.Button(button_frame, text="Start", command=start_video, width=button_width)
start_button.grid(row=0, column=0, padx=5)

resume_button = tk.Button(button_frame, text="Resume", command=resume_video, width=button_width)
resume_button.grid(row=0, column=1, padx=5)

stop_button = tk.Button(button_frame, text="Stop", command=stop_video, width=button_width)
stop_button.grid(row=0, column=2, padx=5)

# Start the Tkinter main loop
root.mainloop()