import cv2
import pytesseract
import pyttsx3
import torch
import heapq
from collections import deque
import threading
import time
import sys
import speech_recognition as sr
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# --------- TTS Setup (Offline) ---------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# --------- STT Setup (Online) ---------
def listen():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            speak("Please say your input.")
            print("🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        text = recognizer.recognize_google(audio)
        print(f"🗣 Recognized: {text}")
        return text
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that. Please try again.")
        print("STT Error: Could not understand audio")
        return ""
    except sr.RequestError as e:
        speak("There is a problem with the speech recognition service.")
        print(f"STT Error: {e}")
        return ""
    except Exception as e:
        speak("Error accessing microphone. Please check your audio device.")
        print(f"STT Error: {e}")
        return ""

# Set Tesseract path
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("Error: Tesseract executable not found at specified path")
    speak("Error: Tesseract OCR not found. Text detection disabled.")

# Load YOLOv5 model
model = None
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    speak("Error loading object detection model. Obstacle detection disabled.")

# Heuristic function
def heuristic(a, b):
    return 0  # Dijkstra (no heuristic)

# Check graph connectivity
def is_connected(start, goal, graph):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False

# A* Algorithm
def a_star(start, goal, graph):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        for neighbor in graph.get(current, {}):
            tentative_g_score = g_score[current] + graph[current][neighbor]
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                heapq.heappush(open_set, (tentative_g_score, neighbor))

    return None, float('inf')

# Function to detect and verify text with visual feedback
def detect_text(frame, expected_room, next_room, direction, distance, path_index, path):
    global last_text_detection, text_detection_cooldown
    if time.time() - last_text_detection < text_detection_cooldown:
        return False, path_index

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        digits = ''.join(filter(str.isdigit, text))

        if digits:
            cv2.putText(frame, f"Detected: {digits}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No digits detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if digits:
            print(f"Detected digits: {digits}")
            last_text_detection = time.time()
            if digits == expected_room:
                if path_index < len(path) - 1:
                    speak(f"{direction} to room {next_room}, distance {distance:.2f} meters.")
                else:
                    speak(f"You have reached room {digits}.")
                return True, path_index + 1 if path_index < len(path) - 1 else path_index
            else:
                speak(f"This is room {digits}. Please go to room {expected_room}.")
                return False, path_index
    except Exception as e:
        print(f"Text Detection Error: {e}")
        speak("Error in text detection.")
    return False, path_index

# Function to detect objects
def detect_objects(frame):
    if model is None:
        return
    try:
        results = model(frame)
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label in ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']:
                speak(f"{label} ahead. Please be careful.")
                break
    except Exception as e:
        print(f"Object Detection Error: {e}")
        speak("Error in object detection.")

# GUI Application
class NavigationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navigation Assistant")
        self.setGeometry(100, 100, 800, 600)

        # Graph and directions
        self.graph = {
            "115": {"114": 4.27, "MA155": 26.05},
            "114": {"115": 4.27, "113": 16.91},
            "113": {"114": 16.91, "112": 5.98},
            "112": {"113": 5.98, "111": 6.11},
            "111": {"112": 6.11, "110": 4.42},
            "110": {"111": 4.42, "109": 10.10},
            "109": {"110": 10.10, "108": 15.95},
            "108": {"109": 15.95, "107": 4.22, "MA155": 30.39},
            "107": {"108": 4.22, "106": 16.00},
            "106": {"107": 16.00, "105": 6.14},
            "105": {"106": 6.14, "104": 4.42},
            "104": {"105": 4.42, "103": 6.72},
            "103": {"104": 6.72, "102": 16.80},
            "102": {"103": 16.80, "101": 5.17},
            "101": {"102": 5.17, "MA150": 26.36},
            "MA155": {"108": 30.39, "115": 26.05, "MA150": 30.0},
            "MA150": {"MA155": 30.0, "101": 26.36}
        }
        self.directions = {
            ("105", "106"): "Turn right.",
            ("110", "111"): "Turn right.",
            ("115", "MA155"): "Turn right.",
            ("MA150", "101"): "Turn right.",
            ("108", "MA155"): "Turn left.",
            ("MA150", "115"): "Turn left."
        }

        # Initialize webcam
        self.cap = None
        for index in range(3):  # Try indices 0, 1, 2
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                print(f"Webcam opened successfully on index {index}")
                break
            self.cap.release()
        if not self.cap or not self.cap.isOpened():
            self.cap = None
            self.show_notification("Error: Could not open webcam. Please check camera connection.")

        # Global variables
        self.path = []
        self.path_index = 0
        self.destination_reached = False
        global last_text_detection, text_detection_cooldown
        last_text_detection = time.time()
        text_detection_cooldown = 5

        # Setup UI
        self.setup_ui()

        # Timer for updating camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)  # Update every 100ms

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Inputs and navigation info
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        # Input fields
        input_layout = QGridLayout()
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("Enter Start Location (e.g., 115)")
        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("Enter Destination (e.g., MA155)")
        navigate_button = QPushButton("Navigate")
        navigate_button.clicked.connect(self.start_navigation)
        voice_input_button = QPushButton("Voice Input")
        voice_input_button.clicked.connect(self.start_voice_input)

        input_layout.addWidget(QLabel("Start Location:"), 0, 0)
        input_layout.addWidget(self.start_input, 0, 1)
        input_layout.addWidget(QLabel("Destination:"), 1, 0)
        input_layout.addWidget(self.goal_input, 1, 1)
        input_layout.addWidget(navigate_button, 2, 0)
        input_layout.addWidget(voice_input_button, 2, 1)
        left_layout.addLayout(input_layout)

        # Path display
        self.path_label = QLabel("Path: Not calculated")
        self.distance_label = QLabel("Distance: 0.00 meters")
        self.time_label = QLabel("Estimated Time: 0 minutes 0 seconds")
        left_layout.addWidget(self.path_label)
        left_layout.addWidget(self.distance_label)
        left_layout.addWidget(self.time_label)

        # Notifications
        self.notification_area = QTextEdit()
        self.notification_area.setReadOnly(True)
        left_layout.addWidget(QLabel("Notifications:"))
        left_layout.addWidget(self.notification_area)

        # Right panel: Camera feed
        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.camera_label, 2)

    def show_notification(self, message):
        self.notification_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        speak(message)
        print(f"Notification: {message}")

    def start_voice_input(self):
        try:
            speak("Please say the starting location.")
            start = listen()
            if start:
                self.start_input.setText(start)
                speak(f"Start location set to {start}. Please say the destination.")
                goal = listen()
                if goal:
                    self.goal_input.setText(goal)
                    speak(f"Destination set to {goal}. Starting navigation.")
                    self.start_navigation()
                else:
                    speak("No destination recognized. Please try again.")
            else:
                speak("No start location recognized. Please try again.")
        except Exception as e:
            self.show_notification(f"Voice input error: {e}")

    def start_navigation(self):
        try:
            start = self.start_input.text().strip()
            goal = self.goal_input.text().strip()

            graph_keys = {k.lower(): k for k in self.graph.keys()}
            start = graph_keys.get(start.lower(), start)
            goal = graph_keys.get(goal.lower(), goal)

            if start not in self.graph or goal not in self.graph:
                self.show_notification(f"Error: Invalid locations. Valid options: {list(self.graph.keys())}")
                return

            if not is_connected(start, goal, self.graph):
                self.show_notification(f"No possible path exists between {start} and {goal}.")
                return

            self.path, total_distance = a_star(start, goal, self.graph)
            if not self.path:
                self.show_notification("Path not found! Check if locations are connected.")
                return

            self.path_index = 0
            self.destination_reached = False
            self.path_label.setText(f"Path: {' -> '.join(self.path)}")
            self.distance_label.setText(f"Distance: {total_distance:.2f} meters")
            avg_speed = 1.4
            estimated_time = total_distance / avg_speed
            minutes = int(estimated_time // 60)
            seconds = int(estimated_time % 60)
            self.time_label.setText(f"Estimated Time: {minutes} minutes {seconds} seconds")

            numeric_nodes = [node for node in self.path if node.isdigit()]
            self.show_notification(f"Starting navigation from {start} to {goal}.")
            self.show_notification(f"Total distance is {int(total_distance)} meters. Estimated time is {minutes} minutes {seconds} seconds.")

            required_number = goal if goal.isdigit() else None
            if not required_number or not numeric_nodes:
                self.show_notification(f"Warning: Destination {goal} is not numeric or path has no numeric nodes. Text verification disabled.")
        except Exception as e:
            self.show_notification(f"Navigation error: {e}")

    def update_frame(self):
        if not self.cap:
            self.camera_label.setText("No camera available")
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.show_notification("Error: Failed to capture frame. Check camera connection.")
                return

            if self.path and not self.destination_reached:
                expected_room = self.path[self.path_index] if self.path_index < len(self.path) else self.path[-1]
                next_room = self.path[self.path_index + 1] if self.path_index + 1 < len(self.path) else None
                direction = self.directions.get((expected_room, next_room), "Move straight") if next_room else ""
                distance = self.graph[expected_room][next_room] if next_room else 0
                verified, new_path_index = detect_text(frame, expected_room, next_room, direction, distance, self.path_index, self.path)
                if verified:
                    self.path_index = new_path_index
                    if self.path_index >= len(self.path) - 1:
                        self.destination_reached = True
                        self.show_notification("Navigation complete. You have arrived at your destination.")

            detect_objects(frame)

            # Convert frame to QImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.camera_label.size(), Qt.KeepAspectRatio))
        except Exception as e:
            self.show_notification(f"Frame update error: {e}")

    def closeEvent(self, event):
        try:
            if self.cap:
                self.cap.release()
            engine.stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        event.accept()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = NavigationApp()
        window.show()
        speak("Navigation assistant started. Press the voice input button to begin.")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        speak("Application error. Please restart.")