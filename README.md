# Indoor-Navigation-For-Blind

# Computer Vision-Based Navigation Assistant for the Visually Impaired

## Project Description

This project aims to develop an assistive technology solution that helps visually impaired individuals navigate indoor environments using computer vision and natural language processing. The system uses a webcam to perceive the surroundings, identify obstacles and room numbers, and provide real-time audio guidance to the user.

## Features

* **Object Detection:** Detects obstacles such as people, chairs, and other moving objects using YOLOv5.
* **Optical Character Recognition (OCR):** Reads room numbers and signs using Tesseract OCR.
* **Pathfinding:** Calculates the shortest path to a destination using the A\* algorithm (Dijkstra's algorithm in this implementation).
* **Text-to-Speech (TTS):** Provides audio feedback, including directions, obstacle warnings, and room number announcements.
* **Speech-to-Text (STT):** Allows users to input their starting location and destination using voice commands.
* **Graphical User Interface (GUI):** A user-friendly interface built with PyQt5 displays the camera feed, path information, and notifications.

## Dependencies

* Python 3.x
* PyTorch
* OpenCV (cv2)
* PyTesseract
* pyttsx3
* SpeechRecognition
* PyQt5
* YOLOv5 (Ultralytics)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    * It is recommended to use a virtual environment (venv or conda) to manage dependencies.

3.  **Install Tesseract OCR:**

    * Download and install Tesseract OCR from [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    * Add the Tesseract executable path to your system's PATH environment variable.  You may need to modify the `tesseract_path` variable in the main Python script to point to the correct location.

4.  **Download YOLOv5 weights (optional):**
    * The code downloads the weights automatically, but if you face issues, you can download them manually from the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) and place them in the appropriate directory.

5.  **Set up your webcam:**
    * Ensure your webcam is connected and accessible to the system. The application will attempt to detect it automatically.

## Usage

1.  **Run the main application script:**

    ```bash
    python main.py
    ```

2.  **Using the GUI:**

    * Enter the starting location and destination in the provided text fields.
    * Click the "Navigate" button to start the navigation.
    * Click the "Voice Input" button to use voice commands for input.
    * The camera feed will be displayed in the GUI window.
    * Audio feedback will be provided throughout the navigation process.

## Code Structure

* `main.py`: Contains the main application logic, including GUI setup, camera handling, pathfinding, object detection, and text recognition.
* `requirements.txt`: Lists the Python packages required to run the application.
* `# (Any other relevant files/directories)`:  Add descriptions for any other important files or directories in your repository.

## Future Enhancements

* Improve the robustness and accuracy of object detection and OCR.
* Implement more advanced pathfinding features, such as dynamic obstacle avoidance.
* Integrate with wearable devices, such as smart glasses.
* Develop a more user-friendly map creation tool.
* Add support for different languages.
* Explore more sophisticated TTS options for more natural-sounding speech.

## Contributing

Contributions are welcome! If you have any ideas for improvements or find any bugs, please feel free to open an issue or submit a pull request.





