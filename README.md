# Vehicle_Driver_State_Estimation

Here is the entire content you requested in **Markdown** format for your GitHub repository:

```markdown
# 🚗 **Drowsiness Detection, rPPG Heart Rate & Yaw Angle Detection** 🎥

Welcome to the **Drowsiness Detection**, **Heart Rate Estimation** (via rPPG), and **Yaw Angle Detection** project! This Python-based system uses **computer vision**, **signal processing**, and **real-time visualization** to analyze your facial movements, detect drowsiness (EAR), estimate your heart rate (rPPG), and track your head position (yaw angle) using only a webcam.

This system could be useful in areas such as driver monitoring, health tracking, and workplace safety.

---

## ✨ **What It Is?**

This project utilizes **computer vision** and **signal processing techniques** to achieve real-time **drowsiness detection**, **heart rate monitoring**, and **head position tracking** from a video stream captured through a webcam. Here’s what the system does:

- **Drowsiness Detection**: Tracks the **Eye Aspect Ratio (EAR)** to detect if the user is drowsy.
- **Heart Rate Detection (rPPG)**: Estimates the **heart rate (BPM)** from the green channel intensity of the webcam video (using **remote Photoplethysmogram** or **rPPG**).
- **Yaw Angle Detection**: Tracks the **yaw angle** of the face to determine the head orientation.

These features are integrated into a real-time application with interactive visual feedback and graphs, providing a comprehensive monitoring system.

---

## 🚀 **How It Is Used**

- **Face Detection**: The system detects facial landmarks using a **pre-trained model** from **dlib**.
- **Drowsiness Detection (EAR)**: The **Eye Aspect Ratio (EAR)** is calculated by tracking the user’s eyes. If EAR falls below a threshold, the system flags drowsiness.
- **Heart Rate Estimation**: By measuring fluctuations in the **green channel** of the video feed, we estimate the heart rate using **remote photoplethysmogram (rPPG)**.
- **Yaw Angle Detection**: Tracks the rotation of the head (side-to-side movement) by analyzing the **yaw angle**.

---

## 🛠 **What Are the Requirements?**

Before running the project, make sure you have Python 3.8+ installed. Below are the required libraries and tools for setting up this project:

- **Python 3.8+** (Recommended)
- **OpenCV**: For video capture and image processing.
- **dlib**: For facial landmark detection.
- **NumPy**: For numerical operations.
- **PyQt5**: For creating the graphical user interface (GUI).
- **pyqtgraph**: For real-time data visualization.
- **Scipy**: For signal processing (filtering the heart rate signal).

To install the required libraries, run the following command:

```bash
pip install opencv-python opencv-python-headless dlib numpy pyqt5 pyqtgraph scipy
```

You'll also need to download the **shape_predictor_68_face_landmarks.dat** file for face landmark detection. [Download it here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

---

## 📝 **How to Set Up**

1. **Clone the Repository**:

   Start by cloning this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Download the Face Landmark Model**:

   Download the **shape_predictor_68_face_landmarks.dat** file from the link provided above and place it inside the `models/` directory.

3. **Install Dependencies**:

   Install all the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 **How to Run the Application**

Once the project is set up, follow these steps to run the application:

1. Make sure your webcam is connected.
2. Open the terminal/command prompt and navigate to the directory where the project is saved.
3. Run the `main.py` script:
   ```bash
   python main.py
   ```

This will launch the application, which will display:

- **Video Feed**: Your webcam feed with real-time drowsiness detection and heart rate estimation.
- **BPM Graph**: A real-time graph showing your heart rate (BPM) over time.
- **Yaw Angle Graph**: A real-time graph showing your yaw angle over time.
- **Green Intensity Graph**: A real-time graph showing the intensity of the green channel from the video feed.

---

## 📊 **What Are the Predictions?**

The system will continuously process the webcam feed to:

- **Detect drowsiness** by calculating the **Eye Aspect Ratio (EAR)**. If the EAR falls below a threshold, the system will flag drowsiness and display "Drowsy: YES" on the video feed.
- **Estimate heart rate (BPM)** by processing fluctuations in the **green channel** of the webcam video. The system will calculate **beats per minute (BPM)** and display it on the graph.
- **Estimate yaw angle** by detecting the user’s head rotation and plotting the angle on the graph.
- **Visual Feedback**: The system will display dynamic graphs for heart rate (BPM), yaw angle, and green intensity over time.

---

## 🔧 **File Structure**

```
project_directory/
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat        # Pre-trained facial landmark model
│
├── utils/
│   ├── face_detection.py        # Face and yaw angle detection, including EAR calculation
│   ├── signal_processing.py     # Signal processing for rPPG heart rate estimation
│   └── visualization.py         # Real-time plotting and data visualization
│
└── main.py                      # Main script to run the application
```

---

## 🔮 **Future Enhancements**

- **Emotion Detection**: Adding functionality to detect emotions (e.g., happy, sad, surprised) based on facial expressions.
- **Advanced Drowsiness Detection**: Using a combination of EAR and other facial features like eye blinking frequency to improve drowsiness detection accuracy.
- **Mobile Support**: Implementing a mobile version of the application for use on smartphones or tablets.
- **Enhanced Heart Rate Monitoring**: Using machine learning to improve the accuracy of heart rate estimation in varying lighting conditions.
- **Data Logging**: Saving heart rate, yaw angle, and drowsiness data to a file for further analysis.

---

## 📜 **License**

This project is open-source and available under the MIT License.

---

## 👤 **Author**

**[Your Name]**  
GitHub: [@your-username](https://github.com/your-username)

---

## 💡 **Acknowledgements**

- **[dlib](http://dlib.net/)**: Used for facial landmark detection.
- **[OpenCV](https://opencv.org/)**: Used for video capture and face detection.
- **[PyQt5](https://riverbankcomputing.com/software/pyqt/intro)**: Used for the GUI and real-time plotting.
- **[PyQtGraph](http://www.pyqtgraph.org/)**: Used for real-time graphing.
- **[Scipy](https://www.scipy.org/)**: Used for signal processing and filtering.
- **Shape Predictor 68**: A pre-trained model for detecting 68 facial landmarks.

---

**Feel free to contribute, and feel free to open issues or pull requests.** Let's make driving and health monitoring safer with real-time feedback!

```

This README file includes all the details you requested, including explanations, setup instructions, and future enhancement ideas, and it's formatted in a colorful and engaging way to attract attention to your GitHub repository. You can copy and paste this content directly into your `README.md` file for your project.
