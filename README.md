## Football Computer Vision Analysis

This project uses computer vision techniques to analyze football (soccer) match videos. It processes video footage to detect and track players, referees, and the ball, assign players to teams, calculate ball possession statistics, and generate an annotated output video.

### Key Features:

* **Object Detection:** Utilizes a YOLOv8 model (specified via `models/best.pt`) to detect players, referees, and the ball in each video frame [cite: uploaded:computer_football_vision/objects/objects.py, uploaded:computer_football_vision/main.py]. Goalkeepers are optionally re-classified as players.
* **Object Tracking:** Employs the ByteTrack algorithm (via the `supervision` library) to assign and maintain unique IDs for detected players and referees across frames [cite: uploaded:computer_football_vision/objects/objects.py].
* **Team Assignment:** Assigns detected players to one of two teams (Team 1 - Gray, Team 2 - Green) based on jersey color analysis. It extracts color histograms (HSV, Lab, or YCrCb) from the jersey area and uses K-means clustering on the initial frames to establish team color profiles [cite: uploaded:computer_football_vision/teams/teams.py]. Allows for hardcoded overrides for specific player IDs [cite: uploaded:computer_football_vision/main.py].
* **Ball Position Interpolation:** Linearly interpolates the ball's bounding box position for frames where detection might have failed, providing a smoother apparent trajectory [cite: uploaded:computer_football_vision/objects/objects.py].
* **Ball Possession Calculation:** Determines the player closest to the ball in each frame and uses a `PossessionHandler` class with a sliding window mechanism to calculate smoothed ball possession time percentages for each team, reducing flicker from momentary changes [cite: uploaded:computer_football_vision/objects/objects.py, uploaded:computer_football_vision/objects/possession_handler.py].
* **Video Annotation:** Generates an output video with visual annotations drawn on each frame [cite: uploaded:computer_football_vision/objects/objects.py]:
    * Ellipses drawn at the feet of players (colored by assigned team) and referees.
    * Track IDs displayed above players.
    * An indicator triangle drawn above the ball and the player currently determined to be in possession.
    * A display showing accumulated ball possession time and percentage for each team.
* **Stub File Caching:** Supports loading and saving computed object tracks to a `.pkl` file (`stubs/track_stubs.pkl`) to speed up subsequent runs on the same video [cite: uploaded:computer_football_vision/objects/objects.py, uploaded:computer_football_vision/main.py].

### Demo:



### Core Libraries:

* OpenCV (`cv2`)
* Ultralytics YOLO
* Supervision (`supervision`)
* NumPy
* Pandas (for ball interpolation)
* Scikit-learn (for K-means in team assignment)

### How to Run:

1.  Ensure you have the necessary libraries installed.
2.  Place your input video file (e.g., `input/video001.mp4`).
3.  Place your trained YOLO model (e.g., `models/best.pt`).
4.  Run the main script: `python main.py` [cite: uploaded:computer_football_vision/main.py].
5.  The annotated output video will be saved (e.g., `output/output_video_annotated.avi`).

*(You can customize the paths and parameters within `main.py`)*

---
