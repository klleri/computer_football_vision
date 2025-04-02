import cv2
from .read_save import read_video, save_video # Import local helper functions

class VideoProcessor:
    """
    A class for handling video reading, saving, and retrieving properties like FPS.
    """

    def __init__(self, video_path: str):
        """
        Initializes the VideoProcessor with the path to the video file.

        Args:
            video_path (str): Path to the input video file.
        """
        self.video_path = video_path
        self.frames = [] # To store frames after reading

    def read_frames(self) -> list:
        """
        Reads all frames from the video file specified during initialization.

        Returns:
            list: A list of frames (NumPy arrays). Stores frames internally.
        """
        print(f"VideoProcessor: Reading frames from {self.video_path}")
        self.frames = read_video(self.video_path)
        return self.frames

    def save_frames_as_video(self, output_frames: list, output_path: str):
        """
        Saves the provided list of frames as a video file.

        Args:
            output_frames (list): List of frames (NumPy arrays) to be saved.
            output_path (str): Path to save the output video file.
        """
        print(f"VideoProcessor: Saving frames to {output_path}")
        save_video(output_frames, output_path)

    def get_fps(self) -> float | None:
        """
        Gets the frames per second (FPS) rate of the video.

        Returns:
            float or None: The FPS of the video, or None if it cannot be determined.
        """
        video_capture = cv2.VideoCapture(self.video_path)
        if not video_capture.isOpened():
            print(f"Error: Could not open video {self.video_path} to get FPS.")
            return None
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        print(f"VideoProcessor: Detected FPS = {fps}")
        return fps