import cv2

def read_video(video_path: str) -> list:
    """
    Reads all frames from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        list: A list containing each frame as a NumPy array.
              Returns an empty list if the video cannot be opened.
    """
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return frames

    while True:
        has_more_frames, frame = video_capture.read()
        if not has_more_frames:
            break # No more frames or error reading frame
        frames.append(frame)

    video_capture.release() # Release the video capture object
    print(f"Read {len(frames)} frames from {video_path}")
    return frames

def save_video(output_frames: list, output_path: str):
    """
    Saves a list of frames as a video file using XVID codec.

    Args:
        output_frames (list): A list of frames (NumPy arrays) to save.
        output_path (str): The path where the output video will be saved.
                           The directory must exist.
    """
    if not output_frames:
        print("Error: No frames provided to save.")
        return

    # Get frame dimensions from the first frame
    frame_height, frame_width, _ = output_frames[0].shape
    frame_size = (frame_width, frame_height)

    # Define the codec and create VideoWriter object
    # Using XVID codec, common for .avi files
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Using a fixed FPS of 24. Consider passing FPS dynamically if needed.
    # TODO: Consider getting FPS from the source video or making it a parameter
    fps = 24
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not out.isOpened():
        print(f"Error: Could not open video writer for path: {output_path}")
        return

    print(f"Saving {len(output_frames)} frames to {output_path} at {fps} FPS...")
    for frame in output_frames:
        # Ensure frame dimensions match the writer's expectations
        if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
            print(f"Warning: Frame dimension mismatch ({frame.shape[1]}x{frame.shape[0]}) expected ({frame_width}x{frame_height}). Resizing.")
            frame = cv2.resize(frame, frame_size)
        out.write(frame)

    out.release() # Release the video writer object
    print(f"Video saved successfully to {output_path}")