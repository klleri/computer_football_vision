import os 
# Import necessary classes from project modules
from videos import VideoProcessor
from teams import TeamAssigner
from objects import Objects # Handles object detection, tracking, and drawing


def process_video(
    input_path: str,
    output_path: str,
    model_path: str,
    stub_path: str = None,
):
    """
    Processes a video to detect and track objects, assign teams, analyze ball possession,
    and save the annotated output video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        model_path (str): Path to the object detection model file (e.g., .pt).
        stub_path (str, optional): Path to a pickle file for loading/saving pre-computed
                                   tracking results to speed up subsequent runs. Defaults to None.
    """

    # 1. Initialize Video Processor
    print(f"Loading video from: {input_path}")
    video_processor = VideoProcessor(input_path)
    frames = video_processor.read_frames()
    if not frames:
        print(f"Error: No frames read from video. Please check the input path: {input_path}")
        return
    fps = video_processor.get_fps()
    if fps is None or fps <= 0:
        print(f"Warning: Could not detect video FPS or FPS is zero. Using default FPS = 24.")
        fps = 24 # Set a reasonable default FPS if detection fails
    print(f"Video loaded: {len(frames)} frames, FPS: {fps:.2f}")

    # 2. Initialize Tracker (Objects class)
    # Initialized once with the correct FPS
    print(f"Initializing tracker with model: {model_path}")
    tracker = Objects(model_path, fps=fps)
    print(f"Tracker initialized.")

    # 3. Get Object Tracks
    # Reads from stub if available and requested, otherwise detects and tracks
    print("Detecting and tracking objects...")
    # 'object_tracks' will store tracking data for players, referees, ball per frame
    object_tracks = tracker.get_objects( # Renamed from get_object for clarity
        frames, read_from_stub=stub_path is not None, stub_path=stub_path
    )
    print("Object tracking complete.")

    # 4. Assign Teams to Players
    print("Assigning teams to players...")
    team_assigner = TeamAssigner(
        n_bins=32, # Histogram bins for color analysis
        color_space='HSV', # Color space for analysis
        initialization_method='first_frames', # Method to determine initial team colors
        num_initial_frames=5, # Number of frames for initialization
        # histogram_comparison_method='bhattacharyya' # Default method in TeamAssigner
    )

    # Initialize team histograms based on the first few frames and set drawing colors
    # Note: This modifies self.team_colors in team_assigner (e.g., to Gray and Green)
    team_assigner.assign_team_color(frames, object_tracks['players'])

    # Loop through frames and players to assign team IDs and colors
    # Includes forced assignments below
    print("Applying team assignments (including hardcoded overrides)...")
    for frame_num, player_track_in_frame in enumerate(object_tracks['players']):
        if frame_num < len(frames): # Check frame index validity
            current_frame = frames[frame_num]
            for player_id, track_info in player_track_in_frame.items():
                if 'bbox' in track_info:
                    # 1. Get the team suggested by the automatic assigner
                    suggested_team = team_assigner.get_player_team(
                        current_frame, track_info['bbox'], player_id
                    )

                    # --- START: Hardcoded/Forced Assignment ---
                    # This section overrides the automatic assignment for specific player IDs.
                    # Be aware that this makes the assignment less general.
                    # Player IDs 227 and 166 are forced to Team 1 (White/Gray)
                    if player_id == 227 or player_id == 166: # Combined condition
                        final_team = 1
                    # Player ID 426 is forced to Team 2 (Green)
                    elif player_id == 426:
                        final_team = 2
                    else:
                        # Use the automatically suggested team for all other players
                        final_team = suggested_team
                    # --- END: Hardcoded/Forced Assignment ---

                    # 3. Update the 'object_tracks' dictionary with the final team ID and color
                    object_tracks['players'][frame_num][player_id]['team'] = final_team
                    # Get the corresponding drawing color from the assigner, default to white if not found
                    object_tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(final_team, (255, 255, 255)) # Default White
                # else:
                    # Optional Warning: Player tracked but no bbox in this frame
                    # print(f"Warning: Player {player_id} in frame {frame_num} has no 'bbox'.")
        # else:
            # Optional Warning: More tracking data than video frames
            # print(f"Warning: Frame number {frame_num} is out of video frame range ({len(frames)}).")

    print("Team assignment complete.")

    # 5. Possession Logic (Now handled entirely within Objects class methods)
    # No explicit possession calculation needed here in main.py, it happens during drawing.

    # 6. Draw Annotations (including possession info)
    print("Drawing annotations on frames...")
    # The 'object_tracks' dictionary now contains team information for drawing correct colors.
    # The draw_annotations method calls draw_team_ball_control internally, which uses PossessionHandler.
    output_video_frames = tracker.draw_annotations(frames, object_tracks, fps)
    print("Finished drawing annotations.")

    # 7. Save the output video
    print(f"Saving annotated video to: {output_path}")
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    video_processor.save_frames_as_video(output_video_frames, output_path)
    print("Annotated video saved successfully.")


# Main execution block
if __name__ == '__main__':
    # Define input/output paths and model location
    # Ensure these paths are correct for your system
    INPUT_VIDEO_PATH = 'input/video001.mp4'
    OUTPUT_VIDEO_PATH = 'output/output_video.avi' # Changed output name for clarity
    DETECTION_MODEL_PATH = 'models/best.pt'
    # Set to None to disable stub usage and force detection/tracking
    TRACKING_STUB_PATH = 'stubs/track_stubs.pkl'
    # Ensure stub directory exists if using stubs
    if TRACKING_STUB_PATH:
        stub_dir = os.path.dirname(TRACKING_STUB_PATH)
        if stub_dir and not os.path.exists(stub_dir):
            os.makedirs(stub_dir)
            print(f"Created stub directory: {stub_dir}")


    print("Starting video processing pipeline...")
    process_video(
        input_path=INPUT_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        model_path=DETECTION_MODEL_PATH,
        stub_path=TRACKING_STUB_PATH,
    )
    print("Processing finished.")