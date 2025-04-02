from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

try:
    # Get the directory containing the current file (__file__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (computer_football_vision)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    # Add the parent directory to sys.path if not already present
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    # Now import modules from the videos and objects packages
    from videos import bounding_box_center, measure_distance 
    from .possession_handler import PossessionHandler
except ImportError as e:
    print(f"Critical Error: Failed to import required project modules: {e}")
    print("Please ensure the script is run from within the 'computer_football_vision' project structure,")
    print("or that the project's root directory is included in your PYTHONPATH.")
    sys.exit(1) # Exit if essential imports fail


class Objects:
    """
    Handles object detection, tracking (ByteTrack via Supervision),
    drawing annotations, interpolating ball positions, and managing ball
    possession calculation logic through the PossessionHandler.
    """
    def __init__(self, model_path: str, fps: float):
        """
        Initializes the YOLO model, ByteTrack tracker, and PossessionHandler.

        Args:
            model_path (str): Path to the YOLO model file (e.g., 'best.pt').
            fps (float): Frames per second of the video, used by PossessionHandler.
        """
        try:
            print(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path) 
            print("Initializing ByteTrack tracker...")
            self.tracker = sv.ByteTrack()  # Initialize ByteTrack tracker from supervision library
            print("Initializing Possession Handler...")
            self.possession_handler = PossessionHandler(fps)  # Initialize the possession handler, passing the video FPS
            print(f"Objects class initialized successfully. Using FPS: {fps:.2f}")
        except Exception as e:
            print(f"Fatal Error during Objects initialization: {e}")
            print("Please check the model path and ensure necessary libraries (ultralytics, supervision) are installed.")
            raise # Re-raise the exception to halt execution if core components fail


    def interpolate_ball_positions(self, ball_positions_per_frame: list) -> list:
        """
        Interpolates missing ball bounding boxes using linear interpolation based on Pandas.

        Args:
            ball_positions_per_frame (list): A list where each element is a dictionary
                                             representing ball detections in a frame.
                                             Format: [{1: {'bbox': [x1,y1,x2,y2]}}, {}, {1: {...}}, ...]
                                             An empty dict {} means no ball was detected in that frame.
                                             Assumes ball track ID is always 1.

        Returns:
            list: A list of the same structure, but with missing ball 'bbox' values
                  linearly interpolated where possible. Frames where interpolation wasn't
                  possible will still have empty dicts.
        """
        # Extract only the bounding boxes, using None if a frame has no ball detection
        # Assumes the ball always has track_id 1 if detected
        bboxes = [frame_data.get(1, {}).get('bbox', None) for frame_data in ball_positions_per_frame]

        # Create a Pandas DataFrame
        df_ball_positions = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Convert columns to numeric, coercing errors (like None) to NaN
        # This prepares the DataFrame for interpolation
        for col in df_ball_positions.columns:
            df_ball_positions[col] = pd.to_numeric(df_ball_positions[col], errors='coerce')

        # Interpolate missing values (NaN) using linear method
        # `limit_direction='both'` helps fill NaNs at the start and end using nearest valid values
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit_direction='both')

        # Fill any remaining NaNs (if the ball was never detected at all)
        # Using back-fill then forward-fill ensures data propagates from any valid point.
        df_ball_positions = df_ball_positions.bfill().ffill()

        # Convert the interpolated DataFrame back to a list of lists/arrays
        # Fill potential remaining NaNs (if all interpolation failed) with a placeholder like -1
        # to distinguish from valid coordinates during reconstruction.
        interpolated_bboxes_list = df_ball_positions.fillna(-1).to_numpy().tolist()

        # Reconstruct the original list of dictionaries structure
        final_interpolated_ball_positions = []
        for bbox_row in interpolated_bboxes_list:
             # Check if the row contains valid data (4 numbers and not the -1 placeholder)
             if len(bbox_row) == 4 and bbox_row[0] != -1:
                 # Ensure bbox values are standard Python floats/ints if needed downstream
                 valid_bbox = [float(coord) for coord in bbox_row]
                 # Recreate the dict structure {1: {'bbox': [...]}}
                 final_interpolated_ball_positions.append({1: {"bbox": valid_bbox}})
             else:
                 # If interpolation failed or the placeholder remains, add an empty dict
                 final_interpolated_ball_positions.append({})

        print(f"Ball interpolation complete. Resulting list length: {len(final_interpolated_ball_positions)}")
        return final_interpolated_ball_positions

    def detect_frames(self, frames: list) -> list:
        """
        Performs object detection on a list of frames using the loaded YOLO model.
        Processes frames in batches for potential efficiency gains.

        Args:
            frames (list): A list of video frames (NumPy arrays in BGR format).

        Returns:
            list: A list containing the detection results from the YOLO model for each frame/batch.
                  The exact format depends on the ultralytics library version.
        """
        batch_size = 20 # Number of frames to process in a single batch
        all_detections = []
        confidence_threshold = 0.3 # Minimum confidence score for a detection to be kept
        num_frames = len(frames)
        print(f"Starting object detection on {num_frames} frames (batch size: {batch_size}, conf: {confidence_threshold})...")

        for i in range(0, num_frames, batch_size):
            start_index = i
            end_index = min(i + batch_size, num_frames)
            frame_batch = frames[start_index:end_index]

            if not frame_batch: # Skip if batch is somehow empty
                continue

            try:
                # Perform prediction on the batch
                detections_batch = self.model.predict(
                    source=frame_batch,
                    conf=confidence_threshold,
                    verbose=False # Suppress detailed YOLO per-image logging
                )
                # `detections_batch` is usually a list of Results objects, one per image in the batch
                all_detections.extend(detections_batch)
            except Exception as e:
                print(f"Error during YOLO prediction on frame batch {start_index}-{end_index}: {e}")
               
        print(f"Object detection complete. Processed {len(all_detections)} detection results (should match frame count if no errors).")
        return all_detections


    def get_objects(self, frames: list, read_from_stub: bool = False, stub_path: str = None) -> dict:
        """
        Orchestrates object detection and tracking.
        Either runs the full detection and tracking pipeline or loads pre-computed
        tracking results from a pickle file (stub) if available and requested.
        Also performs ball position interpolation.

        Args:
            frames (list): List of video frames (NumPy arrays).
            read_from_stub (bool): If True, attempt to load tracks from `stub_path`.
            stub_path (str, optional): Path to the .pkl stub file containing pre-computed tracks.

        Returns:
            dict: A dictionary containing the tracked objects for each frame.
                  Format: {'players': [frame0_players, frame1_players, ...],
                           'referees': [frame0_referees, ...],
                           'ball': [frame0_ball, ...]}
                  Where frameX_players is a dict like {track_id: {'bbox': [...]}, ...},
                  frameX_referees is similar, and frameX_ball is {1: {'bbox': [...]}} or {}.
        """
        # 1. Attempt to load tracks from the stub file if requested and the file exists
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Attempting to load object tracks from stub file: {stub_path}")
            try:
                with open(stub_path, 'rb') as f:
                    tracks = pickle.load(f)
                print("Successfully loaded tracks from stub file.")
                return tracks
            except Exception as e:
                print(f"Warning: Failed to read or parse stub file '{stub_path}': {e}. Proceeding with detection and tracking.")

        # 2. --- Run Detection and Tracking Pipeline ---
        print("Stub file not used or failed. Running detection...")
        # Perform detection on all frames
        detections_per_frame = self.detect_frames(frames)

        # Initialize data structure to store tracks for each object type per frame
        num_frames = len(frames)
        tracks = {
            "players"   : [{} for _ in range(num_frames)],  # List of dicts for player tracks per frame
            "referees"  : [{} for _ in range(num_frames)], # List of dicts for referee tracks per frame
            "ball"      : [{} for _ in range(num_frames)]      # List of dicts for ball tracks per frame (usually only ID 1)
        }

        print("Starting object tracking using ByteTrack...")
        # Process detections frame by frame to perform tracking
        for frame_num, yolo_result in enumerate(detections_per_frame):
            # Check if the detection result is valid
            if yolo_result is None or yolo_result.boxes is None:
                # print(f"Warning: No valid detection data found for frame {frame_num}. Skipping tracking for this frame.") # Optional warning
                continue # Move to the next frame

            # Get class names from the model (e.g., {0: 'player', 1: 'referee', 2: 'ball', 3: 'goalkeeper'})
            class_names_map = yolo_result.names
            # Create a reverse map for convenience: {'player': 0, 'referee': 1, ...}
            class_name_to_id = {name: cid for cid, name in class_names_map.items()}

            # Convert YOLO detection results to Supervision Detections format
            try:
                detections_sv = sv.Detections.from_ultralytics(yolo_result)
            except Exception as e:
                continue # Skip frame if conversion fails

            goalkeeper_class_id = class_name_to_id.get('goalkeeper')
            player_class_id = class_name_to_id.get('player')
            if goalkeeper_class_id is not None and player_class_id is not None:
                 # Iterate through detected class IDs and change goalkeeper IDs to player IDs
                 detections_sv.class_id = np.where(detections_sv.class_id == goalkeeper_class_id,
                                                   player_class_id,
                                                   detections_sv.class_id)
                
            # Update the ByteTrack tracker with the detections for the current frame
            try:
                # The tracker updates its internal state and returns detections associated with track IDs
                tracked_detections = self.tracker.update_with_detections(detections_sv)
            except Exception as e:
                continue # Skip frame if tracker update fails

            # Store the tracked object data (bounding box and track ID) for players and referees
            referee_class_id = class_name_to_id.get('referee')
            # Note: Player class ID might have changed if remapping occurred, use the potentially updated value
            current_player_class_id = class_name_to_id.get('player')

            for detection_data in tracked_detections:
                # Supervision Detections usually yield tuples/arrays like:
                # (xyxy, mask, confidence, class_id, tracker_id, data_dict)
                # We need xyxy (index 0), class_id (index 3), tracker_id (index 4).
                try:
                    bbox_xyxy = detection_data[0].tolist() # Bounding box coordinates
                    detected_class_id = detection_data[3]  # Original detected class ID
                    tracker_id = detection_data[4]         # Track ID assigned by ByteTrack

                    # Store based on class ID
                    if detected_class_id == current_player_class_id:
                        tracks["players"][frame_num][tracker_id] = {"bbox": bbox_xyxy}
                    elif detected_class_id == referee_class_id:
                        tracks["referees"][frame_num][tracker_id] = {"bbox": bbox_xyxy}
                    # Other classes are ignored for tracking storage here (except ball, handled below)

                except IndexError:
                    continue # Skip this problematic detection data

            # Store detected ball position separately (using a fixed ID '1' for the ball track)
            ball_class_id = class_name_to_id.get('ball')
            if ball_class_id is not None:
                # Find detections matching the ball class ID in the Supervision Detections object
                ball_detections = detections_sv[detections_sv.class_id == ball_class_id]
                if len(ball_detections) > 0:
                    # If multiple balls detected, just take the first one's bounding box
                    ball_bbox = ball_detections.xyxy[0].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": ball_bbox} # Assign to fixed track ID 1

        print("Object tracking complete.")

        # 3. Interpolate ball positions after processing all frames
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        # 4. Save the computed tracks to the stub file if a path was provided
        if stub_path is not None:
            print(f"Saving computed tracks to stub file: {stub_path}")
            try:
                # Ensure the directory exists before trying to save
                stub_dir = os.path.dirname(stub_path)
                if stub_dir and not os.path.exists(stub_dir):
                    os.makedirs(stub_dir)
                    print(f"Created directory for stub file: {stub_dir}")
                # Save the tracks dictionary using pickle
                with open(stub_path, 'wb') as f:
                    pickle.dump(tracks, f)
                print("Tracks successfully saved to stub file.")
            except Exception as e:
                print(f"Error saving tracks to stub file '{stub_path}': {e}")

        return tracks


    def draw_ellipse(self, frame: np.ndarray, bbox: list, color: tuple[int, int, int],
                     track_id: int = None) -> np.ndarray:
        """
        Draws a colored ellipse at the bottom-center of a bounding box.
        Optionally, draws the track ID in a colored box above the ellipse.

        Args:
            frame (np.ndarray): The frame to draw on (BGR format).
            bbox (list): The bounding box [x1, y1, x2, y2].
            color (tuple[int, int, int]): The BGR color for the ellipse and ID background.
            track_id (int, optional): If provided, the track ID is drawn above the ellipse. Defaults to None.

        Returns:
            np.ndarray: The frame with the ellipse (and optionally ID) drawn.
        """
        try:
            # Ensure bounding box coordinates are integers
            x1, y1, x2, y2 = map(int, bbox)

            # Basic validation: check if coordinates make sense
            if x1 >= x2 or y1 >= y2:
                # print(f"Warning: Invalid bbox for drawing ellipse: {bbox}") # Optional log
                return frame # Return original frame if bbox is invalid

            # Calculate ellipse parameters
            center_x = (x1 + x2) // 2
            box_width = x2 - x1
            # Controls how 'flat' the ellipse is (relative to box width)
            ellipse_vertical_axis_ratio = 0.35
            ellipse_height = int(ellipse_vertical_axis_ratio * box_width)
            ellipse_width_radius = box_width // 2

            # Draw the ellipse using cv2.ellipse
            cv2.ellipse(
                img=frame,
                center=(center_x, y2), # Anchor ellipse at bottom-center of the bbox
                axes=(ellipse_width_radius, ellipse_height), # Width and height radius
                angle=0.0,              # Rotation angle (0 for upright)
                startAngle=-45,         # Start angle for drawing (creates a partial ellipse effect)
                endAngle=235,           # End angle for drawing
                color=color,            # BGR color provided
                thickness=2,            # Line thickness
                lineType=cv2.LINE_AA    # Use anti-aliasing for smoother lines
            )

            # Draw Track ID if provided
            if track_id is not None:
                id_text = str(track_id)
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_color = (0, 0, 0) # Black text usually provides good contrast on colored backgrounds

                # Get text size to calculate background rectangle dimensions
                (text_width, text_height), baseline = cv2.getTextSize(id_text, font_face, font_scale, font_thickness)

                # Calculate position for the text and its background rectangle
                # Position it slightly above the top of the drawn ellipse
                ellipse_top_y = y2 - ellipse_height # Approximate top Y of the ellipse visual
                padding = 3 # Padding around the text within the background rectangle
                gap = 5     # Gap between ellipse top and background rectangle bottom

                # Background rectangle coordinates
                rect_bottom_y = ellipse_top_y - gap
                rect_top_y = rect_bottom_y - text_height - baseline - (2 * padding)
                rect_center_x = center_x
                rect_left_x = rect_center_x - (text_width // 2) - padding
                rect_right_x = rect_center_x + (text_width // 2) + padding

                # Text position (baseline origin) - centered within the background rectangle
                text_x = rect_center_x - (text_width // 2)
                text_y = rect_bottom_y - baseline - padding

                # Draw the filled background rectangle using the provided color
                cv2.rectangle(frame, (rect_left_x, rect_top_y), (rect_right_x, rect_bottom_y),
                              color, cv2.FILLED)

                # Draw the text ID on top of the background
                cv2.putText(frame, id_text, (text_x, text_y),
                            font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        except Exception as e:
            pass # Continue processing other objects/frames
        return frame # Return the modified (or original if error) frame

    def draw_triangle(self, frame: np.ndarray, bbox: list, color: tuple[int, int, int]) -> np.ndarray:
        """
        Draws a small, filled indicator triangle above the top-center of a bounding box.
        Useful for indicating the ball or player in possession.

        Args:
            frame (np.ndarray): The frame to draw on (BGR format).
            bbox (list): The bounding box [x1, y1, x2, y2].
            color (tuple[int, int, int]): The BGR color for the triangle.

        Returns:
            np.ndarray: The frame with the triangle drawn.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
             # Basic validation
            if x1 >= x2 or y1 >= y2: return frame

            # Calculate triangle parameters
            center_x = (x1 + x2) // 2
            triangle_height = 10  # Height of the indicator triangle
            triangle_base_half_width = 7 # Half the width of the triangle base
            gap_above_box = 5     # Space between bbox top and triangle bottom

            # Define the triangle vertices (pointing downwards, positioned above the bbox)
            # Points are defined as [(x, y)]
            triangle_points = np.array([
                [center_x, y1 - triangle_height - gap_above_box],      # Top vertex (tip)
                [center_x - triangle_base_half_width, y1 - gap_above_box],   # Bottom-left vertex
                [center_x + triangle_base_half_width, y1 - gap_above_box]    # Bottom-right vertex
            ], dtype=np.int32) # Use int32 for OpenCV drawing functions

            # Draw the filled triangle using the specified color
            cv2.drawContours(image=frame, contours=[triangle_points], contourIdx=0,
                             color=color, thickness=cv2.FILLED)
            # Draw a thin black outline for better visibility, especially on similar background colors
            cv2.drawContours(image=frame, contours=[triangle_points], contourIdx=0,
                             color=(0, 0, 0), thickness=1) # Black outline

        except Exception as e:
             # Log drawing errors if they occur
             # print(f"Warning: Error drawing triangle indicator (bbox: {bbox}): {e}")
            pass
        return frame # Return the modified frame

    def get_ball_possessor(self, players_in_frame: dict, ball_dict_in_frame: dict) -> int:
        """
        Determines which player is closest to the ball in the current frame,
        based on the Euclidean distance between the centers of their bounding boxes.
        Requires the player to be within a defined maximum distance threshold.

        Args:
            players_in_frame (dict): Player tracks for the current frame.
                                     Format: {track_id: {'bbox': [...]}, ...}.
            ball_dict_in_frame (dict): Ball track(s) for the current frame.
                                       Format: {1: {'bbox': [...]}} or {}. Assumes ball ID is 1.

        Returns:
            int: Track ID of the player considered closest to the ball and within the
                 distance threshold. Returns -1 if no ball is detected, no players
                 are detected, or no player is within the threshold distance.
        """
        assigned_player_id = -1
        min_distance_to_ball = float('inf')
        ball_center_position = None
        ball_bbox = None

        # --- Configuration: Possession Threshold ---
        # Maximum distance (in pixels) between a player's center and the ball's center
        # for the player to be considered potentially in possession.
        # *** This value likely needs tuning based on video resolution and typical gameplay distances. ***
        MAX_PLAYER_BALL_DISTANCE_THRESHOLD = 80.0
        # ---

        # 1. Safely get ball information (assuming ball track ID is 1)
        ball_info = ball_dict_in_frame.get(1, {}) # Get data for ball ID 1, default to empty dict
        ball_bbox_candidate = ball_info.get('bbox') # Get the bbox if it exists

        # 2. Validate ball bbox and calculate its center position
        if ball_bbox_candidate and len(ball_bbox_candidate) == 4:
            ball_bbox = ball_bbox_candidate
            try:
                ball_center_position = bounding_box_center(ball_bbox)
            except Exception as e:
                 # Log error if center calculation fails (e.g., due to invalid bbox values)
                 print(f"Warning: Error calculating ball center from bbox {ball_bbox}: {e}")
                 return -1 # Cannot determine possession without ball center

        # 3. If ball position is not available, no possession can be determined
        if ball_center_position is None:
            # print("Debug: No valid ball position found in this frame.") # Optional log
            return -1

        # 4. Iterate through players detected in the current frame
        for player_id, player_info in players_in_frame.items():
            player_bbox = player_info.get('bbox')
            # Validate player bounding box
            if not player_bbox or len(player_bbox) != 4:
                # print(f"Debug: Skipping player {player_id} due to missing/invalid bbox.") # Optional log
                continue # Skip player if bbox is missing or invalid

            try:
                # Calculate the center of the player's bounding box
                player_center_position = bounding_box_center(player_bbox)

                # Calculate the Euclidean distance between the player's center and the ball's center
                distance = measure_distance(player_center_position, ball_center_position)

                # Check if this player is the closest found so far *AND*
                # if the distance is within the defined threshold
                if distance < min_distance_to_ball and distance < MAX_PLAYER_BALL_DISTANCE_THRESHOLD:
                    min_distance_to_ball = distance
                    assigned_player_id = player_id
                    # print(f"Debug: Player {player_id} is now closest at distance {distance:.2f}") # Optional log

            except Exception as e:
                 # Log error if distance calculation fails for a specific player
                 print(f"Warning: Error calculating distance for player {player_id} (bbox={player_bbox}): {e}")
                 continue # Skip processing this player and move to the next

        # 5. Return the ID of the closest player found within the threshold, or -1 if none met the criteria
        # if assigned_player_id == -1: print("Debug: No player found within threshold distance.") # Optional log
        return assigned_player_id



    def draw_annotations(self, video_frames, tracks, fps):
        """
        Draws all annotations (players, referees, ball, possession display) onto the video frames.

        Args:
            video_frames (list): The list of original video frames.
            tracks (dict): The dictionary containing all tracking data (including team info).
                           Format: {'players': [frame0_players, ...], 'referees': [...], 'ball': [...]},
                           where frameX_players is {track_id: {'bbox': ..., 'team': ..., 'team_color': ...}}
            fps (float): Frames per second (potentially needed for future annotations).

        Returns:
            list: A list of annotated video frames.
        """
        output_video_frames = []
        num_frames = len(video_frames)
        print("Starting annotation drawing process...")

        # Optional: Reset possession handler at the beginning if needed
        # self.possession_handler.reset()

        for frame_num, frame in enumerate(video_frames):
            frame_copy = frame.copy() # Draw on a copy of the frame

            # --- Corrected Data Access ---
            # Access the dictionary for the current frame from the list
            if frame_num < len(tracks["players"]):
                player_dict = tracks["players"][frame_num]
            else:
                player_dict = {} # Default to empty dict if frame_num is out of bounds
                print(f"Warning: Frame number {frame_num} out of bounds for player tracks.")

            if frame_num < len(tracks["ball"]):
                ball_dict = tracks["ball"][frame_num] # Interpolated/detected ball data
            else:
                ball_dict = {}
                print(f"Warning: Frame number {frame_num} out of bounds for ball tracks.")

            if frame_num < len(tracks["referees"]):
                referee_dict = tracks["referees"][frame_num]
            else:
                referee_dict = {}
                print(f"Warning: Frame number {frame_num} out of bounds for referee tracks.")
            # --- End Corrected Data Access ---

            # Determine player with possession ONCE per frame
            possessing_player_id = self.get_ball_possessor(player_dict, ball_dict)

            # --- Draw Players ---
            for track_id, player in player_dict.items():
                player_bbox = player.get('bbox')
                if player_bbox:
                    # Use assigned team color, default to blue if missing
                    team_color = player.get("team_color", (255, 0, 0)) # BGR Blue default
                    team_color_bgr = tuple(map(int, team_color))
                    frame_copy = self.draw_ellipse(frame_copy, player_bbox, team_color_bgr, track_id)

                    # Draw possession indicator (e.g., triangle) if this player has the ball
                    if track_id == possessing_player_id:
                        indicator_color = (0, 255, 255) # Yellow indicator (BGR)
                        frame_copy = self.draw_triangle(frame_copy, player_bbox, indicator_color)

            # --- Draw Referees ---
            for track_id, referee in referee_dict.items():
                 ref_bbox = referee.get('bbox')
                 if ref_bbox:
                     ref_color = (0, 165, 255) # Orange/Amber (BGR) for referee
                     # Draw ellipse without ID for referee
                     frame_copy = self.draw_ellipse(frame_copy, ref_bbox, ref_color, None)

            # --- Draw Ball ---
            ball_info = ball_dict.get(1, {}) # Assume ball ID is 1
            ball_bbox = ball_info.get('bbox')
            if ball_bbox:
                 ball_color = (30, 220, 255) # Lighter Yellow/Gold for ball (BGR)
                 # Draw a triangle or other indicator on the ball
                 frame_copy = self.draw_triangle(frame_copy, ball_bbox, ball_color)

            # --- Draw Team Possession Time Display ---
            # This function also needs the tracks dict for the current frame
            frame_tracks = {
                "players": player_dict,
                "ball": ball_dict,
                "referees": referee_dict
                # Add other tracked items if needed by draw_team_ball_control
            }
            # Pass the tracks *for the current frame* to the drawing function
            frame_copy = self.draw_team_ball_control(frame_copy, frame_num, frame_tracks)


            # Add the annotated frame to the output list
            output_video_frames.append(frame_copy)

            # Optional: Progress indicator
            # if (frame_num + 1) % 100 == 0 or frame_num == num_frames - 1:
            #      print(f"  Annotated frame {frame_num + 1}/{num_frames}")

        print("Finished drawing annotations.")
        return output_video_frames

    def draw_team_ball_control(self, frame, frame_num, frame_tracks):
         """
         Draws the team possession time display (including percentage) on the frame
         and updates the possession state using PossessionHandler.
         Takes tracks *for the current frame*.
         """
         frame_height, frame_width, _ = frame.shape

         # Use the tracks passed for *this specific frame*
         players_in_frame = frame_tracks.get("players", {})
         ball_dict_in_frame = frame_tracks.get("ball", {})

         # 1. Determine who has possession in this frame
         possessing_player_id = self.get_ball_possessor(players_in_frame, ball_dict_in_frame)

         # 2. Determine the team ID of the possessing player
         team_id = 0 # Default to no possession (Team 0)
         if possessing_player_id != -1:
             player_info = players_in_frame.get(possessing_player_id, {})
             team_id = player_info.get("team", 0)

         # 3. Update the possession handler state
         self.possession_handler.update_possession(frame_num, team_id)

         # 4. Get the calculated possession times from the handler
         team_1_time, team_2_time = self.possession_handler.get_possession_time()

         # 5. Calculate display values (using value * 10 as per previous logic)
         display_value_1 = int(team_1_time * 10)
         display_value_2 = int(team_2_time * 10)

         # --- Calculate Percentages ---
         total_display_value = display_value_1 + display_value_2
         if total_display_value > 0:
             # Calculate percentage, use float division
             display_value_1_porcent = (display_value_1 / total_display_value) * 100.0
             # Ensure percentages sum to 100 (or handle potential floating point inaccuracies)
             display_value_2_porcent = 100.0 - display_value_1_porcent
         else:
             # Avoid division by zero if total is 0
             display_value_1_porcent = 0.0
             display_value_2_porcent = 0.0
         # --- End Percentage Calculation ---

         # 6. Define text drawing parameters
         text_x = int(frame_width * 0.78) # Adjusted start position slightly left
         text_y_base = int(frame_height * 0.05)
         line_height = int(frame_height * 0.05)
         font = cv2.FONT_HERSHEY_SIMPLEX
         font_scale = 0.7
         font_thickness = 2
         team_1_color_bgr = (128, 128, 128) # Default Gray
         team_2_color_bgr = (0, 255, 0)     # Default Green
         text_padding = 5

         # Attempt to get actual assigned team colors
         found_team1_color, found_team2_color = False, False
         for player_info in players_in_frame.values():
             p_team = player_info.get("team")
             p_color = player_info.get("team_color")
             if p_team == 1 and p_color is not None and not found_team1_color:
                 team_1_color_bgr = tuple(map(int, p_color))
                 found_team1_color = True
             elif p_team == 2 and p_color is not None and not found_team2_color:
                 team_2_color_bgr = tuple(map(int, p_color))
                 found_team2_color = True
             if found_team1_color and found_team2_color: break

         # 7. Format display text including percentage (formatted to 0 decimal places)
         team_1_text = f"Team White:  {display_value_1_porcent:.0f}%"
         team_2_text = f"Team Green: {display_value_2_porcent:.0f}%"

         # Draw text with background for readability
         for i, (text, color) in enumerate([(team_1_text, team_1_color_bgr), (team_2_text, team_2_color_bgr)]):
             text_y = text_y_base + i * line_height
             (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

             rect_x1 = text_x
             rect_y1 = text_y - text_h - baseline - text_padding
             rect_x2 = text_x + text_w + (2 * text_padding)
             rect_y2 = text_y + baseline + text_padding

             if rect_y1 < 0 or rect_y2 > frame_height or rect_x1 < 0 or rect_x2 > frame_width:
                  print(f"Warning: Text/Background for possession display is out of frame bounds at frame {frame_num}")
                  continue

             try: # Draw background (semi-transparent black)
                 sub_img = frame[rect_y1:rect_y2, rect_x1:rect_x2]
                 black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                 alpha = 0.6
                 res = cv2.addWeighted(sub_img, 1 - alpha, black_rect, alpha, 1.0)
                 frame[rect_y1:rect_y2, rect_x1:rect_x2] = res
             except ValueError as ve:
                  print(f"Warning: Error creating text background at frame {frame_num}. Rect: {[rect_x1, rect_y1, rect_x2, rect_y2]}. Error: {ve}")

             # Draw text
             cv2.putText(frame, text, (text_x + text_padding, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

         return frame