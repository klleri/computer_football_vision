# computer_football_vision/teams/teams.py

from sklearn.cluster import KMeans
import cv2
import numpy as np

def extract_color_histogram(frame: np.ndarray, bbox: list, mask: np.ndarray = None,
                            n_bins: int = 16, color_space: str = 'HSV') -> np.ndarray | None:
    """
    Extracts a normalized color histogram from a specified region (ROI) within a bounding box.
    Focuses on the upper/middle part of the bounding box (likely the jersey area).

    Args:
        frame (np.ndarray): The image frame (BGR format).
        bbox (list): The bounding box [x1, y1, x2, y2].
        mask (np.ndarray, optional): A mask for the region of interest. Defaults to None. (Currently unused).
        n_bins (int, optional): Number of bins for each histogram channel. Defaults to 16.
        color_space (str, optional): Color space ('HSV', 'Lab', 'YCrCb'). Defaults to 'HSV'.

    Returns:
        np.ndarray or None: Flattened, normalized 3D color histogram, or None if ROI is invalid or extraction fails.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)

        # Basic check for invalid bbox dimensions
        if x1 >= x2 or y1 >= y2:
            # print(f"Warning: Invalid bounding box dimensions: {bbox}") # Optional warning
            return None

        # --- Define Region of Interest (ROI) ---
        # Aim: Exclude head (top ~10%) and legs/shorts (bottom ~40%) to focus on the jersey.
        # Use the middle 50% vertically.
        # Use the central 80% horizontally to avoid edges.
        roi_h = y2 - y1
        roi_w = x2 - x1
        roi_y1 = y1 + int(0.1 * roi_h) # Start below the likely head area
        roi_y2 = y1 + int(0.6 * roi_h) # End before the likely shorts/legs area
        roi_x1 = x1 + int(0.1 * roi_w) # Indent horizontally from left
        roi_x2 = x2 - int(0.1 * roi_w) # Indent horizontally from right

        # Ensure calculated ROI coordinates are within the frame boundaries
        img_h, img_w = frame.shape[:2]
        roi_y1 = max(0, roi_y1)
        roi_y2 = min(img_h, roi_y2)
        roi_x1 = max(0, roi_x1)
        roi_x2 = min(img_w, roi_x2)

        # Check if the calculated ROI itself is valid (has positive width and height)
        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
            # print(f"Warning: Calculated ROI is invalid for bbox {bbox}. ROI: {[roi_x1, roi_y1, roi_x2, roi_y2]}") # Optional
            return None

        # Extract the ROI from the frame
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Check if ROI extraction was successful (sometimes ROI can be empty)
        if roi.size == 0:
            # print(f"Warning: Extracted ROI is empty for bbox {bbox}.") # Optional
            return None

        # --- Convert ROI to the specified color space ---
        hist_channels = [0, 1, 2] # Use all 3 channels
        if color_space == 'HSV':
            roi_transformed = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Hue (0-179), Saturation (0-255), Value (0-255)
            hist_range = [0, 180, 0, 256, 0, 256]
        elif color_space == 'Lab':
            roi_transformed = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
            # L* (0-255), a* (0-255), b* (0-255) - OpenCV ranges differ from standard L*a*b*
            hist_range = [0, 256, 0, 256, 0, 256]
        elif color_space == 'YCrCb':
            roi_transformed = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            # Y (0-255), Cr (0-255), Cb (0-255)
            hist_range = [0, 256, 0, 256, 0, 256]
        else:
            print(f"Error: Unknown color space specified: {color_space}")
            return None # Or raise ValueError("Unknown color space...")

        # --- Apply mask if provided ---
        # Note: The provided mask argument is currently not used in the calculation.
        # If a mask is needed, it should correspond to the `roi` dimensions.
        # Example (if mask was for the ROI):
        # hist_mask = mask[roi_y1:roi_y2, roi_x1:roi_x2] if mask is not None else None
        hist_mask = None # Explicitly set to None as it's unused

        # --- Calculate the 3D color histogram ---
        hist = cv2.calcHist(
            images=[roi_transformed],
            channels=hist_channels,
            mask=hist_mask, # Pass the mask here if using one
            histSize=[n_bins] * 3, # Number of bins for each channel
            ranges=hist_range * 1 # Range for each channel (flattened)
        )

        # --- Normalize the histogram and flatten ---
        # Using MINMAX normalization to scale values between 0 and 1
        hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
        return hist

    except Exception as e:
        print(f"Error during histogram extraction for bbox {bbox}: {e}")
        return None


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray,
                       method: str = 'bhattacharyya') -> float:
    """
    Compares two histograms using a specified OpenCV comparison method.

    Args:
        hist1 (np.ndarray): The first histogram (flattened, normalized).
        hist2 (np.ndarray): The second histogram (flattened, normalized).
        method (str, optional): Comparison method ('bhattacharyya', 'chisqr',
                                'correl', 'intersect'). Defaults to 'bhattacharyya'.

    Returns:
        float: A distance or similarity measure. Lower values are typically better for
               distance metrics (Bhattacharyya, Chi-Square). Returns infinity if
               histograms are invalid or comparison fails.
    """
    # Check if histograms are valid numpy arrays
    if hist1 is None or hist2 is None or not isinstance(hist1, np.ndarray) or not isinstance(hist2, np.ndarray):
        return float('inf') # Return infinity for invalid input

    # Ensure histograms are float32, required by cv2.compareHist
    hist1 = np.float32(hist1)
    hist2 = np.float32(hist2)

    try:
        if method == 'bhattacharyya':
            # Bhattacharyya distance: Lower value indicates more similarity (0 = identical).
            distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        elif method == 'chisqr':
            # Chi-Square distance: Lower value indicates more similarity (0 = identical).
            distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        elif method == 'correl':
            # Correlation: Higher value indicates more similarity (1 = identical).
            # Convert to a distance-like metric (0 is best) by doing 1 - correlation.
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            distance = 1.0 - similarity
        elif method == 'intersect':
            # Intersection: Higher value indicates more similarity. Sum of min(hist1_bin, hist2_bin).
            # Not a true distance metric. Convert to be distance-like (lower is better).
            # Negative intersection makes higher intersection yield a lower (more negative) value.
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
            distance = -similarity # Treat higher intersection as 'closer' -> more negative distance
        else:
            print(f"Error: Unknown histogram comparison method: {method}")
            return float('inf') # Or raise ValueError(...)

        # Handle potential NaN or infinite results from the comparison itself
        if np.isnan(distance) or np.isinf(distance):
            return float('inf')
        return distance

    except Exception as e:
        print(f"Error comparing histograms: {e}")
        return float('inf')


class TeamAssigner:
    """
    Assigns players to one of two teams based on jersey color similarity,
    using K-means clustering on initial frames to establish team color profiles (histograms).
    Also manages the representative BGR colors used for drawing.
    """
    def __init__(self, n_bins: int = 16, color_space: str = 'HSV',
                 initialization_method: str = 'first_frames',
                 num_initial_frames: int = 5,
                 histogram_comparison_method: str = 'bhattacharyya'):
        """
        Initializes the TeamAssigner.

        Args:
            n_bins (int): Number of bins per channel for histograms.
            color_space (str): Color space for histogram extraction ('HSV', 'Lab', 'YCrCb').
            initialization_method (str): Method for initializing team profiles ('first_frames', 'manual').
            num_initial_frames (int): Number of initial frames for 'first_frames' initialization.
            histogram_comparison_method (str): Metric for comparing histograms.
        """
        # --- Configuration ---
        self.n_bins = n_bins
        self.color_space = color_space
        self.initialization_method = initialization_method
        self.num_initial_frames = num_initial_frames
        self.histogram_comparison_method = histogram_comparison_method

        # --- State ---
        # Stores the representative BGR color used for DRAWING each team.
        # These are set/overridden in `assign_team_color`.
        self.team_colors = {
            1: (255, 0, 0), # Default Team 1: Blue (BGR)
            2: (0, 255, 0)  # Default Team 2: Green (BGR)
        }
        # Caches the assigned team ID for each player ID to avoid re-computation per frame.
        self.player_team_cache = {} # Renamed from player_team_dict
        # Stores the representative color histogram for each team (learned during initialization).
        self.team_histograms = {
            1: None, # Learned histogram for Team 1
            2: None  # Learned histogram for Team 2
        }
        # Optional: Could store individual player histograms if needed elsewhere
        # self.player_histograms = {}
        # Optional: Could store history of assignments for debugging/analysis
        # self.team_assignments_history = {}

        print(f"TeamAssigner initialized: bins={n_bins}, space={color_space}, init={initialization_method}({num_initial_frames} frames), compare={histogram_comparison_method}")


    def initialize_teams(self, frames: list, player_detections_per_frame: list):
        """
        Initializes the team reference histograms based on the configured method.
        Called internally by `assign_team_color`.

        Args:
            frames (list): List of video frames (NumPy arrays).
            player_detections_per_frame (list): List of player detection dictionaries,
                                                one dict per frame.
                                                Format: [{player_id: {'bbox': [...]}}, ...]
        """
        print(f"Initializing team profiles using method: {self.initialization_method}")
        if self.initialization_method == 'first_frames':
            self.initialize_teams_first_frames(frames, player_detections_per_frame)
        elif self.initialization_method == 'manual':
            self.initialize_teams_manual() # Call the placeholder manual method
        else:
            print(f"Error: Unknown initialization method: {self.initialization_method}")
            # Or raise ValueError(...)


    def initialize_teams_first_frames(self, frames: list, player_detections_per_frame: list):
        """
        Initializes team reference histograms using K-means clustering on player
        histograms extracted from the first few frames of the video.
        It also attempts to pre-populate the team assignment cache for players
        seen in these initial frames.

        Args:
            frames (list): List of video frames (NumPy arrays).
            player_detections_per_frame (list): List of player detection dictionaries per frame.
        """
        all_player_histograms = [] # List to store all valid histograms extracted
        # Map to link histogram index back to player/frame for initial assignment
        histogram_metadata = [] # Stores tuples: (frame_num, player_id, histogram_index)

        num_frames_to_process = min(self.num_initial_frames, len(frames))
        print(f"Extracting player histograms from the first {num_frames_to_process} frames...")

        # 1. Extract histograms from players in the initial frames
        hist_idx_counter = 0
        for frame_num in range(num_frames_to_process):
            frame = frames[frame_num]
            # Get player detections for this specific frame
            player_detections_in_frame = player_detections_per_frame[frame_num]

            for player_id, player_data in player_detections_in_frame.items():
                bbox = player_data.get("bbox")
                if bbox:
                    player_histogram = self.get_player_histogram(frame, bbox)
                    if player_histogram is not None:
                        # Store the histogram and its metadata
                        all_player_histograms.append(player_histogram)
                        histogram_metadata.append((frame_num, player_id, hist_idx_counter))
                        hist_idx_counter += 1
                    # else: # Optional logging if histogram extraction failed
                        # print(f"Warning: Could not extract histogram for player {player_id} in frame {frame_num}")

        print(f"Extracted {len(all_player_histograms)} valid histograms for initialization.")

        # 2. Cluster histograms using K-means (requires at least 2 histograms)
        if len(all_player_histograms) >= 2:
            print("Clustering histograms using K-means (k=2)...")
            try:
                # Convert list of histograms to a NumPy array for K-means
                histogram_array = np.array(all_player_histograms)

                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # Use n_init=10 for stability
                kmeans.fit(histogram_array)
                cluster_labels = kmeans.labels_ # Cluster assignments (0 or 1) for each histogram

                # 3. Calculate the mean histogram for each cluster (representing each team)
                # Separate histograms based on their assigned cluster label
                team_0_histograms = histogram_array[cluster_labels == 0]
                team_1_histograms = histogram_array[cluster_labels == 1]

                # Calculate the mean histogram for each cluster if histograms exist for it
                mean_hist_0 = np.mean(team_0_histograms, axis=0) if len(team_0_histograms) > 0 else None
                mean_hist_1 = np.mean(team_1_histograms, axis=0) if len(team_1_histograms) > 0 else None

                # Assign cluster 0's mean histogram to Team 1, cluster 1's to Team 2
                # This assignment (0->1, 1->2) is arbitrary, the actual colors determine the team identity later.
                self.team_histograms[1] = mean_hist_0
                self.team_histograms[2] = mean_hist_1

                # Check if initialization was successful (both histograms should be non-None)
                if self.team_histograms[1] is None or self.team_histograms[2] is None:
                     print("Warning: K-means clustering resulted in one or both team reference histograms being None. Team assignment might be unreliable.")
                     # Fallback? Or rely on get_player_team to handle None histograms gracefully.
                else:
                    print("Team reference histograms initialized successfully via K-means.")

                # 4. (Optional but Recommended) Assign initial teams based on clustering results
                # This pre-populates player_team_cache for players seen in early frames.
                print("Pre-populating team cache based on initial clustering...")
                for i, (frame_num, player_id, _) in enumerate(histogram_metadata):
                     # Assign team based on the cluster label (0 -> Team 1, 1 -> Team 2)
                     assigned_team = cluster_labels[i] + 1
                     self.player_team_cache[player_id] = assigned_team
                     # print(f"  Initial assignment: Player {player_id} (Frame {frame_num}) -> Team {assigned_team}") # Verbose log
                print("Initial team cache populated.")

            except Exception as e:
                print(f"Error during K-means clustering or initial assignment: {e}")
                # Fallback: Leave team histograms as None. Subsequent assignments might fail or default.
                self.team_histograms = {1: None, 2: None}

        else:
            print("Warning: Not enough player histograms collected (< 2) in initial frames to perform K-means clustering. Team reference histograms not initialized.")
            # Ensure histograms remain None
            self.team_histograms = {1: None, 2: None}


    def initialize_teams_manual(self):
        """
        Placeholder for manually setting team histograms (e.g., from predefined colors or loading).
        *** NOTE: This method is not implemented. ***
        """
        # TODO: Implement manual initialization if needed.
        # This could involve creating histograms representing specific colors
        # or loading pre-calculated histograms from files.
        print("Warning: Manual team initialization (`initialize_teams_manual`) is not implemented.")
        # Example:
        # self.team_histograms[1] = create_manual_histogram_for_color(...)
        # self.team_histograms[2] = create_manual_histogram_for_color(...)


    def assign_team_color(self, frames: list, player_detections_per_frame: list):
        """
        Public method to trigger team initialization and set the BGR colors
        used for DRAWING each team.

        Args:
            frames (list): List of video frames (NumPy arrays).
            player_detections_per_frame (list): List of player detection dictionaries per frame.
        """
        # 1. Initialize the team reference histograms based on color analysis
        self.initialize_teams(frames, player_detections_per_frame)

        # 2. --- Set Drawing Colors ---
        # Define the BGR colors that will be used visually represent Team 1 and Team 2
        # when drawing ellipses, text, etc. These are INDEPENDENT of the learned
        # histograms but should ideally correspond visually.
        # Overrides the defaults set in __init__.

        # Team 1 will be drawn as Gray
        self.team_colors[1] = (128, 128, 128) # BGR for Gray
        # Team 2 will be drawn as Green
        self.team_colors[2] = (0, 255, 0)      # BGR for Green

        print(f"Drawing colors set: Team 1 -> {self.team_colors[1]}, Team 2 -> {self.team_colors[2]}")


    def get_player_histogram(self, frame: np.ndarray, bbox: list) -> np.ndarray | None:
        """
        Extracts the color histogram for a player based on their bounding box,
        using the instance's configuration (color space, bins).
        Wrapper around the standalone `extract_color_histogram` function.

        Args:
            frame (np.ndarray): The video frame (BGR).
            bbox (list): The player's bounding box [x1, y1, x2, y2].

        Returns:
            np.ndarray or None: The player's flattened, normalized histogram or None if failed.
        """
        # Basic validation of bbox before passing to extraction function
        if not bbox or len(bbox) != 4:
            print(f"Warning: Invalid bbox provided to get_player_histogram: {bbox}")
            return None
        # Call the extraction function with instance settings
        return extract_color_histogram(
            frame,
            bbox,
            n_bins=self.n_bins,
            color_space=self.color_space
            # Mask is not passed here, consistent with extract_color_histogram usage
        )


    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        """
        Determines the team (1 or 2) for a given player in a specific frame.
        It first checks a cache. If the player is not cached, it calculates the
        player's histogram and compares it to the reference team histograms.

        Args:
            frame (np.ndarray): The current video frame (BGR).
            player_bbox (list): The player's bounding box [x1, y1, x2, y2].
            player_id (int): The player's tracking ID.

        Returns:
            int: The assigned team ID (1 or 2). Defaults to 1 if histograms are missing,
                 comparison fails, or the player's histogram cannot be extracted.
        """
        # 1. Check cache first for efficiency
        if player_id in self.player_team_cache:
            return self.player_team_cache[player_id]

        # 2. If not cached, calculate the player's histogram for this frame
        player_histogram = self.get_player_histogram(frame, player_bbox)

        # Handle case where player histogram couldn't be extracted
        if player_histogram is None:
            # Cannot determine team based on color, assign a default (e.g., Team 1)
            # print(f"Warning: Could not get histogram for player {player_id}. Assigning default team 1.") # Optional Warning
            # Cache the default assignment and return it
            self.player_team_cache[player_id] = 1
            return 1

        # 3. Compare player's histogram with the team reference histograms (if available)
        ref_hist_team1 = self.team_histograms.get(1)
        ref_hist_team2 = self.team_histograms.get(2)

        # Handle cases where reference histograms were not successfully initialized
        if ref_hist_team1 is None or ref_hist_team2 is None:
            # Cannot compare if reference profiles are missing. Assign default.
             # print(f"Warning: Team reference histograms not available. Assigning default team 1 to player {player_id}.") # Optional Warning
             assigned_team_id = 1 # Default to Team 1
        else:
            # Compare player histogram to both team reference histograms
            distance_to_team1 = compare_histograms(
                ref_hist_team1, player_histogram, method=self.histogram_comparison_method
            )
            distance_to_team2 = compare_histograms(
                ref_hist_team2, player_histogram, method=self.histogram_comparison_method
            )

            # Assign to the team with the closer histogram (smaller distance metric)
            # If distances are equal, defaults to Team 1 here.
            if distance_to_team1 <= distance_to_team2:
                assigned_team_id = 1
            else:
                assigned_team_id = 2

        # 4. Cache the determined team ID for this player and return it
        self.player_team_cache[player_id] = assigned_team_id
        return assigned_team_id