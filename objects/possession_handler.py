import cv2
from collections import Counter # Use Counter for efficient frequency counting

class PossessionHandler:
    """
    Manages ball possession state and calculates accumulated possession time for two teams.
    It uses a sliding window over recent frames to determine the 'dominant' team,
    smoothing out rapid, momentary changes in ball proximity.
    """
    def __init__(self, fps: float, possession_window_seconds: float = 0.8, dominance_threshold: float = 0.001):
        """
        Initializes the possession handler.

        Args:
            fps (float): Frames per second of the video. Crucial for calculating window size in frames.
            possession_window_seconds (float): Duration (in seconds) of the sliding window used
                                                to determine dominant possession. Needs tuning.
            dominance_threshold (float): Minimum fraction (0.0 to 1.0) of the window frames a team
                                         must be closest to the ball to be considered dominant. Needs tuning.
        """
        # Validate and store FPS
        if fps is None or fps <= 0:
            print("Warning [PossessionHandler]: Invalid or zero FPS provided. Using default FPS = 24.")
            self.fps = 24.0
        else:
            self.fps = float(fps)

        # Calculate window size in number of frames
        self.possession_window_frames = int(possession_window_seconds * self.fps)
        # Ensure window size is at least 1 frame
        if self.possession_window_frames < 1:
             print(f"Warning [PossessionHandler]: Calculated possession window ({self.possession_window_frames} frames) is too small based on FPS. Using minimum of 1 frame.")
             self.possession_window_frames = 1

        # Validate and store dominance threshold
        self.dominance_threshold = max(0.0, min(1.0, dominance_threshold))

        print(f"[PossessionHandler] Initialized: FPS={self.fps:.2f}, Window={self.possession_window_frames} frames ({possession_window_seconds}s), Dominance Threshold={self.dominance_threshold:.2f}")

        # ---- Internal States ----
        # Stores the team ID (0, 1, or 2) deemed closest to the ball for each frame in the window
        self.possession_history = []
        # The team ID (0, 1, or 2) currently considered dominant based on the window
        self.current_dominant_team = 0
        # Stores the total accumulated possession time (in seconds) for Team 1 and Team 2
        self.total_possession_time = {1: 0.0, 2: 0.0} # Excludes time for 'no possession' (team 0)
        # Timestamp (using cv2.getTickCount) when the current dominant possession (1 or 2) started
        self.possession_start_tick = 0


    def _calculate_dominant_possession_in_window(self) -> int:
        """
        Analyzes the possession history window to determine the dominant team.

        Returns:
            int: The ID of the dominant team (1 or 2), or 0 if no team meets the
                 dominance threshold or the history is empty/invalid.
        """
        if not self.possession_history:
            return 0 # No history, no dominant team

        # Count the frequency of each valid team ID (1 or 2) within the window
        # Ignores team ID 0 (no possession) for dominance calculation
        team_counts = Counter(team_id for team_id in self.possession_history if team_id in [1, 2])

        if not team_counts:
            return 0 # No possession by Team 1 or Team 2 found in the current window

        # Find the most frequent team (highest count) among Team 1 and Team 2
        # most_common(1) returns list like [(team_id, count)], e.g., [(1, 15)]
        most_frequent_team, highest_count = team_counts.most_common(1)[0]

        # Check if this team's count meets the dominance threshold relative to window size
        required_count = self.possession_window_frames * self.dominance_threshold
        if highest_count >= required_count:
            return most_frequent_team # This team is considered dominant
        else:
            # If no single team meets the threshold, consider it 'no dominant possession'
            return 0


    def update_possession(self, frame_num: int, closest_team_id_in_frame: int):
        """
        Updates the possession state based on the closest team identified for the current frame.
        Manages the history window and accumulates possession time when the dominant team changes.

        Args:
            frame_num (int): Current frame number (primarily for debugging/logging context).
            closest_team_id_in_frame (int): Team ID (0, 1, or 2) closest to the ball
                                            in this specific frame. 0 indicates no player close enough.
        """
        current_tick = cv2.getTickCount() # Get current timestamp

        # 1. Add the current frame's closest team to the history
        self.possession_history.append(closest_team_id_in_frame)

        # 2. Maintain the sliding window size by removing the oldest entry if needed
        if len(self.possession_history) > self.possession_window_frames:
            self.possession_history.pop(0) # Remove the oldest frame's data

        # 3. Determine the dominant team based on the updated window
        new_dominant_team = self._calculate_dominant_possession_in_window()

        # 4. Check if the dominant team has CHANGED from the previous state
        if self.current_dominant_team != new_dominant_team:
            # --- Possession Change Detected ---

            # 4a. Accumulate time for the PREVIOUS dominant team (if it was Team 1 or 2)
            if self.current_dominant_team != 0 and self.possession_start_tick != 0:
                try:
                    # Calculate time elapsed since the previous possession started
                    elapsed_time_seconds = (current_tick - self.possession_start_tick) / cv2.getTickFrequency()
                    # Ensure time is non-negative (can happen with tick counter rollover, though rare)
                    elapsed_time_seconds = max(0.0, elapsed_time_seconds)
                    # Add the elapsed time to the total for the team that WAS possessing
                    self.total_possession_time[self.current_dominant_team] += elapsed_time_seconds
                    # Debugging log (optional)
                    # print(f"Frame {frame_num}: Possession changed from {self.current_dominant_team} to {new_dominant_team}. Added {elapsed_time_seconds:.2f}s to Team {self.current_dominant_team}.")
                except ZeroDivisionError:
                     print("Error [PossessionHandler]: cv2.getTickFrequency() returned zero during time calculation.")
                except Exception as e:
                     print(f"Error [PossessionHandler]: Failed to calculate elapsed time on possession change: {e}")

            # 4b. Update the current dominant team state
            self.current_dominant_team = new_dominant_team

            # 4c. Reset the start timer for the NEW possession interval
            # If the new dominant team is 1 or 2, record the current tick as the start time
            if self.current_dominant_team != 0:
                self.possession_start_tick = current_tick
                # Debugging log (optional)
                # print(f"Frame {frame_num}: New possession start for Team {self.current_dominant_team} at tick {current_tick}")
            else:
                # If the new state is 'no possession' (0), reset the start tick
                self.possession_start_tick = 0
                # Debugging log (optional)
                # print(f"Frame {frame_num}: Possession ended (now Team 0). Resetting start tick.")


    def get_possession_time(self) -> tuple[float, float]:
        """
        Calculates the total possession time for each team suitable for display.
        This includes the already accumulated time from completed possession intervals
        PLUS the duration of the current, ongoing possession interval (if any).

        Returns:
            tuple[float, float]: (total_time_team_1, total_time_team_2) in seconds.
        """
        current_tick = cv2.getTickCount()
        ongoing_interval_duration = 0.0

        # Calculate the duration of the current possession interval if a team (1 or 2) has dominance
        if self.current_dominant_team != 0 and self.possession_start_tick != 0:
            try:
                ongoing_interval_duration = (current_tick - self.possession_start_tick) / cv2.getTickFrequency()
                # Ensure duration is non-negative
                ongoing_interval_duration = max(0.0, ongoing_interval_duration)
            except ZeroDivisionError:
                 # print("Error [PossessionHandler]: cv2.getTickFrequency() returned zero in get_possession_time.") # Optional log
                 ongoing_interval_duration = 0.0
            except Exception as e:
                 print(f"Error [PossessionHandler]: Failed to calculate current time delta in get_possession_time: {e}")
                 ongoing_interval_duration = 0.0

        # Get the base accumulated times (from completed intervals)
        # Use .get() for safety, defaulting to 0.0 if a key somehow doesn't exist
        display_time_team1 = self.total_possession_time.get(1, 0.0)
        display_time_team2 = self.total_possession_time.get(2, 0.0)

        # Add the duration of the current ongoing interval to the correct team's display time
        if self.current_dominant_team == 1:
            display_time_team1 += ongoing_interval_duration
        elif self.current_dominant_team == 2:
            display_time_team2 += ongoing_interval_duration

        # Return the calculated times for display purposes
        return display_time_team1, display_time_team2


    def reset(self):
        """Resets all internal states of the possession handler."""
        print("[PossessionHandler] Resetting internal states.")
        self.possession_history = []
        self.current_dominant_team = 0
        self.total_possession_time = {1: 0.0, 2: 0.0}
        self.possession_start_tick = 0