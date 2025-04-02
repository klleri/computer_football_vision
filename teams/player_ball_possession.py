# computer_football_vision/teams/player_ball_possession.py
# *** NOTE: This entire class appears to be UNUSED. ***
# The functionality is handled by `get_ball_possessor` in `objects/objects.py`.

import sys
sys.path.append('../') # Add parent directory to path
# Attempt to import from videos, might fail if run standalone
try:
    from videos import bounding_box_center, measure_distance
except ImportError:
    print("Warning: Could not import from 'videos' module. Ensure PYTHONPATH is set correctly.")
    # Define dummy functions if import fails to avoid crashing immediately
    def bounding_box_center(bbox): return (0,0)
    def measure_distance(p1,p2): return 99999


class PlayerBallPossession():
    """
    Determines which player is closest to the ball based on foot position distance.
    *** NOTE: This class appears to be UNUSED and potentially outdated. ***
    """
    def __init__(self):
        """Initializes the checker with a maximum distance threshold."""
        self.max_player_ball_distance = 70 # Max distance in pixels

    def assign_ball_to_player(self, players: dict, ball_bbox: list) -> int:
        """
        Finds the player ID closest to the ball's center, below a threshold distance,
        considering the distance from the ball to the player's feet (approximated).

        Args:
            players (dict): Dictionary of player tracks for the current frame.
                            Format: {player_id: {'bbox': [x1, y1, x2, y2]}, ...}
            ball_bbox (list): Bounding box of the ball [x1, y1, x2, y2].

        Returns:
            int: The ID of the player assigned possession, or -1 if no player is close enough.
        """
        if not ball_bbox: # Check if ball_bbox is valid
             return -1

        try:
            ball_position = bounding_box_center(ball_bbox)
        except Exception as e:
             print(f"Error calculating ball center in PlayerBallPossession: {e}")
             return -1

        minimum_distance = float('inf') # Use infinity for initial min distance
        assigned_player = -1

        for player_id, player_data in players.items():
            player_bbox = player_data.get('bbox')
            if not player_bbox: # Skip if player has no bbox
                continue

            try:
                # Calculate distance from ball center to estimated left/right foot positions
                # Foot position is approximated as bottom-left and bottom-right corners
                distance_left = measure_distance((player_bbox[0], player_bbox[3]), ball_position) # x1, y2
                distance_right = measure_distance((player_bbox[2], player_bbox[3]), ball_position) # x2, y2
                # Use the minimum of the two distances
                distance = min(distance_left, distance_right)

                # Check if this player is closer than the current minimum AND within the threshold
                if distance < self.max_player_ball_distance:
                    if distance < minimum_distance:
                        minimum_distance = distance
                        assigned_player = player_id
            except Exception as e:
                 print(f"Error calculating distance for player {player_id} in PlayerBallPossession: {e}")
                 continue # Skip player on error

        return assigned_player