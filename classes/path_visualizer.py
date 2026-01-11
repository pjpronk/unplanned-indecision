import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


class PathVisualizer:
    """
    Visualizer for path planning - shows obstacles and planned paths in 2D.
    Can display static plots or save to files.
    """

    def __init__(self, obstacles_2d, goal):
        """
        Initialize the visualizer with static elements.

        Args:
            obstacles_2d: List of 2D obstacle dictionaries
            goal: Goal 2D position [x, y]
        """
        self.obstacles = obstacles_2d
        self.goal = np.array(goal)
        self.fig = None
        self.ax = None

    def show(self, path, current_pos=None, title="Path Planning Visualization"):
        """
        Display a 2D visualization of the path and obstacles.

        Args:
            path: List of 2D waypoints representing the planned path
            current_pos: Current 2D position [x, y] (optional)
            title: Plot title
        """
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self._draw_obstacles()
        self._draw_path(path)
        self._draw_positions(current_pos)
        self._setup_plot(title)

        plt.show()

    def show_obstacles_only(self, goals=None, current_pos=None, title="Environment Obstacles"):
        """
        Display only the obstacles without any path.

        Args:
            goals: Optional list of goal positions to show (list of [x, y] arrays or single [x, y])
            current_pos: Current 2D position [x, y] (optional)
            title: Plot title
        """
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self._draw_obstacles()

        # Draw current position if provided
        if current_pos is not None:
            self.ax.plot(
                current_pos[0], current_pos[1], "go", markersize=10, label="Start", zorder=7
            )

        # Draw goal(s) if provided
        if goals is not None:
            # Handle single goal or list of goals
            if isinstance(goals, (list, tuple)) and len(goals) > 0:
                # Check if it's a list of goals or a single goal
                if isinstance(goals[0], (list, tuple, np.ndarray)):
                    # Multiple goals
                    for i, goal in enumerate(goals):
                        self.ax.plot(
                            goal[0],
                            goal[1],
                            "ro",
                            markersize=10,
                            label=f"Goal {i + 1}" if i == 0 else "",
                            zorder=7,
                        )
                else:
                    # Single goal [x, y]
                    self.ax.plot(goals[0], goals[1], "ro", markersize=10, label="Goal", zorder=7)
            elif isinstance(goals, np.ndarray):
                # Single goal as numpy array
                self.ax.plot(goals[0], goals[1], "ro", markersize=10, label="Goal", zorder=7)

        self._setup_plot(title)
        plt.show()

    def save(self, path, filename, current_pos=None, title="Path Planning Visualization"):
        """
        Save a 2D visualization to a file.

        Args:
            path: List of 2D waypoints
            filename: Output filename
            current_pos: Current position [x, y] (optional)
            title: Plot title
        """
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        self._draw_obstacles()
        self._draw_path(path)
        self._draw_positions(current_pos)
        self._setup_plot(title)

        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to {filename}")

    def _draw_obstacles(self):
        """Draw all obstacles on the plot."""
        for obs in self.obstacles:
            pos = obs["position"]

            if obs["type"] == "sphere" or obs["type"] == "cylinder":
                # Simple filled circle
                circle = Circle(pos, obs["radius"], color="gray", alpha=0.8)
                self.ax.add_patch(circle)

            elif obs["type"] == "box":
                # Simple filled rectangle
                rect = Rectangle(
                    (pos[0] - obs["width"] / 2, pos[1] - obs["length"] / 2),
                    obs["width"],
                    obs["length"],
                    color="gray",
                    alpha=0.8,
                )
                self.ax.add_patch(rect)

    def _draw_path(self, path):
        """Draw the planned path."""
        if path and len(path) > 0:
            path_array = np.array(path)
            # Simple blue line for the path
            self.ax.plot(
                path_array[:, 0], path_array[:, 1], "b-", linewidth=2, label="Path", zorder=5
            )
            # Add waypoint dots
            self.ax.plot(path_array[:, 0], path_array[:, 1], "b.", markersize=6, zorder=5)

    def _draw_positions(self, current_pos):
        """Draw current position and goal."""
        # Draw current position as green circle
        if current_pos is not None:
            self.ax.plot(
                current_pos[0], current_pos[1], "go", markersize=10, label="Start", zorder=7
            )

        # Draw goal as red circle
        self.ax.plot(self.goal[0], self.goal[1], "ro", markersize=10, label="Goal", zorder=7)

    def _setup_plot(self, title):
        """Set up plot formatting and labels."""
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title(title)

        self.ax.grid(True, linewidth=0.5, alpha=0.5)
        self.ax.set_axisbelow(True)
        self.ax.axis("equal")

        self.ax.legend(loc="upper right", framealpha=0.9)

        # Auto-scale with padding
        self.ax.autoscale()
        self.ax.margins(0.15)

        # Set background
        self.ax.set_facecolor("white")

        plt.tight_layout()
