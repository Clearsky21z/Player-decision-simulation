import numpy as np

# Constants based on tactical theory and pitch dimensions
PITCH_X, PITCH_Y = 120.0, 80.0
GRID_W, GRID_H = 104, 64
MAX_SPEED = 13.0  # Max sprinting speed (m/s)
REACTION_TIME = 0.7  # Reaction latency (s)
LAMBDA = 4.0  # Logistic scaling factor for control
SIGMA_PRESSURE = 3.0  # Influence radius for defensive pressure
LANE_SIGMA = 1.5  # Passing lane width scaling
INTERCEPT_DIST = 2.0  # Defender reach for interceptions


def calculate_pitch_control_channel(teammates_locs, teammates_vels,
                                    opponents_locs, opponents_vels):
    """
    Vectorized calculation of spatial ownership (Pitch Control).
    Uses time-to-intercept to determine team dominance at each grid point.
    """
    y_grid, x_grid = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing='ij')

    def get_arrival_times(locs, vels):
        if not locs:
            return np.full((GRID_H, GRID_W), 999.0)

        # Convert list to array for broadcasting: (N_players, 1, 1, 2)
        locs_arr = np.array(locs)[:, np.newaxis, np.newaxis, :]

        # Calculate distance from all players to all grid points
        dx = x_grid - locs_arr[..., 0]
        dy = y_grid - locs_arr[..., 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Physical model: Time = Distance/Speed + Reaction Latency
        # Scaling factor (104/120) accounts for grid vs metric pitch size
        t_arrival = dist / (MAX_SPEED * (GRID_W / PITCH_X)) + REACTION_TIME
        return np.min(t_arrival, axis=0)

    t_att = get_arrival_times(teammates_locs, teammates_vels)
    t_def = get_arrival_times(opponents_locs, opponents_vels)

    # Logistic function to map time difference to [0, 1] probability
    return 1 / (1 + np.exp(LAMBDA * (t_att - t_def)))


def calculate_defensive_pressure_channel(opponents_locs):
    """
    Vectorized defensive influence field using Gaussian kernels.
    Identifies high-stress zones based on opponent proximity.
    """
    if not opponents_locs:
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    y_grid, x_grid = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing='ij')
    opps = np.array(opponents_locs)[:, np.newaxis, np.newaxis, :]

    # Calculate Gaussian influence for all opponents simultaneously
    dist_sq = (x_grid - opps[..., 0]) ** 2 + (y_grid - opps[..., 1]) ** 2
    pressure_fields = np.exp(-dist_sq / (2 * SIGMA_PRESSURE ** 2))

    # Take max pressure across all opponents for each cell
    return np.max(pressure_fields, axis=0)


def calculate_passing_lane_channel(actor_loc, teammates_locs, opponents_locs):
    """
    Models 'Defensive Shadows' by calculating the obstruction factor
    of passing lanes to all available teammates.
    """
    lane_map = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    if not teammates_locs:
        return lane_map

    # Origin point (Ball Carrier)
    ax, ay = actor_loc[0] * (GRID_W / PITCH_X), actor_loc[1] * (GRID_H / PITCH_Y)
    y_grid, x_grid = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing='ij')

    for (tx, ty) in teammates_locs:
        # Define passing vector from actor to teammate
        vx, vy = tx - ax, ty - ay
        length_sq = vx ** 2 + vy ** 2
        if length_sq == 0: continue

        # Project grid points onto the passing segment
        t = np.clip(((x_grid - ax) * vx + (y_grid - ay) * vy) / length_sq, 0, 1)
        proj_x, proj_y = ax + t * vx, ay + t * vy

        # Base lane intensity using Gaussian decay from the segment line
        dist_sq = (x_grid - proj_x) ** 2 + (y_grid - proj_y) ** 2
        lane_intensity = np.exp(-dist_sq / (2 * LANE_SIGMA ** 2))

        # Check for opponent obstructions (Defensive Shadows)
        block_factor = 1.0
        if opponents_locs:
            opps = np.array(opponents_locs)
            # Find projection of opponents onto the specific passing lane
            t_opp = ((opps[:, 0] - ax) * vx + (opps[:, 1] - ay) * vy) / length_sq
            # Only consider opponents positioned between the two players
            mask = (t_opp > 0.1) & (t_opp < 0.9)
            if np.any(mask):
                valid_opps = opps[mask]
                t_valid = t_opp[mask][:, np.newaxis]
                p_ox, p_oy = ax + t_valid * vx, ay + t_valid * vy
                d_opp = np.sqrt((valid_opps[:, 0] - p_ox.flatten()) ** 2 + (valid_opps[:, 1] - p_oy.flatten()) ** 2)
                if np.any(d_opp < INTERCEPT_DIST):
                    block_factor = 0.1  # Lane is effectively obstructed

        lane_map = np.maximum(lane_map, lane_intensity * block_factor)

    return lane_map


def create_new_channels(freeze_frame, actor_loc, velocity_dict):
    """
    Generates a 3-channel tactical tensor (C14-C16) for neural network input[cite: 195].
    """
    # Scale actor to grid
    ax_g = np.clip(actor_loc[0] * (GRID_W / PITCH_X), 0, GRID_W - 1)
    ay_g = np.clip(actor_loc[1] * (GRID_H / PITCH_Y), 0, GRID_H - 1)

    # Separate teams and collect velocities/locations
    team_locs, team_vels = [(ax_g, ay_g)], [velocity_dict.get(None, (0, 0))]
    opp_locs, opp_vels = [], []

    for p in freeze_frame:
        px = np.clip(p['location'][0] * (GRID_W / PITCH_X), 0, GRID_W - 1)
        py = np.clip(p['location'][1] * (GRID_H / PITCH_Y), 0, GRID_H - 1)
        vel = velocity_dict.get(p.get('entity_id'), (0, 0))

        if p['teammate']:
            team_locs.append((px, py));
            team_vels.append(vel)
        else:
            opp_locs.append((px, py));
            opp_vels.append(vel)

    # Construct the final tactical tensor
    c14 = calculate_pitch_control_channel(team_locs, team_vels, opp_locs, opp_vels)
    c15 = calculate_defensive_pressure_channel(opp_locs)
    c16 = calculate_passing_lane_channel(actor_loc, team_locs[1:], opp_locs)

    return np.stack([c14, c15, c16], axis=0)