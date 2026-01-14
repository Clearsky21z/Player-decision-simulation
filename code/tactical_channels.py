import numpy as np

# Constants: 120x80 Pitch mapped to 104x64 Grid
PITCH_X, PITCH_Y = 120.0, 80.0
GRID_W, GRID_H = 104, 64
MAX_SPEED = 13.0          # m/s
REACTION_TIME = 0.7       # s
LAMBDA = 4.0              # Pitch control scaling
SIGMA_PRESSURE = 3.0      # Pressure radius
LANE_SIGMA = 1.5          # Lane width
INTERCEPT_DIST = 2.0      # Defender reach

def calculate_pitch_control_channel(teammates_locs, teammates_vels,
                                    opponents_locs, opponents_vels):
    """
    Calculates Pitch Control (C14) based on Time-to-Intercept.

    Args:
        teammates_locs (list[tuple]): Teammate (x, y) in Grid Units.
        teammates_vels (list[tuple]): Teammate (vx, vy).
        opponents_locs (list[tuple]): Opponent (x, y) in Grid Units.
        opponents_vels (list[tuple]): Opponent (vx, vy).

    Returns:
        np.ndarray: (GRID_H, GRID_W) array with values [0, 1] (1=Teammate control).
    """
    y_grid, x_grid = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing='ij')

    def get_arrival_times(locs, vels):
        if not locs:
            return np.full((GRID_H, GRID_W), 999.0)

        locs_arr = np.array(locs)[:, np.newaxis, np.newaxis, :]
        dx = x_grid - locs_arr[..., 0]
        dy = y_grid - locs_arr[..., 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Time = Distance / Speed + Reaction
        t_arrival = dist / (MAX_SPEED * (GRID_W / PITCH_X)) + REACTION_TIME
        return np.min(t_arrival, axis=0)

    t_att = get_arrival_times(teammates_locs, teammates_vels)
    t_def = get_arrival_times(opponents_locs, opponents_vels)

    return 1 / (1 + np.exp(LAMBDA * (t_att - t_def)))


def calculate_defensive_pressure_channel(opponents_locs):
    """
    Calculates Defensive Pressure (C15) using Gaussian influence.

    Args:
        opponents_locs (list[tuple]): Opponent (x, y) in Grid Units.

    Returns:
        np.ndarray: (GRID_H, GRID_W) array of pressure intensity.
    """
    if not opponents_locs:
        return np.zeros((GRID_H, GRID_W), dtype=np.float32)

    y_grid, x_grid = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing='ij')
    opps = np.array(opponents_locs)[:, np.newaxis, np.newaxis, :]

    dist_sq = (x_grid - opps[..., 0]) ** 2 + (y_grid - opps[..., 1]) ** 2
    pressure_fields = np.exp(-dist_sq / (2 * SIGMA_PRESSURE ** 2))

    return np.max(pressure_fields, axis=0)


def calculate_passing_lane_channel(actor_loc, teammates_locs, opponents_locs):
    """
    Calculates Passing Availability (C16) considering Defensive Shadows.

    Args:
        actor_loc (tuple): Ball carrier (x, y) in METRIC coordinates.
        teammates_locs (list[tuple]): Receivers (x, y) in GRID coordinates.
        opponents_locs (list[tuple]): Opponents (x, y) in GRID coordinates.

    Returns:
        np.ndarray: (GRID_H, GRID_W) array representing open passing lanes.
    """
    lane_map = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    if not teammates_locs:
        return lane_map

    ax, ay = actor_loc[0] * (GRID_W / PITCH_X), actor_loc[1] * (GRID_H / PITCH_Y)
    y_grid, x_grid = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing='ij')

    for (tx, ty) in teammates_locs:
        vx, vy = tx - ax, ty - ay
        length_sq = vx ** 2 + vy ** 2
        if length_sq == 0: continue

        t = np.clip(((x_grid - ax) * vx + (y_grid - ay) * vy) / length_sq, 0, 1)
        proj_x, proj_y = ax + t * vx, ay + t * vy

        dist_sq = (x_grid - proj_x) ** 2 + (y_grid - proj_y) ** 2
        lane_intensity = np.exp(-dist_sq / (2 * LANE_SIGMA ** 2))

        block_factor = 1.0
        if opponents_locs:
            opps = np.array(opponents_locs)
            t_opp = ((opps[:, 0] - ax) * vx + (opps[:, 1] - ay) * vy) / length_sq
            mask = (t_opp > 0.1) & (t_opp < 0.9)
            if np.any(mask):
                valid_opps = opps[mask]
                t_valid = t_opp[mask][:, np.newaxis]
                p_ox, p_oy = ax + t_valid * vx, ay + t_valid * vy
                d_opp = np.sqrt((valid_opps[:, 0] - p_ox.flatten()) ** 2 +
                                (valid_opps[:, 1] - p_oy.flatten()) ** 2)
                if np.any(d_opp < INTERCEPT_DIST):
                    block_factor = 0.1

        lane_map = np.maximum(lane_map, lane_intensity * block_factor)

    return lane_map


def create_new_channels(freeze_frame, actor_loc, velocity_dict):
    """
    Generates the 3-channel tactical tensor (C14-C16).

    Args:
        freeze_frame (list[dict]): StatsBomb 360 frames with 'location' and 'teammate'.
        actor_loc (tuple): Ball carrier (x, y) in Metric coordinates.
        velocity_dict (dict): Map of entity_id to (vx, vy).

    Returns:
        np.ndarray: Stacked tensor (3, GRID_H, GRID_W).
    """
    ax_g = np.clip(actor_loc[0] * (GRID_W / PITCH_X), 0, GRID_W - 1)
    ay_g = np.clip(actor_loc[1] * (GRID_H / PITCH_Y), 0, GRID_H - 1)

    team_locs, team_vels = [(ax_g, ay_g)], [velocity_dict.get(None, (0, 0))]
    opp_locs, opp_vels = [], []

    for p in freeze_frame:
        px = np.clip(p['location'][0] * (GRID_W / PITCH_X), 0, GRID_W - 1)
        py = np.clip(p['location'][1] * (GRID_H / PITCH_Y), 0, GRID_H - 1)
        vel = velocity_dict.get(p.get('entity_id'), (0, 0))

        if p['teammate']:
            team_locs.append((px, py)); team_vels.append(vel)
        else:
            opp_locs.append((px, py)); opp_vels.append(vel)

    c14 = calculate_pitch_control_channel(team_locs, team_vels, opp_locs, opp_vels)
    c15 = calculate_defensive_pressure_channel(opp_locs)
    c16 = calculate_passing_lane_channel(actor_loc, team_locs[1:], opp_locs)

    return np.stack([c14, c15, c16], axis=0)