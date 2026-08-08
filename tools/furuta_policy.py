"""Run the rig's trained Furuta balance policy inside the simulator.

The policy comes from dat07032002/pendulum_moving_board, where it was trained in
MuJoCo and deployed to an ESP32 that runs it at 200 Hz. Nothing here retrains or
reinterprets it: the observation is assembled exactly as rl/furuta_env_2d.py
assembles it, and the action is turned into torque exactly as rl/furuta_2d.xml
does, so the same weights see the same numbers they were trained on.

Getting any of that wrong is silent -- the pendulum simply falls, and it is easy
to blame the physics rather than a scale factor. So each constant below names
where it comes from.
"""
import numpy as np

# --- observation scaling (rl/furuta_env.py, rl/furuta_env_2d.py) ---
TH_SCALE = 15.0                        # rad/s
PHI_SCALE = 25.0                       # rad/s
BOARD_ANGLE_SCALE = np.deg2rad(15.0)
BOARD_RATE_SCALE = np.deg2rad(80.0)
VEL_EMA = 0.5                          # firmware's velocity filter
CONTROL_DT = 0.005                     # 200 Hz, the rate the policy was trained at

# --- actuator (rl/furuta_2d.xml: motor gear=".0127" ctrlrange="-6 6") ---
V_MAX = 6.0                            # volts
GEAR = 0.0127                          # N.m per volt


class FurutaPolicy:
    """The trained actor, plus the state its observation depends on."""

    def __init__(self, weights_path, act_history=2):
        w = np.load(weights_path)
        self.w0 = w["actor__latent_pi__0__weight"]
        self.b0 = w["actor__latent_pi__0__bias"]
        self.w2 = w["actor__latent_pi__2__weight"]
        self.b2 = w["actor__latent_pi__2__bias"]
        self.wmu = w["actor__mu__weight"]
        self.bmu = w["actor__mu__bias"]
        self.obs_dim = self.w0.shape[1]
        self.act_history = act_history
        if self.obs_dim != 10 + act_history:
            raise ValueError("weights want %d inputs; 10 + %d action history is %d"
                             % (self.obs_dim, act_history, 10 + act_history))
        self.reset()

    def reset(self):
        self.thd_f = 0.0
        self.phid_f = 0.0
        self.prev_action = 0.0
        self.act_hist = [0.0] * self.act_history

    def observe(self, theta_up, theta_dot, phi, phi_dot, roll, pitch, roll_rate, pitch_rate):
        """Build the 12-D vector, in the order rl/furuta_env_2d.py builds it."""
        self.thd_f = VEL_EMA * self.thd_f + (1.0 - VEL_EMA) * theta_dot
        self.phid_f = VEL_EMA * self.phid_f + (1.0 - VEL_EMA) * phi_dot
        return np.array([
            np.cos(theta_up),
            np.sin(theta_up),
            self.thd_f / TH_SCALE,
            np.clip(phi / np.pi, -2.0, 2.0),
            self.phid_f / PHI_SCALE,
            self.prev_action,
            np.clip(roll / BOARD_ANGLE_SCALE, -2.0, 2.0),
            np.clip(pitch / BOARD_ANGLE_SCALE, -2.0, 2.0),
            roll_rate / BOARD_RATE_SCALE,
            pitch_rate / BOARD_RATE_SCALE,
            *self.act_hist,
        ], dtype=np.float32)

    def act(self, obs):
        """Deterministic TQC action: tanh of the mean, as SB3 predicts at eval."""
        h = np.maximum(self.w0 @ obs + self.b0, 0.0)
        h = np.maximum(self.w2 @ h + self.b2, 0.0)
        action = float(np.tanh(self.wmu @ h + self.bmu))
        # history shifts BEFORE prev_action updates, matching furuta_env_2d.step
        if self.act_history:
            self.act_hist = [self.prev_action] + self.act_hist[:-1]
        self.prev_action = action
        return action

    @staticmethod
    def torque(action):
        """Volts through the motor's gear, which is what MuJoCo applied."""
        return float(np.clip(action, -1.0, 1.0)) * V_MAX * GEAR
