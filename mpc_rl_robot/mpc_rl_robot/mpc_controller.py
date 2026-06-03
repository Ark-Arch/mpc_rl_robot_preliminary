#!/usr/bin/env python3
"""
MPC Low-Level Controller for Differential Drive Robot
PhD Preliminary Results — MPC-RL Hierarchical Navigation
"""

import casadi as ca
import numpy as np

class MPCController:
    def __init__(self):
        # --- Robot Physical Constraints ---
        self.v_max = 0.22      # Max linear velocity (m/s) - TurtleBot3 limit
        self.v_min = -0.22     # Min linear velocity (m/s)
        self.w_max = 2.84      # Max angular velocity (rad/s) - TurtleBot3 limit
        self.w_min = -2.84     # Min angular velocity (rad/s)

        # --- MPC Horizon ---
        self.N = 15            # Prediction horizon steps
        self.dt = 0.1          # Time step (seconds) → 10 Hz

        # --- Cost Weights ---
        self.Q = np.diag([2.0, 2.0, 0.5])   # State cost [x, y, theta]
        self.R = np.diag([0.1, 0.05])         # Control cost [v, omega]

        # --- Build the MPC Solver ---
        self.solver, self.args = self._build_solver()
        print("[MPC] Controller initialized successfully.")

    def _build_solver(self):
        """Build the CasADi NLP solver for MPC."""

        # State and control variables
        x  = ca.SX.sym('x')
        y  = ca.SX.sym('y')
        th = ca.SX.sym('theta')
        states = ca.vertcat(x, y, th)
        n_states = states.size1()

        v = ca.SX.sym('v')
        w = ca.SX.sym('omega')
        controls = ca.vertcat(v, w)
        n_controls = controls.size1()

        # Differential drive kinematic model
        rhs = ca.vertcat(
            v * ca.cos(th),
            v * ca.sin(th),
            w
        )
        f = ca.Function('f', [states, controls], [rhs])

        # Decision variables
        X = ca.SX.sym('X', n_states,  self.N + 1)  # States over horizon
        U = ca.SX.sym('U', n_controls, self.N)      # Controls over horizon
        P = ca.SX.sym('P', n_states + n_states)     # [initial state, goal state]

        # Cost function and constraints
        obj = 0
        g   = []

        # Initial condition constraint
        g.append(X[:, 0] - P[:n_states])

        Q = ca.DM(self.Q)
        R = ca.DM(self.R)

        for k in range(self.N):
            state   = X[:, k]
            control = U[:, k]
            obj += ca.mtimes([(state - P[n_states:]).T, Q, (state - P[n_states:])])
            obj += ca.mtimes([control.T, R, control])

            # Runge-Kutta 4 integration
            k1 = f(state, control)
            k2 = f(state + self.dt/2 * k1, control)
            k3 = f(state + self.dt/2 * k2, control)
            k4 = f(state + self.dt * k3, control)
            next_state = state + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            g.append(X[:, k+1] - next_state)

        # Flatten decision variables
        opt_vars = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(U, -1, 1)
        )

        nlp = {'f': obj, 'x': opt_vars, 'g': ca.vertcat(*g), 'p': P}
        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-6,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Bounds
        n_vars = n_states * (self.N+1) + n_controls * self.N
        lbx = [-ca.inf] * n_states * (self.N+1) + \
              [self.v_min, self.w_min] * self.N
        ubx = [ ca.inf] * n_states * (self.N+1) + \
              [self.v_max, self.w_max] * self.N
        lbg = [0.0] * (n_states * (self.N+1))
        ubg = [0.0] * (n_states * (self.N+1))

        args = {
            'lbx': lbx, 'ubx': ubx,
            'lbg': lbg, 'ubg': ubg
        }

        return solver, args

    def compute_control(self, current_state, goal_state):
        """
        Compute optimal control given current and goal state.
        Returns (v, omega) velocity commands.
        """
        n_states  = 3
        n_controls = 2

        p = np.concatenate([current_state, goal_state])

        x0 = np.zeros((n_states * (self.N+1) + n_controls * self.N, 1))

        self.args['p']  = p
        self.args['x0'] = x0

        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )

        u = sol['x'][n_states * (self.N+1):]
        v     = float(u[0])
        omega = float(u[1])

        return v, omega


# --- Quick Test ---
if __name__ == '__main__':
    mpc = MPCController()

    current = np.array([0.0, 0.0, 0.0])   # Robot at origin
    goal    = np.array([2.0, 0.0, 0.0])   # Goal 2 meters ahead

    v, w = mpc.compute_control(current, goal)
    print(f"[MPC] Computed control: v={v:.4f} m/s, omega={w:.4f} rad/s")
