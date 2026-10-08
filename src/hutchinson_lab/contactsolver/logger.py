import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WBCLogger:
    """Records reference vs. achieved values across WBC solves, plots after N ticks.

    Usage:
        logger = WBCLogger()
        ...
        tau = wbc.solve(q, dq, p_com_ref, box_ref, f_ref, logger=logger)
        if logger.full():
            logger.plot()
    """
    CONTACTS = ['right_hand', 'right_foot']

    def __init__(self, n_ticks=20):
        self.n_ticks = n_ticks
        self.t = []

        self.f_ref = {name: [] for name in self.CONTACTS}
        self.f_actual = {name: [] for name in self.CONTACTS}

        self.tau = []
        self.qdd_norm = []

        self.pdd_com_des = []
        self.pdd_com_actual = []  # Jcom @ qdd, filled in by record()

    def record(self, tick, f_des, fc_sol, tau_sol, qdd_sol, pdd_com_des, Jcom):
        self.t.append(tick)
        for name in self.CONTACTS:
            self.f_ref[name].append(np.array(f_des[name]).flatten())
            self.f_actual[name].append(np.array(fc_sol[name]).flatten())

        self.tau.append(np.array(tau_sol).flatten())
        self.qdd_norm.append(float(np.linalg.norm(qdd_sol)))

        self.pdd_com_des.append(np.array(pdd_com_des).flatten())
        self.pdd_com_actual.append(np.array(Jcom @ qdd_sol).flatten())

    def full(self):
        return len(self.t) >= self.n_ticks

    def plot(self, save_path="wbc_log.png"):
        t = np.array(self.t)

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # --- hand forces: ref vs actual, x and z ---
        ax = axes[0, 0]
        for name, style in [('right_hand', '-')]:
            ref = np.array(self.f_ref[name])
            act = np.array(self.f_actual[name])
            ax.plot(t, ref[:, 0], style, color='tab:blue', label=f"{name} fx ref")
            ax.plot(t, act[:, 0], style, color='tab:orange', label=f"{name} fx actual")
        ax.set_title("Hand fx: ref vs actual")
        ax.legend(fontsize=7)

        ax = axes[0, 1]
        for name, style in [('right_hand', '-')]:
            ref = np.array(self.f_ref[name])
            act = np.array(self.f_actual[name])
            ax.plot(t, ref[:, 1], style, color='tab:blue', label=f"{name} fz ref")
            ax.plot(t, act[:, 1], style, color='tab:orange', label=f"{name} fz actual")
        ax.set_title("Hand fz: ref vs actual")
        ax.legend(fontsize=7)

        # --- foot forces: ref vs actual, x and z ---
        ax = axes[1, 0]
        for name, style in [('right_foot', '-')]:
            ref = np.array(self.f_ref[name])
            act = np.array(self.f_actual[name])
            ax.plot(t, ref[:, 0], style, color='tab:blue', label=f"{name} fx ref")
            ax.plot(t, act[:, 0], style, color='tab:orange', label=f"{name} fx actual")
        ax.set_title("Foot fx: ref vs actual")
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        for name, style in [('right_foot', '-')]:
            ref = np.array(self.f_ref[name])
            act = np.array(self.f_actual[name])
            ax.plot(t, ref[:, 1], style, color='tab:blue', label=f"{name} fz ref")
            ax.plot(t, act[:, 1], style, color='tab:orange', label=f"{name} fz actual")
        ax.set_title("Foot fz: ref vs actual")
        ax.legend(fontsize=7)

        # --- CoM acceleration task: desired vs achieved (Jcom @ qdd) ---
        ax = axes[2, 0]
        des = np.array(self.pdd_com_des)
        act = np.array(self.pdd_com_actual)
        ax.plot(t, des[:, 0], label="pdd_com_des x")
        ax.plot(t, act[:, 0], label="Jcom@qdd x")
        ax.plot(t, des[:, 1], label="pdd_com_des z")
        ax.plot(t, act[:, 1], label="Jcom@qdd z")
        ax.set_title("CoM accel task: desired vs achieved")
        ax.legend(fontsize=7)

        # --- tau and qdd norm ---
        ax = axes[2, 1]
        tau = np.array(self.tau)
        for j in range(tau.shape[1]):
            ax.plot(t, tau[:, j], alpha=0.6)
        ax.set_title("Joint torques (tau)")
        ax2 = ax.twinx()
        ax2.plot(t, self.qdd_norm, color='k', linestyle='--', label="||qdd||")
        ax2.set_ylabel("||qdd||")
        ax2.legend(fontsize=7, loc='upper right')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel("tick")
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)