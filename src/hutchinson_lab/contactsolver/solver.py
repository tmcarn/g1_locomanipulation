import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

''' Physical parameters '''
# Environment Parameters
g = 9.81    # gravity

# Box Parameters
m_b = 250.0  # box mass
l = 0.5     # box half-length
mu_bg = 0.1  # box-ground friction
p_b = np.array([-0.5, 0.5]) # CoM of box
dp_b = np.array([0.0, -1.0]) # Box velocity

# Robot Parameters
m_r = 40.0                   # Robot mass
p_c = np.array([0.5, 0.75])   # Robot CoM
p_f = np.array([[0.25, 0.0], # Foot positions
               [0.75, 0.0]])      
mu_fg = 0.9 # foot-ground friction

# Robot-Box Contact Parameters
mu_hb = 1.0 # hand-box friction


import cvxpy as cp

# def stage1_qp(h_y, W_req=-300):
#     Fh_x, Fh_y, f_x = cp.Variable(), cp.Variable(), cp.Variable()
#     F_g = -m_b * g
#     F_N = -(F_g + Fh_y)                     # fixed: sign

#     r = np.array([l, h_y - p_b[1]]) # lever arm from box CoM to contact point

#     cons = [
#         Fh_x + f_x == W_req,                 # f_x is free, not mu_bg*F_N
#         r[1]*Fh_x - r[0]*Fh_y >= 0,              # fixed: sign, matches right-side contact
#         F_N >= 0,
#         Fh_x <= 0,
#         f_x  <=  mu_bg*F_N - 0,  -f_x  <= mu_bg*F_N - 0,
#         Fh_y <= -mu_hb*Fh_x,     -Fh_y <= -mu_hb*Fh_x,
#     ]
#     prob = cp.Problem(cp.Minimize(Fh_x**2 + Fh_y**2 + f_x**2), cons)
#     prob.solve(solver=cp.CLARABEL)
#     print(prob.status)
#     return Fh_x.value, Fh_y.value, prob.status == cp.OPTIMAL

def stage1_qp(h_y, W_req=-100):
    Fh_x, Fh_y, f_x = cp.Variable(), cp.Variable(), cp.Variable()
    F_g = -m_b * g
    F_N = -(F_g + Fh_y)

    cons = [
        Fh_x + f_x == W_req,
        h_y*Fh_x - l*Fh_y == p_b[1]*W_req,
        F_N >= 0,
        Fh_x <= 0,
        f_x >= 0,  f_x <= mu_bg*F_N,          # was: -mu_bg*F_N <= f_x <= mu_bg*F_N
        Fh_y <= -mu_hb*Fh_x, -Fh_y <= -mu_hb*Fh_x,
    ]
    prob = cp.Problem(cp.Minimize(Fh_x**2 + Fh_y**2 + f_x**2), cons)
    prob.solve(solver=cp.CLARABEL)
    return Fh_x.value, Fh_y.value, prob.status == cp.OPTIMAL


def visualize(z_h, foot_forces=None, ax=None):
    """
    foot_forces: optional dict {0: (Fx,Fz), 1: (Fx,Fz)} for stage-2 results,
                 indexed matching rows of p_f.
    force_scale: meters per Newton, for arrow length. Tune for your force magnitudes.
    """
    p_h = np.array([p_b[0]+l, z_h])
    # F_h, F_N, feasible = stage1(p_h)

    # print(p_h)

    Fhx, Fhy, feasible = stage1_qp(z_h)

    if not feasible:
        return None
    F_g = -m_b * g
    F_N = -(F_g + Fhy)

    F_h = np.array([Fhx, Fhy])

    
 
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))
 
    # ground
    ax.axhline(0, color='k', linewidth=1)
 
    # box
    box = patches.Rectangle((p_b[0]-l, 0), 2*l, 2*l,
                             linewidth=1.5, edgecolor='saddlebrown',
                             facecolor='burlywood', alpha=0.6)
    ax.add_patch(box)
    ax.plot(*p_b, 'o', color='saddlebrown', label='box CoM')
 
    # robot: CoM, feet, simple stick links
    ax.plot(*p_c, 'o', color='navy', markersize=10, label='robot CoM')
    for i, foot in enumerate(p_f):
        ax.plot(*foot, 's', color='navy', markersize=8)
        ax.plot([p_c[0], foot[0]], [p_c[1], foot[1]], '-', color='navy', linewidth=1, alpha=0.5)
    ax.plot([p_c[0], p_h[0]], [p_c[1], p_h[1]], '-', color='navy', linewidth=1, alpha=0.5)
 
    # hand contact
    ax.plot(*p_h, 'D', color='crimson', markersize=8, label=f'hand contact (z_h={z_h:.2f})')
 
    # --- force arrows ---
    def arrow(origin, vec, color, label, force_scale=0.5):
        ax.annotate('', xy=origin + np.array(vec)*force_scale, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.plot([], [], color=color, label=label)  # legend proxy
 
    # hand force ON the box (world frame, as computed)
    F_h_norm = F_h / np.linalg.norm(F_h)
    print(F_h_norm)
    arrow(p_h, F_h_norm, 'crimson', f'Fh=({F_h[0]:.0f},{F_h[1]:.0f}) N')
 
    # ground reaction on box (normal + friction), applied at (p_b[0], 0)
    box_ground_pt = np.array([p_b[0], 0.0])
    f_box_x = -F_h[0] + mu_bg * F_N * np.sign(-dp_b[0]) if dp_b[0] != 0 else -F_h[0]

    F_f = np.array([mu_bg*F_N, 0])
    F_f_norm = F_f / np.linalg.norm(F_f)

    F_N = np.array([0, F_N])
    F_N_norm = F_N / np.linalg.norm(F_N)

    

    arrow(box_ground_pt, F_N_norm, 'saddlebrown', f'N_box={F_N[1]:.0f} N')
    arrow(box_ground_pt, F_f_norm, 'peru', 'friction (box)')
 
    # foot forces, if stage 2 has been solved
    if foot_forces is not None:
        colors = ['teal', 'darkgreen']
        for i, F in foot_forces.items():
            arrow(p_f[i], F, colors[i % 2], f'foot{i} F=({F[0]:.0f},{F[1]:.0f}) N')
 
    status = "FEASIBLE" if feasible else "INFEASIBLE (denom<=0 or N_b<0)"
    ax.set_title(f"z_h={z_h:.2f} m — {status}")
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(-0.3, 2.0)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
 
    if standalone:
        plt.tight_layout()
        plt.savefig('stage1_viz.png', dpi=130)
        print(f"Fh_x={F_h[0]:.2f}, Fh_z={F_h[1]:.2f}, N_b={F_N:.2f}, feasible={feasible}")
    return ax
 
 
if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, z_h in zip(axes, [0.25, 0.5, 0.75]):
        visualize(z_h, ax=ax)
    plt.tight_layout()
    plt.savefig('stage1_sweep.png', dpi=130)
    print("saved sweep")
 

