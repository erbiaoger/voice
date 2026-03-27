
import numpy as np

# ===================== Core utilities =====================

def arrival_time(t_emit, pos, c=340.0):
    """Compute observed time: t' = t_emit + |pos| / c"""
    R = np.linalg.norm(pos, axis=1)
    return t_emit + R / c

def radial_velocity(pos, vel):
    """v_r = d|pos|/dt = (pos · vel)/|pos|"""
    R = np.linalg.norm(pos, axis=1)
    R = np.where(R == 0, 1e-12, R)
    return np.einsum('ij,ij->i', pos, vel) / R

def observed_frequency(f0, vr, c=340.0):
    """Doppler: f(t') = f0 / (1 + v_r/c)"""
    return f0 / (1.0 + vr / c)

def sort_by_tprime(t_prime, arrs):
    idx = np.argsort(t_prime)
    tps = t_prime[idx]
    outs = [a[idx] for a in arrs]
    return tps, outs

# ===================== 1) Straight-line acceleration/deceleration =====================

def forward_accel(
    f0=25.0, l=1000.0, v0=100.0, a=-3.0, t0=20.0,
    t_min=0.0, t_max=40.0, dt=0.01, c=340.0
):
    """
    Straight-line motion with constant acceleration along x:
        x(t) = v0*(t-t0) + 0.5*a*(t-t0)^2,  y(t) = l
        v(t) = v0 + a*(t-t0)
    Returns: t_prime, f_obs, x(t), y(t)
    """
    t = np.arange(t_min, t_max + dt, dt)
    tau = t - t0
    x = v0 * tau + 0.5 * a * tau**2
    y = np.full_like(x, l)
    vx = v0 + a * tau
    vy = np.zeros_like(vx)

    pos = np.stack([x, y], axis=1)
    vel = np.stack([vx, vy], axis=1)

    vr = radial_velocity(pos, vel)
    f_obs = observed_frequency(f0, vr, c)
    t_prime = arrival_time(t, pos, c)

    t_prime, (f_obs, x, y) = sort_by_tprime(t_prime, [f_obs, x, y])
    return t_prime, f_obs, x, y

# ===================== 2) Turning (two straight lines + circular arc) =====================

def _solve_tangent_x_for_turn(l1, l2, r, theta, ccw=True):
    """
    Find the first straight-line contact x1 (point P1=(x1,l1)) so that:
      - A circular arc of radius r rotates by theta (CCW if theta>0, else CW)
      - The tangent of the final line segment has direction angle theta
      - The signed perpendicular distance of the final line to the origin equals l2
    Geometry:
      - If ccw=True: circle center y = l1 + r; else y = l1 - r
      - Center C = (x1, l1 + s*r), s = +1 (ccw) or -1 (cw)
      - Initial radial angle alpha0 = -pi/2 (ccw) or +pi/2 (cw)
      - Final radial angle alpha = alpha0 + theta
      - End point P2 = C + r * [cos alpha, sin alpha]
      - Final line normal n2 = (-sin theta, cos theta)
      - Constraint: signed distance d = n2 · P2 = l2
    """

    s = +1 if ccw else -1
    alpha0 = -np.pi/2 if ccw else +np.pi/2
    n2 = np.array([-np.sin(theta), np.cos(theta)])

    def signed_distance_from_x1(x1):
        C = np.array([x1, l1 + s * r])
        alpha = alpha0 + theta
        P2 = C + r * np.array([np.cos(alpha), np.sin(alpha)])
        d = float(n2 @ P2)
        return d - l2  # target root

    lo, hi = -20000.0, 20000.0
    flo, fhi = signed_distance_from_x1(lo), signed_distance_from_x1(hi)
    if flo * fhi > 0:
        lo, hi = lo * 10, hi * 10
        flo, fhi = signed_distance_from_x1(lo), signed_distance_from_x1(hi)
        if flo * fhi > 0:
            raise RuntimeError("No geometric solution for given l1,l2,r,theta.")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fmid = signed_distance_from_x1(mid)
        if abs(fmid) < 1e-6:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)

def forward_turn(
    f0=25.0, l1=600.0, l2=-20.0, v0=100.0, r=100.0, theta_deg=173.0, t0=20.0,
    t_pre=10.0, t_arc=None, t_post=10.0, dt=0.02, c=340.0
):
    """
    Turn maneuver modeled by: straight (y=l1, +x) -> arc (radius r, angle theta) -> straight (angle theta, signed distance l2).
    Returns: t_prime, f_obs, x(t), y(t)  (sorted by observed time t')
    """
    theta = np.deg2rad(theta_deg)
    ccw = theta > 0
    x1 = _solve_tangent_x_for_turn(l1, l2, r, theta, ccw=ccw)
    s = +1 if ccw else -1
    alpha0 = -np.pi/2 if ccw else +np.pi/2
    C = np.array([x1, l1 + s * r])

    # Segment 1: pre-turn straight (+x)
    n1 = int(np.ceil(t_pre / dt))
    t1 = np.arange(n1) * dt
    P1 = np.column_stack([x1 - v0 * (t_pre - t1), np.full(n1, l1)])
    V1 = np.column_stack([np.full(n1, v0), np.zeros(n1)])

    # Segment 2: arc
    if t_arc is None:
        t_arc = abs(r * theta) / v0
    n2 = max(2, int(np.ceil(t_arc / dt)))
    t2 = np.linspace(0.0, t_arc, n2)
    alpha = alpha0 + (theta) * (t2 / t_arc)
    P2 = np.column_stack([C[0] + r * np.cos(alpha), C[1] + r * np.sin(alpha)])
    tang = np.column_stack([-np.sin(alpha), np.cos(alpha)]) if ccw else np.column_stack([np.sin(alpha), -np.cos(alpha)])
    V2 = v0 * tang

    # Segment 3: post-turn straight (angle theta)
    d2 = np.array([np.cos(theta), np.sin(theta)])
    n3 = int(np.ceil(t_post / dt))
    t3 = np.arange(n3) * dt
    P3 = P2[-1] + np.outer(t3, v0 * d2)
    V3 = np.tile(v0 * d2, (n3, 1))

    # Concatenate
    pos = np.vstack([P1, P2, P3])
    vel = np.vstack([V1, V2, V3])

    # Emission time with offset
    t_emit = np.concatenate([t1, t_pre + t2, t_pre + t_arc + t3]) + (t0 - t_pre)

    vr = radial_velocity(pos, vel)
    f_obs = observed_frequency(f0, vr, c)
    t_prime = arrival_time(t_emit, pos, c)

    t_prime, (f_obs, pos) = sort_by_tprime(t_prime, [f_obs, pos])
    return t_prime, f_obs, pos[:,0], pos[:,1]
