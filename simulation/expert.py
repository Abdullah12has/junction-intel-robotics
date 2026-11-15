"""Simulation-only kinematic planning and physical joint-position execution.

IK uses a separate MjData scratch buffer. Execution writes only actuator
targets through env.step; free object states are never set by this controller.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from sorting_env import PLATES


class PlanningError(RuntimeError):
    pass


class PickPlaceExpert:
    grasp_point = np.array([0.008, 0, -0.093])

    def __init__(self, env, on_step=None):
        self.env = env
        self.scratch = mujoco.MjData(env.model)
        self.on_step = on_step
        self.phase = "idle"
        self.trace = []
        self.last_action = env.data.qpos[env.qadr].copy()
        self.last_observation = env.observe() if on_step else None

    def solve(self, position, yaw, grip, seed=None):
        env = self.env
        q0 = self.last_action.copy() if seed is None else np.array(seed, float)
        target_x = np.array([np.cos(yaw), np.sin(yaw), 0])
        # Prefer a vertical gripper; at the edge of reach let it tilt, as the real operator does.
        for vertical in (.7, .1):
            def residual(q):
                self.scratch.qpos[env.qadr] = [*q, grip]
                mujoco.mj_kinematics(env.model, self.scratch)
                b = self.scratch.body('gripper')
                r = b.xmat.reshape(3,3)
                p = b.xpos + r @ self.grasp_point
                return np.r_[(p - position) * 10, (r[:,2] - [0,0,1]) * vertical, (r[:,0] - target_x) * .3]
            result = least_squares(residual, np.clip(q0[:5],env.limits[:5,0]+1e-6,env.limits[:5,1]-1e-6),
                                   bounds=(env.limits[:5,0],env.limits[:5,1]), max_nfev=120,
                                   ftol=1e-7, xtol=1e-7, gtol=1e-7)
            error = np.linalg.norm(residual(result.x)[:3])/10
            if error <= .008:
                return np.r_[result.x, grip]
        raise PlanningError(f"Unreachable waypoint {position}: position residual {error:.4f} m")

    def tick(self, action):
        env = self.env
        before = self.last_observation
        obs, reward, done, truncated, info = env.step(action)
        self.last_observation = obs
        self.last_action = np.array(action).copy()
        pos = env.data.body(env.brick_body_name).xpos.copy()
        gripper_bodies = {env.model.body('gripper').id,env.model.body('moving_jaw_so101_v1').id}
        contacts = []
        for c in env.data.contact:
            if c.geom1 in env.brick_geoms and env.model.geom_bodyid[c.geom2] in gripper_bodies:
                contacts.append(int(env.model.geom_bodyid[c.geom2]))
            elif c.geom2 in env.brick_geoms and env.model.geom_bodyid[c.geom1] in gripper_bodies:
                contacts.append(int(env.model.geom_bodyid[c.geom1]))
        record = {'phase':self.phase,'time':float(env.data.time),'object_xyz':pos.tolist(),
                  'gripper_contact_bodies':sorted(set(contacts)), 'success':info['success']}
        self.trace.append(record)
        if self.on_step:
            self.on_step(before, action.copy(), record, env)
        if truncated:
            raise PlanningError('Episode timed out')
        if done and self.phase not in ('release', 'retreat', 'settle_placement'):
            # The brick slipped out and happened to land in the bowl: not a clean demonstration.
            raise PlanningError('Brick placed before release')
        return done

    def move(self, target, seconds, phase):
        self.phase = phase
        start = self.last_action.copy()
        steps = max(int(seconds*30), int(np.max(np.abs(target-start))/.025)+1)
        for i in range(1,steps+1):
            t=i/steps
            alpha=t*t*(3-2*t)
            if self.tick(start+(target-start)*alpha):return True
        return False

    def grasp_position(self):
        b = self.env.data.body('gripper')
        return b.xpos + b.xmat.reshape(3,3) @ self.grasp_point

    def held_position(self):
        """While carrying, steer by where the brick is, not the gripper: it is rarely held dead centre."""
        brick = self.env.data.body(self.env.brick_body_name).xpos
        return np.r_[brick[:2], self.grasp_position()[2]]

    def correct(self, target, yaw, grip, phase, tolerance=.002, rounds=3, point=None):
        """The fitted motors sag under gravity, as the real arm does. Like an operator, aim off by the
        observed error until the gripper actually sits at the target."""
        aim = np.array(target, float)
        for _ in range(rounds):
            error = np.asarray(target) - (point or self.grasp_position)()
            if np.linalg.norm(error) < tolerance:
                break
            aim = aim + error
            if self.move(self.solve(aim, yaw, grip), .5, phase):
                return True
        return False

    def hold(self, seconds, phase):
        self.phase = phase
        for _ in range(int(seconds*30)):
            if self.tick(self.last_action):return True
        return False

    def run(self):
        env=self.env
        self.hold(.3,'settle')
        obj=env.data.body(env.brick_body_name)
        xyz=obj.xpos.copy()
        r=obj.xmat.reshape(3,3)
        # Grip across the short side of the brick. Either sign is equivalent.
        yaw=np.arctan2(r[1,1],r[0,1])
        candidates=[]
        angles = (yaw,yaw+np.pi)
        for angle in angles:
            try:
                q=self.solve(np.r_[xyz[:2],.10],angle,.7)
                candidates.append((np.linalg.norm(q[:5]-self.last_action[:5]),angle,q))
            except PlanningError:pass
        if not candidates:raise PlanningError('No reachable pregrasp')
        _,yaw,above=min(candidates,key=lambda item:item[0])
        self.move(above,2,'approach')
        # Line up over the brick at a safe height first; correcting next to it would knock it.
        self.correct(np.r_[xyz[:2],.10],yaw,.7,'approach')
        down=self.solve(np.r_[xyz[:2],.012],yaw,.7,seed=self.last_action)
        self.move(down,1.5,'descend')
        self.correct(np.r_[xyz[:2],.012],yaw,.7,'descend')
        self.hold(.25,'align')
        # Real gripper cannot close past its calibrated closed position.
        shut=max(-.08,env.limits[5,0])
        close=self.last_action.copy();close[5]=shut
        self.move(close,1,'close')
        self.hold(.5,'grasp')
        lift=self.solve(np.r_[xyz[:2],.10],yaw,shut)
        self.move(lift,1.5,'lift')
        self.hold(.3,'lift_check')
        if env.data.body(env.brick_body_name).xpos[2] < .06:
            raise PlanningError('Grasp failed: block did not lift')
        plate=np.array(PLATES[env.destination])
        carry=self.solve(np.r_[plate,.10],yaw,shut)
        self.move(carry,2,'carry')
        self.correct(np.r_[plate,.10],yaw,shut,'carry',point=self.held_position)
        offset=self.grasp_position()[:2]-self.held_position()[:2]
        place=self.solve(np.r_[plate+offset,.024],yaw,shut,seed=self.last_action)
        self.move(place,1.5,'lower')
        self.correct(np.r_[plate,.024],yaw,shut,'lower',point=self.held_position)
        # Open wide: a brick that tilted in the grip can stay wedged on the fixed finger at 0.7 rad.
        opened=self.last_action.copy();opened[5]=1.2
        if self.move(opened,.8,'release'):return True
        if self.hold(.3,'release'):return True
        retreat=self.solve(np.r_[plate,.10],yaw,.7)
        if self.move(retreat,1.5,'retreat'):return True
        return self.hold(2,'settle_placement')
