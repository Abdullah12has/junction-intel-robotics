import unittest
import numpy as np
from visual_servo import (Calibration, AlignmentController, bounded_target, detect_target, fit_jacobian, validate_chunk, GraspVerifier)

class ServoTests(unittest.TestCase):
    def calibration(self):
        return Calibration('wrist',(640,480),np.array([0,0,0,0,0,50.]),np.array([320.,240.]),
                           np.array([[8.,2.],[-1.,6.]]),[0,1],10.,80.,2.,'test')

    def test_closed_loop_converges_with_measured_jacobian(self):
        c=self.calibration(); ctl=AlignmentController(c); q=c.reference_state.copy(); error=np.array([12.,-8.])
        for i in range(150):
            command,status=ctl.update(q,c.reference_uv+error,i/30,i/30)
            error += c.jacobian @ (command-q)[c.joint_indices]
            self.assertEqual(command[5],50.)
            self.assertLessEqual(np.max(np.abs(command-q)), .1000001)
            q=command
            if status=='aligned': break
        self.assertEqual(status,'aligned'); self.assertLessEqual(np.linalg.norm(error),2.)

    def test_converges_with_servo_deadband(self):
        c=self.calibration();ctl=AlignmentController(c,max_seconds=10.)
        q=c.reference_state.copy();error=np.array([12.,-8.]);previous=q.copy()
        for i in range(300):
            command,status=ctl.update(q,c.reference_uv+error,i/30,i/30)
            self.assertLessEqual(np.max(np.abs(command-previous)),.100001)
            previous=command.copy()
            difference=command-q
            # Joint stays still until accumulated goal exceeds a 0.4-unit deadband.
            actual=np.where(np.abs(difference)>.4, difference*.6, 0.)
            error += c.jacobian @ actual[c.joint_indices]
            q += actual
            if status=='aligned':break
        self.assertEqual(status,'aligned')

    def test_stalled_actuator_cannot_accumulate_unbounded_commands(self):
        c=self.calibration();ctl=AlignmentController(c)
        with self.assertRaisesRegex(ValueError,'tracking gap'):
            for i in range(100):
                ctl.update(c.reference_state,c.reference_uv+[50.,0.],i/30,i/30)

    def test_stale_feedback_and_repeated_frame_stop(self):
        c=self.calibration(); ctl=AlignmentController(c)
        with self.assertRaises(ValueError):ctl.update(c.reference_state,c.reference_uv,0,1)
        ctl=AlignmentController(c);ctl.update(c.reference_state,c.reference_uv,1,1)
        with self.assertRaises(ValueError):ctl.update(c.reference_state,c.reference_uv,1,1.01)

    def test_outside_pose_and_target_stop(self):
        for q,uv in [(np.array([20,0,0,0,0,50]),[320,240]),(self.calibration().reference_state,[500,240])]:
            with self.assertRaises(ValueError):AlignmentController(self.calibration()).update(q,uv,1,1)

    def test_nan_and_invalid_rate_rejected(self):
        with self.assertRaises(ValueError):validate_chunk([[np.nan]*6])
        with self.assertRaises(ValueError):bounded_target([0]*6,[0]*5+[50],-1,1/30)

    def test_fit_recovers_sign_and_scale(self):
        x=np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1]])
        j=self.calibration().jacobian
        actual,rmse=fit_jacobian(x,x@j.T,['shoulder_pan','shoulder_lift'])
        np.testing.assert_allclose(actual,j);self.assertLess(rmse,1e-10)
        with self.assertRaises(ValueError):fit_jacobian(np.ones((6,2)),np.ones((6,2)),['shoulder_pan','shoulder_lift'])

    def test_detector_centroid_and_ambiguity(self):
        frame=np.zeros((100,100,3),np.uint8);frame[40:50,30:40]=[255,0,0]
        args=dict(hsv_low=[170,150,100],hsv_high=[10,255,255],roi=[0,0,100,100])
        uv,_,_=detect_target(frame,**args);np.testing.assert_allclose(uv,[34.5,44.5])
        frame[70:80,70:80]=[255,0,0]
        with self.assertRaises(ValueError):detect_target(frame,**args)

    def test_stability_is_not_one_frame_success(self):
        c=self.calibration();ctl=AlignmentController(c)
        self.assertEqual(ctl.update(c.reference_state,c.reference_uv,0,0)[1],'settling')
        self.assertEqual(ctl.update(c.reference_state,c.reference_uv,.03,.03)[1],'settling')
        self.assertEqual(ctl.update(c.reference_state,c.reference_uv,.06,.06)[1],'aligned')

class GraspTests(unittest.TestCase):
    def test_requires_measured_colift_and_revokes_on_drop(self):
        v=GraspVerifier();v.begin_lift([0,0,0],[0,0,.03],'cyan')
        for i in range(5):
            self.assertFalse(v.update([0,0,0],[0,0,.06],'cyan',i/30,i/30))
        for i in range(5,9):
            self.assertFalse(v.update([0,0,.03],[0,0,.06],'cyan',i/30,i/30))
        self.assertTrue(v.update([0,0,.03],[0,0,.06],'cyan',.3,.3))
        self.assertFalse(v.update([0,0,0],[0,0,.06],'cyan',.34,.34))
        self.assertFalse(v.update([0,0,.03],[0,0,.06],'other',.38,.38))

if __name__=='__main__':unittest.main()
