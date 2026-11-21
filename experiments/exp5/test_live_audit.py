"""Regression tests for Exp5 aggregation and counter/identity interpretation."""
import importlib.util
from pathlib import Path
import unittest
from collections import Counter
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('live_audit',Path(__file__).with_name('live_audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class AuditTest(unittest.TestCase):
    def test_exact_age_boundaries_and_missing(self):
        self.assertEqual(audit.age_labels([0,1,2,3,4,5,6,999,np.nan]).tolist(),['0','1','2','3','4','5','>5','>5','missing'])

    def test_sample_weighting_and_missing_errors(self):
        one,two={},{}
        audit.accumulate(one,'x',[0.]*10+[np.nan])
        audit.accumulate(two,'x',[11.])
        audit.merge_stats(one,two)
        self.assertEqual(one['x'][0],11)
        self.assertEqual(one['x'][1]/one['x'][0],1.)
        self.assertEqual(one['x'][3],11.)

    def test_velocity_body_have_independent_ages(self):
        f=pd.DataFrame(dict(actual_view=['wide','narrow'],est_pos_count=[0,1],est_vel_count=[3,5],est_face_count=[2,4],
            n_est_pos_d=[0.,1.],n_est_vel_d=[.3,np.nan],n_est_body=[-20.,40.],see_used=[1,0]))
        stats,counts={},Counter()
        audit.state_stats(stats,counts,'self',f,'est_',['pos_count','vel_count','face_count'])
        self.assertEqual(stats['state/self/velocity/own/3/all'][0],1)
        self.assertEqual(stats['state/self/velocity/pos/0/all'][0],1)
        self.assertNotIn('state/self/velocity/own/5/all',stats)
        self.assertEqual(stats['state/self/body/own/2/all'][1],20.)
        self.assertEqual(counts['measured/self/velocity'],1)

    def test_unknown_counter_is_not_mapped_to_fresh(self):
        f=pd.DataFrame(dict(actual_view=['normal'],est_pos_count=[np.nan],est_vel_count=[np.nan],est_face_count=[np.nan],
            n_est_pos_d=[1.],n_est_vel_d=[.2],n_est_body=[3.],see_used=[0]))
        stats,counts={},Counter()
        audit.state_stats(stats,counts,'self',f,'est_',['pos_count','vel_count','face_count'])
        self.assertNotIn('state/self/position/own/0/all',stats)
        self.assertEqual(stats['state/self/position/all/all/all'][0],1)

if __name__=='__main__':unittest.main()
