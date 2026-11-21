"""Numerical safety, observable-only inputs and grouped evaluation checks."""
import unittest
import numpy as np
from refinement_select import features,corrected,tune,summary
from refinement_final import accepted,bootstrap
from refinement_models import WMResidualModel
import torch

def fixture():
    return dict(x=np.zeros((8,51),np.float32),wm=np.zeros((8,2),np.float32),y=np.ones((8,2)),
                age=np.arange(1,9),count=np.arange(8),unum=np.arange(1,9),distance=np.arange(8)*10.)

class RefinementTests(unittest.TestCase):
    def test_zero_residual_head_starts_exactly_at_wm(self):
        m=WMResidualModel();x=torch.zeros(2,32,51);x[:,-1,10]=.1;x[:,-1,2]=.2;x[:,-1,16]=.4
        with torch.no_grad():p=m(x,torch.ones(2,dtype=torch.long))['point']
        expected=x[:,-1,2:4]*1.5*x[:,-1,10:11]*30+x[:,-1,16:18]*10
        torch.testing.assert_close(p,expected)

    def test_truth_and_true_distance_cannot_change_runtime_features(self):
        d=fixture();p=np.ones((8,2));before=features(d,p)
        d['y'][:]=9999;d['distance'][:]=9999
        after=features(d,p)
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_estimated_distance_uses_wm_and_observer(self):
        d=fixture();d['wm'][:]=[3,4];d['x'][:,24:26]=[0,0]
        basic,_=features(d,np.ones((8,2)))
        np.testing.assert_allclose(basic[:,8],5)

    def test_cap_bounds_added_error_for_any_truth(self):
        rng=np.random.default_rng(5);wm=rng.normal(size=(10000,2));p=rng.normal(size=(10000,2))*10;y=rng.normal(size=(10000,2))*20
        for cap in [.25,.5,1.]:
            q=corrected(p,wm,.75,cap)
            extra=np.linalg.norm(q-y,axis=1)-np.linalg.norm(wm-y,axis=1)
            self.assertLessEqual(extra.max(),cap+1e-10)
            self.assertLessEqual(np.linalg.norm(q-wm,axis=1).max(),cap+1e-10)

    def test_poscount_zero_is_preserved_even_for_always_policy(self):
        d=fixture();a=accepted({'candidate':'always'},None,d,np.ones((8,2)))
        self.assertFalse(a[0]);self.assertTrue(a[1:].all())

    def test_threshold_budget_can_fall_back_to_wm(self):
        score=np.ones(8);b=np.zeros(8);e=np.ones(8)
        policies=tune(score,b-e,np.ones(8,bool),np.ones(8,bool),e,b)
        self.assertEqual(policies['harm1']['calibration_hurt25'],0)
        self.assertGreaterEqual(policies['harm1']['threshold'],1)

    def test_fast_threshold_search_matches_direct_scoring(self):
        from refinement_select import MODES
        rng=np.random.default_rng(55);g=rng.normal(size=1000);b=np.ones(1000)*3;e=b-g
        available=rng.random(1000)>.2;near=rng.random(1000)>.3
        for classification in [False,True]:
            scores=rng.random(1000) if classification else rng.normal(size=1000)
            fast=tune(scores,g,available,near,e,b,classification)
            grid=[-1e9,.1,.25,.4,.5,.6,.7,.8,.9,.95,.99,1e9] if classification else [-1e9,-.25,-.1,0,.02,.05,.1,.2,.5,1.,2.,1e9]
            for mode,(penalty,use_near,budget) in MODES.items():
                vals=[];mask=near if use_near else np.ones(1000,bool)
                for cut in grid:
                    yes=(scores>cut)&available;hurt=100*(yes&(e-b>.25)).mean()
                    value=np.where(yes[mask],g[mask]-penalty*np.maximum(-g[mask],0),0).mean()
                    vals.append(value if budget is None or hurt<=budget else -np.inf)
                expected=int(np.argmax(vals))
                self.assertEqual(fast[mode]['threshold'],grid[expected])
                self.assertAlmostEqual(fast[mode]['utility'],vals[expected],places=10)

    def test_bootstrap_preserves_sample_weights(self):
        gain=np.array([1.,1.,1.,-1.]);game=np.array([0,0,0,1])
        r=bootstrap(gain,game)
        self.assertEqual(r['games'],2);self.assertAlmostEqual(r['gain_mean'],.5)
        self.assertLessEqual(r['ci95'][0],.5);self.assertGreaterEqual(r['ci95'][1],.5)

if __name__=='__main__':unittest.main()
