"""Leakage, reset, probability and padding regressions for Exp5-S1."""
import unittest
import numpy as np
import pandas as pd
import torch
from prepare import make_features,last_indices,OBS_COLS
from model import PositionModel,loss,log_prob,covariance
from evaluate import apply_gate,metrics,AGES
from data import Data

def fixture(future=2.):
    a=pd.DataFrame(dict(cycle=np.arange(1,7),block=[1,1,1,2,2,2],est_pos_x=0.,est_pos_y=0.,est_pos_count=0.,est_face=0.,sb_view_width='normal',view_width='normal',see_used=1))
    d={k:np.full((6,23),np.nan) for k in OBS_COLS[3:]}
    for i in [0,1,2]:
        d['trk_pos_x'][:,i]=i*3;d['trk_pos_y'][:,i]=0;d['trk_vel_x'][:,i]=.5;d['trk_vel_y'][:,i]=0
        d['trk_pos_count'][:,i]=0;d['trk_vel_count'][:,i]=0;d['track_match'][:,i]=1
    for t,x in [(0,0.),(5,future)]:
        for k,value in dict(seen=1,unum_visible=1,match_method=1,ls_pos_x=x,ls_pos_y=0,ls_vel_x=.5,ls_vel_y=0,dist_chg=.5,dir_chg=0,body_rel=0).items():d[k][t,1]=value
    return a,d

class DatasetTests(unittest.TestCase):
    def test_windows_pad_before_reset_and_never_read_future(self):
        data=Data.__new__(Data);data.x=np.ones((6*21,51),np.float32)
        data.x[:3*21]=99;data.x[5*21:]=777
        data.start=np.zeros(6*21,np.uint32);data.start[3*21:]=np.tile(np.arange(21)+3*21,3)
        data.unum=np.ones(6*21,np.uint8);data.y=np.zeros((6*21,2),np.float32)
        x,_,_=data.batch(np.array([4*21+2]),3,'cpu')
        self.assertTrue((x[0,0]==0).all());self.assertTrue((x[0,1:]==1).all())

    def test_future_sighting_cannot_change_past_features(self):
        a,d=fixture(2);x=make_features(a,d,np.array([1,2]),'l')
        a,d=fixture(99);y=make_features(a,d,np.array([1,2]),'l')
        np.testing.assert_array_equal(x[0][:5],y[0][:5])
        np.testing.assert_array_equal(x[1][:5],y[1][:5])

    def test_truth_cannot_change_inputs(self):
        a,d=fixture();x=make_features(a,d,np.array([1,2]),'l')[0]
        d['tru_pos_x']=np.full((6,23),999999.);d['true_dist']=np.full((6,23),-1000.)
        np.testing.assert_array_equal(x,make_features(a,d,np.array([1,2]),'l')[0])

    def test_no_history_crosses_playon_reset(self):
        a,d=fixture();result=make_features(a,d,np.array([1,2]),'l')
        self.assertTrue(result[4][2,0]);self.assertFalse(result[4][3,0]);self.assertTrue(result[4][5,0])
        self.assertEqual(result[3][2,0],2)

    def test_named_identity_required_for_history(self):
        a,d=fixture();d['unum_visible'][:]=0
        self.assertFalse(make_features(a,d,np.array([1,2]),'l')[4].any())

class ProbabilityTests(unittest.TestCase):
    def test_all_heads_have_finite_gradients(self):
        torch.set_num_threads(1)
        for kind in ['point','gaussian','mixture']:
            model=PositionModel(51,kind);x=torch.randn(8,4,51);x[:,-1,10]=.1
            o=model(x,torch.ones(8,dtype=torch.long));l=loss(o,torch.zeros(8,2));l.backward()
            self.assertTrue(torch.isfinite(l))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if kind!='point':
                torch.testing.assert_close(o['weight'].sum(-1),torch.ones(8))
                v=covariance(o);self.assertTrue(((v[:,0]*v[:,1]-v[:,2]**2)>0).all())

    def test_gaussian_log_density(self):
        o=dict(means=torch.zeros(2,1,2),scale=torch.ones(2,1,2),rho=torch.zeros(2,1),weight=torch.ones(2,1))
        expected=torch.tensor([-np.log(2*np.pi),-np.log(2*np.pi)-.5],dtype=torch.float32)
        torch.testing.assert_close(log_prob(o,torch.tensor([[0.,0.],[1.,0.]])),expected)

    def test_conservative_policy_bounds_added_distance_error(self):
        from types import SimpleNamespace
        rng=np.random.default_rng(50);truth=rng.normal(size=(1000,2));wm=rng.normal(size=(1000,2))
        data=SimpleNamespace(wm=wm,y=truth,age=np.ones(1000,int))
        pred=dict(pred=rng.normal(size=(1000,2))*20)
        point,accept=apply_gate(data,np.arange(1000),pred,{k:1e9 for k in AGES},None)
        self.assertTrue(accept.all())
        self.assertLessEqual(np.linalg.norm(point-wm,axis=1).max(),.25000001)
        self.assertLessEqual((np.linalg.norm(point-truth,axis=1)-np.linalg.norm(wm-truth,axis=1)).max(),.25000001)

    def test_harm_and_benefit_signs(self):
        result=metrics(np.array([.5,1.5,1.]),np.ones(3))
        self.assertAlmostEqual(result['gain'],0.)
        self.assertAlmostEqual(result['help_pct'],100/3)
        self.assertAlmostEqual(result['hurt_pct'],100/3)
        self.assertAlmostEqual(result['max_harm'],.5)

if __name__=='__main__':unittest.main()
