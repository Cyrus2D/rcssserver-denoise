"""Small numerical and fallback checks for the offline follow-up studies."""
import unittest
import numpy as np
from confidence_gating import gaussian_circle_probability,summarize
from selector_study import selector_features
from types import SimpleNamespace

class FollowupTests(unittest.TestCase):
    def test_isotropic_circle_has_analytic_probability(self):
        sigma=np.array([.05,.2,.7,2.])
        cov=np.column_stack([sigma**2,sigma**2,np.zeros(4)])
        actual=gaussian_circle_probability(cov,np.zeros((4,2)),.25)
        expected=1-np.exp(-.25**2/(2*sigma**2))
        np.testing.assert_allclose(actual,expected,atol=3e-6)

    def test_rotating_distribution_preserves_circle_probability(self):
        c=np.array([[.03,.008],[.008,.15]]);d=np.array([.1,-.2]);a=.71
        r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        pack=lambda c:np.array([[c[0,0],c[1,1],c[0,1]]])
        first=gaussian_circle_probability(pack(c),d[None],.5)
        second=gaussian_circle_probability(pack(r@c@r.T),(r@d)[None],.5)
        np.testing.assert_allclose(first,second,atol=1e-10)

    def test_rejected_rows_stay_in_denominator_and_keep_wm(self):
        m=summarize(np.array([.5,10.,.1]),np.array([1.,1.,.2]),np.array([True,False,False]))
        self.assertEqual(m['n'],3);self.assertEqual(m['accepted'],1)
        self.assertAlmostEqual(m['mean'],(0.5+1.+.2)/3)
        self.assertEqual(m['max_added_error'],0)
        self.assertEqual(m['worsened_pct']['0.0'],0)

    def test_selector_features_do_not_read_truth_labels(self):
        data=SimpleNamespace(x=np.zeros((3,51),np.float32),wm=np.zeros((3,2)),
                             age=np.ones(3),unum=np.array([1,7,11]),y=np.zeros((3,2)))
        pred=np.ones((3,2));ids=np.arange(3)
        before=selector_features(data,ids,pred)
        data.y[:]=99999
        np.testing.assert_array_equal(before,selector_features(data,ids,pred))
        self.assertEqual(before.shape,(3,71))

if __name__=='__main__':unittest.main()
