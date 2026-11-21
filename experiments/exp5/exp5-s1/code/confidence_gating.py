#!/usr/bin/env python3
"""Fixed-radius confidence and mixture-dominance fallback experiments.

Original predictors are frozen. Probabilities of Euclidean error <=R are
calibrated on the ten calibration games; all fixed gates are then evaluated
on the previously examined20 test games (exploratory reuse).
"""
import argparse,json,hashlib,math,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
import joblib

THRESHOLDS=[.5,.8,.9,.95]
RADII=[.25,.5]
EXTRA=[0.,.05,.1,.25,.5,1.,2.]
TOL=1e-5

def gaussian_circle_probability(cov,delta,radius,nodes=48):
    """P(||X||<=R), X~N(delta,cov), via 1D quadrature after rotation.

    The circle is rotation invariant. Integrating along the smaller-variance
    axis in standard-normal units also handles highly anisotropic normals.
    """
    cov=np.asarray(cov,np.float64);delta=np.asarray(delta,np.float64)
    a,b,c=cov.T;disc=np.hypot(a-b,2*c)
    vmax=np.maximum((a+b+disc)/2,1e-10);vmin=np.maximum((a+b-disc)/2,1e-10)
    theta=.5*np.arctan2(2*c,a-b);co=np.cos(theta);si=np.sin(theta)
    dx=-si*delta[:,0]+co*delta[:,1];dy=co*delta[:,0]+si*delta[:,1]
    sx=np.sqrt(vmin);sy=np.sqrt(vmax)
    low=np.maximum((-radius-dx)/sx,-9.);high=np.minimum((radius-dx)/sx,9.)
    width=np.maximum(high-low,0)/2;mid=(high+low)/2
    z,w=np.polynomial.legendre.leggauss(nodes)
    grid=mid[:,None]+width[:,None]*z
    xx=dx[:,None]+sx[:,None]*grid
    yy=np.sqrt(np.maximum(radius*radius-xx*xx,0))
    mass=ndtr((yy-dy[:,None])/sy[:,None])-ndtr((-yy-dy[:,None])/sy[:,None])
    integrand=np.exp(-.5*grid*grid)/math.sqrt(2*math.pi)*mass
    return np.clip(width*(integrand*w).sum(1),0,1)

def circle_probabilities(arrays,kind,radius,output,workers=4,chunk=65536):
    n=len(arrays['pred'])
    if output.exists():
        cached=np.load(output,mmap_mode='r')
        if cached.shape!=(n,):raise ValueError('Probability cache shape mismatch')
        return cached
    temp=output.with_suffix('.tmp.npy');out=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(n,))
    if kind=='gaussian':
        def compute(start):
            end=min(start+chunk,n);cov=arrays['cov'][start:end]
            delta=arrays['center'][start:end]-arrays['pred'][start:end]
            return start,end,gaussian_circle_probability(cov,delta,radius)
    else:
        means=arrays['means'].reshape(n,3,2);scale=arrays['scale'].reshape(n,3,2)
        weight=arrays['weight'];rho=arrays['rho'];maximum=weight.argmax(1)
        def compute(start):
            end=min(start+chunk,n);anchor=means[np.arange(start,end),maximum[start:end]]
            result=np.zeros(end-start,np.float64)
            for k in range(3):
                s=scale[start:end,k];cov=np.stack([s[:,0]**2,s[:,1]**2,rho[start:end,k]*s[:,0]*s[:,1]],1)
                delta=means[start:end,k]-anchor
                result+=weight[start:end,k]*gaussian_circle_probability(cov,delta,radius)
            return start,end,result
    begin=time.monotonic();jobs=list(range(0,n,chunk))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for j,(start,end,p) in enumerate(pool.map(compute,jobs),1):
            out[start:end]=p
            if j%50==0 or j==len(jobs):print(f'{kind} R={radius}: {end:,}/{n:,} probability rows; {time.monotonic()-begin:.0f}s',flush=True)
    out.flush();del out;temp.replace(output)
    return np.load(output,mmap_mode='r')

def selected_point(arrays,kind):
    if kind=='gaussian':return arrays['pred']
    n=len(arrays['pred']);mu=arrays['means'].reshape(n,3,2);w=arrays['weight']
    return mu[np.arange(n),w.argmax(1)]

def summarize(candidate_error,baseline,accept,accuracy_radius=None):
    candidate_error=np.asarray(candidate_error);baseline=np.asarray(baseline)
    accept=np.asarray(accept,bool)
    used=np.where(accept,candidate_error,baseline)
    extra=used.astype(np.float64)-baseline.astype(np.float64)
    accepted=int(accept.sum());gain=baseline.astype(np.float64)-used.astype(np.float64)
    return dict(n=len(used),accepted=accepted,use_model_pct=float(100*accept.mean()),
                mean=float(used.astype(float).mean()),stock_mean=float(baseline.astype(float).mean()),
                gain=float(gain.mean()),gain_pct=float(100*gain.mean()/baseline.mean()),
                p95=float(np.quantile(used,.95)),max_error=float(used.max()),max_added_error=float(max(0,extra.max())),
                worsened_pct={str(t):float(100*(extra>t+TOL).mean()) for t in EXTRA},
                absolute_error_over_pct={str(t):float(100*(used>t+TOL).mean()) for t in [.25,.5,1.,2.]},
                selected_mean_error=float(candidate_error[accept].mean()) if accepted else None,
                selected_within_radius_pct=float(100*(candidate_error[accept]<=accuracy_radius).mean()) if accepted and accuracy_radius else None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--validation-targets',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--radii',type=float,nargs='+',default=RADII)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    signature=dict(original=json.loads((args.evaluation/'signature.json').read_text()),
                   code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),radii=args.radii,thresholds=THRESHOLDS)
    sf=args.output/'signature.json'
    if sf.exists() and json.loads(sf.read_text())!=signature:raise ValueError('Changed confidence experiment: use a new output directory')
    sf.write_text(json.dumps(signature,indent=2)+'\n')
    targetfile=np.load(args.validation_targets)
    with np.load(args.evaluation/'cohort.npz') as cohort:
        baseline=cohort['baseline'];truth=cohort['truth']
    rows=[];calibration_stats={}
    result=dict(semantics='Fixed-radius probability means P(position error <=R) about the actual point used; an80% region coverage level is not a per-row certainty score.',
                thresholds=THRESHOLDS,radii_m=args.radii,extra_error_thresholds_m=EXTRA,n=len(baseline),
                caveat='Exploratory reuse of previously inspected20 test games. Gaussian/mixture weights frozen. Isotonic maps fit only on cached ten-game calibration rows. Thresholds/radii fixed before test scoring; no ground-truth runtime features.',
                mixture_readouts='Probability-radius gates use the most-probable component center. Dominance gates compare original weighted-mean and most-probable-component readouts separately.',
                baseline=summarize(baseline,baseline,np.zeros(len(baseline),bool)),policies=rows,calibration=calibration_stats)
    for kind in ['gaussian','mixture']:
        arrays_by_split={}
        for split in ['calibration','test']:
            with np.load(args.evaluation/(kind+'-'+split+'.npz')) as saved:
                fields=['pred','error','cov','center'] if kind=='gaussian' else ['pred','error','means','scale','rho','weight']
                arrays_by_split[split]={k:saved[k] for k in fields}
        cal=arrays_by_split['calibration'];test=arrays_by_split['test']
        calpoint=selected_point(cal,kind);testpoint=selected_point(test,kind)
        calerror=cal['error'] if kind=='gaussian' else np.linalg.norm(calpoint-targetfile['calibration_y'],axis=1)
        testerror=test['error'] if kind=='gaussian' else np.linalg.norm(testpoint-truth,axis=1)
        rows.append(dict(kind=kind,gate='always',readout='mean' if kind=='gaussian' else 'original weighted mean',
                         radius_m=None,threshold=None,**summarize(test['error'],baseline,np.ones(len(baseline),bool))))
        if kind=='mixture':
            rows.append(dict(kind=kind,gate='always',readout='highest-weight center',radius_m=None,threshold=None,
                             **summarize(testerror,baseline,np.ones(len(baseline),bool))))
            maxw=test['weight'].max(1)
            for threshold in THRESHOLDS:
                accept=maxw>=threshold
                for readout,error in [('original weighted mean',test['error']),('highest-weight center',testerror)]:
                    rows.append(dict(kind=kind,gate='max_component_weight',readout=readout,radius_m=None,threshold=threshold,
                                     **summarize(error,baseline,accept,.25)))
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        for radius in args.radii:
            rawcal=circle_probabilities(cal,kind,radius,args.output/(kind+f'-R{radius:g}-calibration.npy'),args.workers)
            calibrator=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(rawcal,(calerror<=radius).astype(float))
            joblib.dump(calibrator,args.output/(kind+f'-R{radius:g}-isotonic.joblib'))
            rawtest=circle_probabilities(test,kind,radius,args.output/(kind+f'-R{radius:g}-test.npy'),args.workers)
            probability=calibrator.predict(rawtest)
            calibration_stats[kind+f'/R{radius:g}']=dict(raw_mean_probability=float(rawtest.mean()),
                calibrated_mean_probability=float(probability.mean()),actual_within_radius=float((testerror<=radius).mean()),calibration_rows=len(rawcal))
            for threshold in THRESHOLDS:
                rows.append(dict(kind=kind,gate='calibrated_probability_within_radius',readout='mean' if kind=='gaussian' else 'highest-weight center',
                                 radius_m=radius,threshold=threshold,**summarize(testerror,baseline,probability>=threshold,radius)))
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(kind,'radius',radius,'confidence gates complete',flush=True)
        del arrays_by_split
    print('Confidence study complete:',args.output/'results.json',flush=True)

if __name__=='__main__':main()
