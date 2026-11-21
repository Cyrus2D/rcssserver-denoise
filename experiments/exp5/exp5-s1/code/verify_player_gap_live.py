"""Validate native live inputs against the original causal Python encoder."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from prepare import make_features,OBS_COLS,AGENT_COLS,FEATURES

def verify(root, model_dir=None, observers=None, cycles=None):
    if model_dir:
        import torch,pickle
        from refinement_models import get_model
        from refinement_select import features
        torch.set_num_threads(1)
        bundle=json.loads((model_dir/'manifest.json').read_text())
        checkpoint=Path(bundle['checkpoint'])
        model=get_model(checkpoint,'cpu')
        selector=pickle.load((checkpoint.parent.parent/(checkpoint.parent.name+'-selectors.pkl')).open('rb'))['strong_cost']['model']

    result=dict(observers=0,applied=0,eligible=0,feature_rows=0,max_feature_error=0.,features={},modes={})
    result.update(output_samples=0,max_position_difference_m=0.,max_selector_difference=0.,selection_mismatches=0)
    result.update(routing_rows=0,routing_applied_rows=0)
    for file in sorted(root.glob('*/meta.json')):
        if observers and file.parent.name not in observers:continue
        meta=json.loads(file.read_text());assert meta['final'] and meta['status']=='ok',file
        options=meta.get('run_options',{});mode=options.get('player_gap','off');policy=options.get('player_gap_pos_count','preserve');path=file.parent
        result['observers']+=1;result['modes'][mode]=result['modes'].get(mode,0)+1
        predictions=pd.read_csv(path/'player_prediction.csv.gz')
        if len(predictions):
            eligible=predictions.eligible.eq(1);applied=predictions.applied.eq(1)
            assert not (applied&~eligible).any();assert not (applied&~predictions.selected.eq(1)).any()
            if mode!='apply':assert not applied.any()
            assert (predictions.loc[eligible,'stock_count']>0).all()
            assert predictions.loc[eligible,'named_age'].between(1,30).all()
            if policy=='zero':assert predictions.loc[applied,'decision_count'].eq(0).all()
            elif policy=='half':assert predictions.loc[applied,'decision_count'].eq(predictions.loc[applied,'stock_count'].floordiv(2).clip(lower=1)).all()
            else:assert predictions.loc[applied,'decision_count'].eq(predictions.loc[applied,'stock_count']).all()
            assert predictions.loc[eligible&~applied,'decision_count'].eq(predictions.loc[eligible&~applied,'stock_count']).all()
            result['eligible']+=int(eligible.sum());result['applied']+=int(applied.sum())
        cpp=pd.read_csv(path/'player_prediction_features.csv.gz')
        if not len(cpp) and not model_dir:continue
        a=pd.read_csv(path/'agent_cycle.csv.gz',usecols=AGENT_COLS);a=a[a.decision_made.eq(1)].copy()
        continuous=a.stopped.eq(0)&a.stopped.shift().eq(0)&a.cycle.diff().eq(1)&a.game_mode.eq('play_on')&a.game_mode.shift().eq('play_on')
        a['block']=(~continuous).cumsum();a=a[a.stopped.eq(0)&a.game_mode.eq('play_on')].reset_index(drop=True)
        columns=OBS_COLS+['trk_stock_pos_x','trk_stock_pos_y','trk_stock_pos_count','trk_prediction_active']
        o=pd.read_csv(path/'objects.csv.gz',usecols=columns,low_memory=False)
        if len(predictions):
            # Check the actually exposed WM before replacing decision inputs with stock.
            j=predictions[predictions.object_id.gt(0)].merge(o,on=['cycle','stopped','object_id'],validate='one_to_one')
            valid=j.stock_x.notna()&j.trk_stock_pos_x.notna()
            for left,right in [('used_x','trk_pos_x'),('used_y','trk_pos_y'),('stock_x','trk_stock_pos_x'),('stock_y','trk_stock_pos_y')]:
                assert np.allclose(j.loc[valid,left],j.loc[valid,right],atol=1e-7,rtol=0),(file,left)
            assert j.loc[valid,'stock_count'].eq(j.loc[valid,'trk_stock_pos_count']).all()
            assert j.loc[valid,'decision_count'].eq(j.loc[valid,'trk_pos_count']).all()
            assert j.loc[valid,'applied'].eq(j.loc[valid,'trk_prediction_active']).all()
            result['routing_rows']+=int(valid.sum())
            result['routing_applied_rows']+=int(j.loc[valid,'applied'].eq(1).sum())
        else:
            assert not o.trk_prediction_active.eq(1).any()
        for key in ['pos_x','pos_y','pos_count']:
            o.loc[o.object_id.gt(0),'trk_'+key]=o.loc[o.object_id.gt(0),'trk_stock_'+key]
        o=o[o.stopped.eq(0)].merge(a[['cycle']].assign(row=np.arange(len(a))),on='cycle',validate='many_to_one')
        d={};rows=o.row.to_numpy(int);ids=o.object_id.to_numpy(int)
        for key in OBS_COLS[3:]:
            values=o[key].eq('unum').astype(float) if key in ['match_method','track_match'] else o[key]
            grid=np.full((len(a),23),np.nan,np.float32);grid[rows,ids]=values.to_numpy(float);d[key]=grid
        self_id=meta['unum']+(0 if meta['side']=='l' else 11);targets=np.array([i for i in range(1,23) if i!=self_id]);lookup={int(v):i for i,v in enumerate(targets)}
        expected,anchor,_,_,_,_,_,wm=make_features(a,d,targets,meta['side'])
        merged=cpp.merge(a[['cycle']].assign(row=np.arange(len(a))),on='cycle',validate='many_to_one')
        correct=expected[merged.row.to_numpy(int),np.array([lookup[int(v)] for v in merged.object_id],dtype=int)]
        actual=merged[['f'+str(i) for i in range(51)]].to_numpy(float)
        error=np.max(np.abs(actual-correct),axis=0) if len(actual) else np.zeros(51);worst=float(error.max())
        result['feature_rows']+=len(merged);result['max_feature_error']=max(result['max_feature_error'],worst)
        for key,v in zip(FEATURES,error):result['features'][key]=max(result['features'].get(key,0),float(v))
        if model_dir:
            candidates=predictions[predictions.eligible.eq(1)]
            if cycles:candidates=candidates[candidates.cycle.isin(cycles)]
            if len(candidates):
                chosen=candidates.iloc[np.linspace(0,len(candidates)-1,min(24,len(candidates))).astype(int)]
                rows={int(v):i for i,v in enumerate(a.cycle)}
                rr=np.array([rows[int(v)] for v in chosen.cycle])
                jj=np.array([lookup[int(v)] for v in chosen.object_id])
                windows=np.zeros((len(rr),32,51),np.float32)
                blocks=a.block.to_numpy()
                for k,(r,j) in enumerate(zip(rr,jj)):
                    start=r
                    while start>0 and r-start<31 and blocks[start-1]==blocks[r]:start-=1
                    windows[k,32-(r-start+1):]=expected[start:r+1,j]
                unum=(chosen.object_id.to_numpy(int)-1)%11+1
                with torch.inference_mode():
                    point=model(torch.from_numpy(windows),torch.from_numpy(unum))['point'].numpy()
                stock=(wm[rr,jj]-anchor[rr,jj]).astype(np.float32)
                score=selector.predict_proba(features(dict(x=windows[:,-1],wm=stock,count=chosen.stock_count.to_numpy(),age=chosen.named_age.to_numpy(),unum=unum),point)[1])[:,1]
                predicted=(point+anchor[rr,jj])*(1 if meta['side']=='l' else -1)
                delta=np.linalg.norm(predicted-chosen[['pred_x','pred_y']].to_numpy(),axis=1)
                score_delta=np.abs(score-chosen.score.to_numpy())
                mismatch=int(np.count_nonzero((score>bundle['threshold'])!=chosen.selected.eq(1)))
                result['output_samples']+=len(chosen)
                result['max_position_difference_m']=max(result['max_position_difference_m'],float(delta.max()))
                result['max_selector_difference']=max(result['max_selector_difference'],float(score_delta.max()))
                result['selection_mismatches']+=mismatch
                assert delta.max()<.0003 and mismatch==0,(file,float(delta.max()),float(score_delta.max()),mismatch)
        if not np.allclose(actual,correct,atol=3e-6,rtol=3e-7):
            t,k=np.unravel_index(np.argmax(np.abs(actual-correct)),actual.shape)
            raise AssertionError(dict(observer=file.parent.name,feature=FEATURES[k],cycle=int(merged.iloc[t].cycle),target=int(merged.iloc[t].object_id),cpp=float(actual[t,k]),python=float(correct[t,k]),error=worst))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('logs',type=Path);p.add_argument('--output',type=Path);p.add_argument('--model-dir',type=Path);p.add_argument('--observer',action='append');p.add_argument('--cycle',type=int,action='append');a=p.parse_args();r=verify(a.logs,a.model_dir,a.observer,a.cycle)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='features'},indent=2))
