#!/usr/bin/env python3
"""Assert finished pilot configuration and visual quantization/fixed-view behavior."""
import argparse,json
from pathlib import Path
from collections import Counter
import pandas as pd
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("match",type=Path);a=p.parse_args()
    files=list((a.match/"player"/a.match.name).glob("*/meta.json"))
    assert len(files)==22
    c=Counter();errors={}
    for mp in files:
        m=json.loads(mp.read_text());policy=m["run_options"]["view_policy"]
        exact=m["server_param"].get("see_no_quantization")=="true"
        assert m["final"]and m["status"]=="ok"and not m["config"]["use_fullstate"]
        assert not m["config"]["full_view"]
        f=pd.read_csv(mp.with_name("agent_cycle.csv.gz"),usecols=["cycle","stopped","game_mode","view_width","sb_view_width","see_used"])
        f["view_width"]=f.sb_view_width.fillna(f.view_width)
        f=f[f.stopped.eq(0)&f.game_mode.eq("play_on")&f.cycle.ge(10)]
        if policy!="adaptive":assert f.view_width.eq(policy).all(),(mp,f[f.view_width.ne(policy)])
        c.update({str(k):int(v)for k,v in f.view_width.value_counts().items()})
        c["decisions"]+=len(f);c["see_updates"]+=int(f.see_used.sum())
        cols=["seen","unum_visible","match_ambiguous","n_dist","n_dir","n_dist_chg","n_dir_chg","n_body_rel","n_ls_pos_d"]
        o=pd.read_csv(mp.with_name("objects.csv.gz"),usecols=cols)
        o=o[o.seen.eq(1)&o.unum_visible.eq(1)&o.match_ambiguous.fillna(0).eq(0)]
        for key in cols[3:]:
            value=float(o[key].abs().max())if o[key].notna().any()else 0.
            errors[key]=max(errors.get(key,0.),value)
    if exact:
        assert all(v<=.000005 for v in errors.values()),errors
    else:assert errors["n_dist"]>.01 and errors["n_dir"]>.01,errors
    result=dict(agents=22,exact=exact,view_policy=policy,view_counts=c,
                see_fraction=c["see_updates"]/c["decisions"],max_absolute_errors=errors)
    (a.match/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
