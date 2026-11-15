#!/usr/bin/env python3
"""Audit a combined Exp3/Exp4 live batch with paired, per-row metrics."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

KEY = ["cycle", "stopped"]
AGENT = "game_mode decision_made see_used est_pos_x est_pos_y est_face msec_sb_to_decision est_kickable body_cmd action_label".split()
FIELDS = """object_id truth_valid tru_on_field true_dist true_dir tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body
body_rel dist_chg ls_pos_x ls_pos_y ls_vel_x ls_vel_y trk_pos_x trk_pos_y trk_vel_x trk_vel_y trk_body trk_pos_count trk_body_count trk_vel_count
denoise_pos_x denoise_pos_y denoise_applied denoise_us
exp4_pos_x exp4_pos_y exp4_vel_x exp4_vel_y exp4_body exp4_body_applied
exp4_next_pos_x exp4_next_pos_y exp4_next_vel_x exp4_next_vel_y
exp4_stock_pos_x exp4_stock_pos_y exp4_stock_vel_x exp4_stock_vel_y exp4_stock_body exp4_applied""".split()
WIDTH, SIZE = .005, 20001


def norm(x, y):
    return np.hypot(x, y)


def vector(frame, prefix):
    return frame[[prefix + "_x", prefix + "_y"]].to_numpy()


def angle_error(x, y):
    return np.abs((x - y + 180) % 360 - 180)


def add(stats, key, old, new, width=WIDTH):
    old, new = np.asarray(old), np.asarray(new)
    mask = np.isfinite(old) & np.isfinite(new)
    old, new = old[mask], new[mask]
    if not len(old):
        return
    s = np.array([len(old), old.sum(dtype=np.float64), new.sum(dtype=np.float64),
                  (new > old + .0002).sum(), (new > old + .1).sum(), (new > old + .5).sum(),
                  (new > old + 1).sum()])
    h = np.stack([np.bincount(np.minimum((a / width).astype(np.int64), SIZE - 1), minlength=SIZE)
                  for a in (old, new)])
    stats[key] = (s, h, width)


def pair(stats, key, frame, old, new, truth, body=False):
    if body:
        add(stats, key, angle_error(old, truth), angle_error(new, truth), .1)
    else:
        add(stats, key, np.linalg.norm(old - truth, axis=1), np.linalg.norm(new - truth, axis=1))


def read_observer(path):
    meta = json.loads((path / "meta.json").read_text())
    a = pd.read_csv(path / "agent_cycle.csv.gz", usecols=KEY + AGENT)
    a = a[a.game_mode.eq("play_on") & a.decision_made.eq(1) & a.stopped.eq(0)]
    cols = set(pd.read_csv(path / "objects.csv.gz", nrows=0).columns)
    o = pd.read_csv(path / "objects.csv.gz", usecols=KEY + [f for f in FIELDS if f in cols])
    # Keep only decisions in scope. Reference is never used in feature construction.
    o = o.merge(a, on=KEY, validate="many_to_one")
    counters = Counter(decisions=len(a), sees=int(a.see_used.sum()), mode=0)
    latency = np.bincount(np.minimum(a.msec_sb_to_decision.fillna(0).clip(lower=0).astype(int), 2000), minlength=2001)
    stats = {}
    ball = o[o.object_id.eq(0)].set_index("cycle")
    player = o[o.object_id.ne(0)]
    pos = player[player.denoise_applied.eq(1) & player.truth_valid.eq(1) & player.tru_on_field.eq(1)].copy()
    if len(pos):
        theta = np.deg2rad(pos.est_face + pos.true_dir)
        relative_truth = np.column_stack((pos.est_pos_x + pos.true_dist * np.cos(theta),
                                          pos.est_pos_y + pos.true_dist * np.sin(theta)))
        pair(stats, "player_position_relative", pos, vector(pos, "ls_pos"), vector(pos, "denoise_pos"), relative_truth)
        pair(stats, "player_position_global", pos, vector(pos, "ls_pos"), vector(pos, "denoise_pos"), vector(pos, "tru_pos"))
    candidates = o[o.exp4_vel_x.notna()]
    p = candidates[candidates.object_id.ne(0) & candidates.truth_valid.eq(1) & candidates.tru_on_field.eq(1)]
    b = candidates[candidates.object_id.eq(0) & candidates.truth_valid.eq(1) & candidates.tru_on_field.eq(1)]
    counters.update(exp4_player=len(p), exp4_ball=len(b), exp4_applied=int(candidates.exp4_applied.eq(1).sum()),
                    exp3_applied=int(player.denoise_applied.eq(1).sum()),
                    reported_body_overwritten=int((p.body_rel.notna() & p.exp4_body_applied.eq(1)).sum()))
    if len(p):
        oldvel = vector(p, "exp4_stock_vel")
        valid_vel = np.isfinite(oldvel).all(axis=1)
        oldvel = np.nan_to_num(oldvel)
        newvel = vector(p, "exp4_vel")
        truthvel = vector(p, "tru_vel")
        stock_body = p.exp4_stock_body.to_numpy()
        missing_body = ~np.isfinite(stock_body)
        stock_body = np.where(missing_body, np.rad2deg(np.arctan2(oldvel[:, 1], oldvel[:, 0])), stock_body)
        predicted_body = p.exp4_body.to_numpy()
        gated = np.where(p.exp4_body_applied.eq(1), predicted_body, stock_body)
        raw_body = p.body_rel.notna().to_numpy()
        near_ball = norm(p.tru_pos_x - p.cycle.map(ball.tru_pos_x), p.tru_pos_y - p.cycle.map(ball.tru_pos_y)) < 2
        groups = {"all": np.ones(len(p), bool), "raw_motion": p.dist_chg.notna().to_numpy(),
                  "no_raw_motion": p.dist_chg.isna().to_numpy(), "raw_body": raw_body, "no_raw_body": ~raw_body,
                  "stock_velocity_valid": valid_vel, "stock_body_valid": ~missing_body, "near_ball": near_ball,
                  "under_10m": p.true_dist.lt(10).to_numpy(), "10_to_40m": p.true_dist.between(10, 40, inclusive="left").to_numpy(),
                  "at_least_40m": p.true_dist.ge(40).to_numpy()}
        for name, mask in groups.items():
            add(stats, "player_velocity/" + name, np.linalg.norm(oldvel[mask] - truthvel[mask], axis=1),
                np.linalg.norm(newvel[mask] - truthvel[mask], axis=1))
            add(stats, "player_body/" + name, angle_error(stock_body[mask], p.tru_body.to_numpy()[mask]),
                angle_error(gated[mask], p.tru_body.to_numpy()[mask]), .1)
        counters["player_velocity_mismatch"] = int((np.linalg.norm(newvel - vector(p, "trk_vel"), axis=1) > .0002).sum())
        counters["player_body_mismatch"] = int((angle_error(gated, p.trk_body) > .0002).sum())
    event = {}
    if len(b):
        for metric, oldprefix, newprefix, truthprefix in [
            ("position", "exp4_stock_pos", "exp4_pos", "tru_pos"),
            ("velocity", "exp4_stock_vel", "exp4_vel", "tru_vel")]:
            truth = vector(b, truthprefix)
            old = vector(b, oldprefix)
            if metric == "velocity": old = np.nan_to_num(old)
            pair(stats, "ball_" + metric + "/all", b, old, vector(b, newprefix), truth)
            for key, mask in [("under_5m", b.true_dist.lt(5)), ("under_2m", b.true_dist.lt(2)),
                              ("at_least_20m", b.true_dist.ge(20)), ("observer_kickable", b.est_kickable.eq(1))]:
                pair(stats, "ball_" + metric + "/" + key, b, old[mask], vector(b, newprefix)[mask], truth[mask])
        counters["ball_position_mismatch"] = int((np.linalg.norm(vector(b,"exp4_pos") - vector(b,"trk_pos"), axis=1) > .0002).sum())
        counters["ball_velocity_mismatch"] = int((np.linalg.norm(vector(b,"exp4_vel") - vector(b,"trk_vel"), axis=1) > .0002).sum())
        # Shift by actual simulator cycle, never across a play-mode or stopped-cycle gap.
        nxt = ball.reindex(b.cycle.to_numpy() + 1)
        valid = nxt.truth_valid.eq(1).to_numpy() & nxt.tru_on_field.eq(1).to_numpy()
        current = b.iloc[np.flatnonzero(valid)]
        nxt = nxt.iloc[np.flatnonzero(valid)]
        truth_pos, truth_vel = vector(nxt, "tru_pos"), vector(nxt, "tru_vel")
        stock_pos = vector(current, "exp4_stock_pos"); stock_vel = np.nan_to_num(vector(current, "exp4_stock_vel"))
        pred_pos, pred_vel = vector(current, "exp4_next_pos"), vector(current, "exp4_next_vel")
        current_pos, current_vel = vector(current, "exp4_pos"), vector(current, "exp4_vel")
        decay = float(meta["server_param"]["ball_decay"])
        coast_pos, coast_vel = stock_pos + stock_vel, decay * stock_vel
        updated_coast = current_pos + current_vel
        impulse = np.linalg.norm(truth_vel - decay * vector(current, "tru_vel"), axis=1) > .2
        # Current kickability is the cue the live agent uses. Geometric truth is evaluation only.
        control = current.est_kickable.eq(1).to_numpy()
        player_min = player.assign(ball_dist=norm(player.exp4_stock_pos_x-player.cycle.map(ball.exp4_stock_pos_x),
                                                   player.exp4_stock_pos_y-player.cycle.map(ball.exp4_stock_pos_y))).groupby("cycle").ball_dist.min()
        control |= current.cycle.map(player_min).le(float(meta["server_param"]["player_size"])+float(meta["server_param"]["kickable_margin"])+float(meta["server_param"]["ball_size"])).to_numpy()
        groups = {"all": np.ones(len(current),bool), "next_impulse": impulse, "next_coasting": ~impulse,
                  "estimated_control": control, "estimated_free": ~control, "under_5m": current.true_dist.lt(5).to_numpy(),
                  "observer_kickable": current.est_kickable.eq(1).to_numpy(),
                  "observer_kicks": current.body_cmd.eq("kick").to_numpy()}
        for name, mask in groups.items():
            pair(stats,"ball_next_position/"+name,current,coast_pos[mask],pred_pos[mask],truth_pos[mask])
            pair(stats,"ball_next_velocity/"+name,current,coast_vel[mask],pred_vel[mask],truth_vel[mask])
            pair(stats,"forecast_vs_updated_coast/"+name,current,updated_coast[mask],pred_pos[mask],truth_pos[mask])
            pair(stats,"forecast_vs_updated_decay/"+name,current,(decay*current_vel)[mask],pred_vel[mask],truth_vel[mask])
        seam = np.linalg.norm(pred_pos - updated_coast,axis=1)
        add(stats,"trajectory_separation_next",np.zeros(len(seam)),seam)
        separation5 = pred_pos + (1-decay**4)/(1-decay)*pred_vel - (current_pos+(1-decay**5)/(1-decay)*current_vel)
        add(stats,"trajectory_separation_5",np.zeros(len(seam)),np.linalg.norm(separation5,axis=1))
        predicted_impulse = np.linalg.norm(pred_vel - decay * current_vel,axis=1) > .2
        event = dict(n=len(current), true_impulse=int(impulse.sum()), predicted_impulse=int(predicted_impulse.sum()),
                     correct_impulse=int((predicted_impulse & impulse).sum()))
    counters["exp4_nonfinite"] = int((~np.isfinite(candidates.exp4_vel_x)).sum())
    return path.parent.name, path.name[0], counters, stats, latency, event


def quantile(h, q, width):
    return (np.searchsorted(h.cumsum(),q*h.sum())+.5)*width if h.sum() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("game_logs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers",type=int,default=20)
    parser.add_argument("--observers",type=int,default=11)
    args=parser.parse_args()
    jobs=[]; configs=Counter(); problems=[]; results={}; modes=Counter()
    scores={}; steps=[]
    expected={"full_view":"true","see_no_fade_unum":"true","see_no_fade_all":"false",
              "see_noise_free":"false","say_msg_size":"0","quantize_step":"0.1"}
    for game in sorted(args.logs.iterdir()):
        if not game.is_dir():continue
        path = args.game_logs / "result.d" / game.name
        matches = re.findall(r"Score: (\d+) - (\d+)", path.read_text()) if path.exists() else []
        if not matches:
            continue
        files=sorted(game.glob("*/meta.json"))
        if len(files)!=22:problems.append([game.name,"observer count",len(files)])
        for p in files:
            m=json.loads(p.read_text());c=m["config"]
            if not (m["final"] and m["status"]=="ok" and not c["use_fullstate"] and c["debug_fullstate"] and not c["gaussian_see"] and all(m["server_param"].get(k)==v for k,v in expected.items())):problems.append(str(p))
            modes[m["side"],c.get("exp4_state"),c.get("player_position_routing")]+=1
            if m["unum"]<=args.observers:jobs.append(p.parent)
        path=args.game_logs/"result.d"/game.name
        if path.exists():
            matches=re.findall(r"Score: (\d+) - (\d+)",path.read_text())
            if matches:scores[game.name]=list(map(int,matches[-1]))
    print("metadata checked",len(jobs),"modes",modes,"issues",len(problems),flush=True)
    counts={s:Counter() for s in "lr"}; summaries={}; lat={s:np.zeros(2001,np.int64) for s in "lr"}; events=Counter();per_game={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        running={pool.submit(read_observer,p):p for p in jobs}
        for i,future in enumerate(as_completed(running),1):
            game,side,c,stat,l,event=future.result()
            counts[side].update(c);lat[side]+=l
            if event:events.update(event)
            for key,(s,h,w) in stat.items():
                k=side+"/"+key
                if k not in summaries:summaries[k]=[np.zeros(7),np.zeros((2,SIZE),np.int64),w]
                summaries[k][0]+=s;summaries[k][1]+=h
                if key in ("player_position_relative","player_velocity/all","player_body/all","ball_position/all","ball_velocity/all","ball_next_position/all"):
                    per_game.setdefault(game,{}).setdefault(side,{}).setdefault(key,np.zeros(7))
                    per_game[game][side][key]+=s
            if i%110==0:print("audited",i,"/",len(jobs),flush=True)
            del running[future]
    out={}
    for key,(s,h,w) in summaries.items():
        n=s[0]
        out[key]=dict(n=int(n),mean_before=float(s[1]/n),mean_after=float(s[2]/n),
                      p90_before=quantile(h[0],.9,w),p90_after=quantile(h[1],.9,w),
                      worse_share=float(s[3]/n),worse_10cm_share=float(s[4]/n),worse_50cm_share=float(s[5]/n),
                      worse_1m_share=float(s[6]/n))
    rng=np.random.default_rng(42)
    match={}
    if scores:
        a=np.array(list(scores.values()));d=a[:,0]-a[:,1];draw=rng.integers(len(d),size=(20000,len(d)))
        match=dict(n=len(d),goals=a.sum(axis=0).tolist(),wdl=[int((d>0).sum()),int((d==0).sum()),int((d<0).sum())],
                   goal_difference=float(d.mean()),goal_ci95=np.quantile(d[draw].mean(axis=1),[.025,.975]).tolist(),
                   point_ci95=np.quantile((3*np.sign(d))[draw].mean(axis=1),[.025,.975]).tolist(),
                   decisive_win_share=float((d>0).sum()/max(1,(d!=0).sum())))
    serial={g:{side:{k:v.tolist() for k,v in metrics.items()} for side,metrics in ss.items()} for g,ss in per_game.items()}
    result=dict(metrics=out,counts=counts,metadata_issues=problems,modes={str(k):v for k,v in modes.items()},
                latency_ms={s:[quantile(h,q,1) for q in (.5,.95,.99)] for s,h in lat.items()},event=events,matches=match,per_game=serial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("Wrote",args.output,flush=True)

if __name__=="__main__":
    main()
