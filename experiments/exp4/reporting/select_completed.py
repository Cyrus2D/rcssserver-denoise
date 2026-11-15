#!/usr/bin/env python3
"""Freeze scored games whose 22 agent logs have all finalized successfully."""
import argparse,json,re
from pathlib import Path
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs",type=Path);p.add_argument("game_logs",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,default=0)
    a=p.parse_args();selected=[];skipped=[]
    for game in sorted(a.logs.iterdir()):
        if not game.is_dir():continue
        scorefile=a.game_logs/"result.d"/game.name
        scores=re.findall(r"Score: (\d+) - (\d+)",scorefile.read_text())if scorefile.exists()else[]
        files=list(game.glob("*/meta.json"))
        if not scores or len(files)!=22:skipped.append(game.name);continue
        meta=[json.loads(f.read_text())for f in files]
        if not all(m.get("final")and m.get("status")=="ok"for m in meta):skipped.append(game.name);continue
        selected.append([game.name,list(map(int,scores[-1]))])
    if a.limit:selected=selected[:a.limit]
    if not selected:raise ValueError("No scored, fully finalized games")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError("Use a new snapshot filename: "+str(a.output))
    a.output.write_text(json.dumps(selected,indent=2)+"\n")
    print("Frozen",len(selected),"games; incomplete/unscored directories",len(skipped))
