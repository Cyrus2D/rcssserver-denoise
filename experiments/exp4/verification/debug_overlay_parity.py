#!/usr/bin/env python3
"""Compare actual debug overlay numbers with same-cycle dataset fields."""
import argparse,json,re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("match",type=Path);p.add_argument("--side",choices=["l","r"],default="l")
    p.add_argument("--observer",type=int,default=6)
    a=p.parse_args()
    source=a.match/"dataset"/a.match.name/(a.side+"_"+str(a.observer))
    fields="""cycle stopped object_id trk_pos_x trk_pos_y trk_vel_x trk_vel_y trk_body
exp4_stock_pos_x exp4_stock_pos_y exp4_stock_vel_x exp4_stock_vel_y exp4_stock_body
tru_pos_x tru_pos_y tru_vel_x tru_vel_y tru_body""".split()
    rows=pd.read_csv(source/"objects.csv.gz",usecols=fields).set_index(["cycle","stopped","object_id"])
    factor=1 if a.side=="l" else -1
    counts=Counter();sizes=[];status=Counter()
    for line in (a.match/"debug-packets.txt").read_text().splitlines():
        assert len(line.encode())+1<=7800
        sizes.append(len(line.encode())+1)
        banner=re.search(r'\(message "([^"]*)"\)',line)
        assert banner and len(banner[1])<=1023 and "CLIPPED"not in banner[1]
        text=banner[1];cycle,stopped=map(int,re.search(r"\(time (\d+),(\d+)\)",line).groups())
        for s in re.findall(r"(?:pos|vel|body):([^ |]+)",text):status[s]+=1
        for match in re.finditer(r"(ball|tm\d+|opp\d+) (STOCK|USED|TRUTH) p=\(([-\d.]+),([-\d.]+)\) v=(n/a|\(([-\d.]+),([-\d.]+)\))(?: body=([-\d.]+|n/a))?",text):
            name,kind=match[1],match[2]
            if name=="ball":identity=0
            else:
                side=a.side if name.startswith("tm")else ("r" if a.side=="l"else "l")
                identity=int(re.search(r"\d+",name)[0])+(11 if side=="r"else 0)
            key=(cycle,stopped,identity)
            if key not in rows.index:counts["unlogged_object_rows"]+=1;continue
            row=rows.loc[key]
            prefix={"STOCK":"exp4_stock","USED":"trk","TRUTH":"tru"}[kind]
            for field,values in [("pos",[match[3],match[4]]),("vel",None if match[5]=="n/a" else [match[6],match[7]])]:
                if values is None:continue
                expected=row[[prefix+"_"+field+"_x",prefix+"_"+field+"_y"]].to_numpy(dtype=float)*factor
                if not np.isfinite(expected).all():continue
                actual=np.array(values,dtype=float)
                assert np.max(np.abs(expected-actual))<=.00056,(key,kind,field,expected,actual)
                counts[kind+"_"+field]+=1
            if match[8]and match[8]!="n/a":
                expected=float(row[prefix+"_body"])+(180 if factor==-1 else 0)
                if np.isfinite(expected):
                    diff=abs((float(match[8])-expected+180)%360-180)
                    assert diff<=.00056,(key,kind,"body",expected,match[8])
                    counts[kind+"_body"]+=1
    assert counts["USED_pos"]>100 and counts["STOCK_pos"]>100 and counts["TRUTH_pos"]>100
    data=dict(side=a.side,packets=len(sizes),max_packet_bytes=max(sizes),matched_fields=counts,status=status)
    (a.match/"debug-parity.json").write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))

if __name__=="__main__":main()
