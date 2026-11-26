#!/usr/bin/env python3
"""Train causal residual GRUs and compare their estimates with physical baselines."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED = 60625
METHODS = ("visual", "kalman", "kalman_guard", "gru", "gru_guard")


class ResidualGRU(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.history = nn.GRU(features, hidden, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, features):
        out, _ = self.history(features)
        delta = self.readout(out[:, -1])
        # Learn both how much of the physical correction to trust and an
        # additional residual. All inputs come from the current/past windows.
        physical = features[:, -1, 9:11]
        trust = torch.sigmoid(delta[:, :1] + 1.4)
        return trust * physical + 2.0 * torch.tanh(delta[:, 1:])


def guard(delta, observed_distance):
    norm = delta.norm(dim=1, keepdim=True).clamp_min(1.e-8)
    corrected = delta * (.25/norm).clamp(max=1.)
    return torch.where(observed_distance[:,None] >= 5., corrected, 0.)


class Data:
    def __init__(self, folder):
        self.folder = folder
        self.meta = json.loads((folder/"manifest.json").read_text())
        self.width = self.meta["width"]
        self.history = self.meta["history"]
        self.features = len(self.meta["features"])
        self.offset = self.history*self.features

    def read(self,record,target):
        info = record["data"][target]
        return np.memmap(self.folder/info["path"],dtype="<f4",mode="r",shape=(info["windows"],self.width))

    def records(self,split):
        return [r for r in self.meta["records"] if r["split"]==split]

    def unpack(self,tensor):
        features = tensor[:,:self.offset].reshape(-1,self.history,self.features).contiguous()
        return features, tensor[:,self.offset:]


def load_samples(data,target,split,count,rng,whole):
    records = data.records(split)
    if whole:
        return np.concatenate([data.read(r,target) for r in records])
    goal = int(np.ceil(count/len(records)))
    parts = []
    for r in records:
        rows = data.read(r,target)
        indices = rng.choice(len(rows),size=min(goal,len(rows)),replace=False)
        parts.append(rows[indices])
    joined = np.concatenate(parts)
    rng.shuffle(joined)
    return joined[:count]


@torch.inference_mode()
def measure_validation(data, model, batches, device):
    model.eval()
    totals = np.zeros(3,dtype=np.float64)
    for batch in batches:
        x,y = data.unpack(batch.to(device))
        prediction = model(x)
        target = y[:,:2]
        near = y[:,3] < 50.
        bounded = (guard(prediction,y[:,2])-target).norm(dim=1)
        unbounded = (prediction-target).norm(dim=1)
        totals += torch.stack((bounded[near].sum(),unbounded[near].sum(),near.sum())).cpu().numpy()
    return dict(guard_cm=float(100*totals[0]/totals[2]),raw_cm=float(100*totals[1]/totals[2]))


def train(data, target, args, device):
    torch.manual_seed(SEED)
    model = ResidualGRU(data.features,args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.0001)
    rng = np.random.default_rng(SEED)
    valid_rows = load_samples(data,target,"validation",120000,rng,False)
    validation = torch.from_numpy(valid_rows).split(args.batch)
    limit = args.player_windows if target=="player" else args.ball_windows
    best, best_epoch = np.inf, None
    state_path = args.output/(target+".pt")
    log = []
    for epoch in range(1,args.epochs+1):
        samples = load_samples(data,target,"train",limit,rng,False)
        model.train()
        loss_sum = torch.zeros((),device=device)
        for batch in torch.from_numpy(samples).split(args.batch):
            x,y = data.unpack(batch.to(device))
            target_delta = y[:,:2]
            predicted = model(x)
            raw_squared = (predicted-target_delta).square().sum(dim=1)
            capped = guard(predicted,y[:,2])
            cap_squared = (capped-target_delta).square().sum(dim=1)
            # Give close and medium ranges weight while retaining far examples.
            weights = (1.5/(.65+.025*y[:,2])).clamp(.45,2.)
            loss = (weights*(torch.sqrt(raw_squared+.0001)+.3*torch.sqrt(cap_squared+.0001))).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            optim.step()
            loss_sum += loss.detach()*len(batch)
        observed = measure_validation(data,model,validation,device)
        observed.update(epoch=epoch,train_windows=len(samples),loss=float(loss_sum.cpu())/len(samples))
        log.append(observed)
        criterion = observed["guard_cm"]+.20*observed["raw_cm"]
        if criterion<best:
            best,best_epoch = criterion,epoch
            checkpoint = dict(model=model.state_dict(),features=data.features,hidden=args.hidden,
                              history=data.history,target=target,epoch=epoch,validation=observed)
            torch.save(checkpoint,state_path)
        print(target,epoch,{k:round(v,4) for k,v in observed.items()},flush=True)
    model.load_state_dict(torch.load(state_path,map_location=device,weights_only=True)["model"])
    return model,dict(best_epoch=best_epoch,history=log,parameters=sum(p.numel() for p in model.parameters()))


class Totals:
    def __init__(self):
        self.n = 0
        self.error = 0.
        self.baseline = 0.
        self.better = 0
        self.worse25 = 0
        self.worse50 = 0
        self.max_added = 0.

    def add(self,error,baseline):
        if not len(error):
            return
        delta = error-baseline
        self.n += len(error)
        self.error += float(error.sum(dtype=np.float64))
        self.baseline += float(baseline.sum(dtype=np.float64))
        self.better += int((delta<-.01).sum())
        self.worse25 += int((delta>.250002).sum())
        self.worse50 += int((delta>.500002).sum())
        self.max_added = max(self.max_added,float(delta.max()))

    def report(self):
        return dict(n=self.n,error_cm=100*self.error/self.n,baseline_cm=100*self.baseline/self.n,
                    improved_1cm=self.better/self.n,worsened_25cm=self.worse25/self.n,
                    worsened_50cm=self.worse50/self.n,max_added_cm=100*self.max_added)


@torch.inference_mode()
def evaluate(data,model,target,split,device,batch_size):
    totals = defaultdict(Totals)
    games = defaultdict(lambda:np.zeros(3,dtype=np.float64))
    model.eval()
    for record in data.records(split):
        rows = data.read(record,target)
        for i in range(0,len(rows),batch_size):
            x,y = data.unpack(torch.from_numpy(np.array(rows[i:i+batch_size])).to(device))
            truth = y[:,:2]
            physical = y[:,4:6]
            learned = model(x)
            methods = (torch.zeros_like(learned),physical,guard(physical,y[:,2]),learned,guard(learned,y[:,2]))
            scores = torch.stack([(delta-truth).norm(dim=1) for delta in methods]).cpu().numpy()
            distance = y[:,3].cpu().numpy()
            groups = {"all":np.ones(len(distance),bool),"under50":distance<50.,"under3":distance<3.,
                      "3-10":(distance>=3.) & (distance<10.),"10-20":(distance>=10.) & (distance<20.),
                      "20-35":(distance>=20.) & (distance<35.),"35-50":(distance>=35.) & (distance<50.)}
            for group,mask in groups.items():
                for method,errors in zip(METHODS,scores):
                    totals[group,method].add(errors[mask],scores[0,mask])
            near = groups["under50"]
            games[record["game"]] += [int(near.sum()),scores[2,near].sum(dtype=np.float64),
                                      scores[4,near].sum(dtype=np.float64)]
    rng = np.random.default_rng(SEED)
    games = np.array(list(games.values()))
    sample = games[rng.integers(len(games),size=(5000,len(games)))].sum(axis=1)
    difference = 100*(sample[:,2]-sample[:,1])/sample[:,0]
    return dict(rows=[dict(distance=group,method=method,**metric.report())
                      for (group,method),metric in sorted(totals.items()) if metric.n],
                guarded_gru_minus_guarded_kalman95_cm=np.quantile(difference,[.025,.975]).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--batch",type=int,default=1024)
    p.add_argument("--hidden",type=int,default=40)
    p.add_argument("--player-windows",type=int,default=220000)
    p.add_argument("--ball-windows",type=int,default=120000)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device)
    torch.set_num_threads(4)
    data = Data(args.data)
    result = dict(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                  seed=SEED,history=data.history,features=data.meta["features"],
                  config={k:v for k,v in vars(args).items() if k not in ("data","output")},
                  data_counts=data.meta["counts"],split=data.meta["split"],targets={})
    for target in ("player","ball"):
        model,history = train(data,target,args,device)
        result["targets"][target] = dict(training=history,**{
            split:evaluate(data,model,target,split,device,args.batch*4)
            for split in ("fresh_test","previous_test")})
        (args.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
        print("Evaluated",target,flush=True)


if __name__ == "__main__":
    main()
