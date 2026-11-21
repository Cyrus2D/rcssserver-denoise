"""Regression for causal baselines, PlayOn resets and missing-velocity fallback."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
import pandas as pd
spec=importlib.util.spec_from_file_location("exp5_analysis",Path(__file__).with_name("analyze.py"))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class BaselineTest(unittest.TestCase):
    def run_fixture(self,future=2.,missing_velocity=False,third_seen=True,stopped_interruption=False):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)
            (p/"meta.json").write_text(json.dumps(dict(config=dict(use_fullstate=False),server_param=dict(
                full_view="false",say_msg_size="0",see_no_quantization="true",see_no_fade_all="true",player_decay=".4"))))
            agents=[];objects=[]
            for cycle in range(1,7):
                agents.append(dict(cycle=cycle,stopped=0,decision_made=1,game_mode="before_kick_off"if cycle==4 and not stopped_interruption else "play_on",
                    see_used=int(cycle in [1,3,6]),view_width="normal",sb_view_width="normal",n_est_pos_d=0,n_est_vel_d=0,n_est_face=0))
                seen=cycle in ([1,3,6]if third_seen else [1,6])
                truth={1:0.,2:1.,3:future,4:100.,5:100.,6:100.}[cycle]
                position=truth if seen else float("nan")
                velocity=float("nan")if missing_velocity else 1.
                row=dict(cycle=cycle,stopped=0,object_id=2,truth_valid=1,tru_on_field=1,seen=int(seen),
                    unum_visible=int(seen),match_ambiguous=0,true_dist=20.,trk_pos_x=truth,trk_pos_y=0.,
                    trk_vel_x=velocity,trk_vel_y=0. if not missing_velocity else float("nan"),
                    ls_pos_x=position,ls_pos_y=0. if seen else float("nan"),ls_vel_x=velocity,
                    ls_vel_y=0. if not missing_velocity else float("nan"),tru_pos_x=truth,tru_pos_y=0.)
                for name in ["n_dist","n_dir","n_dist_chg","n_dir_chg","n_body_rel","n_ls_pos_d","n_ls_rpos_d"]:row[name]=0.
                objects.append(row)
                if stopped_interruption and cycle==3:
                    agents.append({**agents[-1],"stopped":1,"game_mode":"before_kick_off"})
            pd.DataFrame(agents).to_csv(p/"agent_cycle.csv.gz",index=False)
            pd.DataFrame(objects).to_csv(p/"objects.csv.gz",index=False)
            return audit.observer((p,False))
    def test_no_future_sighting_leak(self):
        first=self.run_fixture(2.)[2];second=self.run_fixture(99.)[2]
        for method in audit.METHODS:self.assertEqual(first[method+"/age/1"][:2],second[method+"/age/1"][:2])
        self.assertEqual(first["constant_velocity/age/1"][1],0.)
        self.assertEqual(first["hold/age/1"][1],1.)
    def test_mode_reset(self):
        _,counts,stats=self.run_fixture()
        self.assertEqual(counts["history_rows"],4) # cycle 5 cannot borrow cycle 3 across setplay
        self.assertNotIn("wm/age/2",stats)
    def test_stopped_time_setplay_resets_history(self):
        _,counts,stats=self.run_fixture(stopped_interruption=True)
        self.assertEqual(counts["history_rows"],4)
        self.assertEqual(stats["wm/age/1"][0],1)
    def test_missing_velocity_is_zero_fallback_not_truth(self):
        stats=self.run_fixture(missing_velocity=True)[2]
        self.assertEqual(stats["constant_velocity/age/1"][1],1.)
    def test_decay_and_constant_velocity(self):
        stats=self.run_fixture(third_seen=False)[2]
        self.assertAlmostEqual(stats["constant_velocity/age/2"][1],0.)
        self.assertAlmostEqual(stats["decay/age/2"][1],.6)
if __name__=="__main__":unittest.main()
