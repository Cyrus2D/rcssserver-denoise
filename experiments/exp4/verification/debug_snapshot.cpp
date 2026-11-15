// Test-only private setup; production snapshot/serialization code is unchanged.
#include <rcsc/player/world_model.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/common/player_type.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char **argv) {
    assert(argc==2);
    using namespace rcsc;
    WorldModel wm;
    PlayerType type;
    wm.M_our_side=LEFT;wm.M_self.M_unum=6;wm.M_self.M_player_type=&type;
    wm.M_self.M_pos={0,0};wm.M_self.M_vel={0,0};
    wm.M_self.M_pos_count=wm.M_self.M_vel_count=wm.M_self.M_face_count=0;
    wm.M_time=GameTime(8,0);wm.M_game_mode.update("play_on",wm.M_time);
    wm.M_ball.updateAll({10,2},{.1,.1},0,{10,2},{.1,.1},{.3,.4},{.01,.01},0);
    wm.M_opponents.emplace_back();auto & p=wm.M_opponents.back();
    p.M_side=RIGHT;p.M_unum=7;p.M_pos={12,3};p.M_pos_count=0;
    p.M_vel={.1,.2};p.M_vel_count=1000;p.M_body=123;p.M_body_count=1000;
    wm.captureDenoiseDebugStock();
    assert(wm.denoiseDebugTime()==wm.time());
    assert(wm.denoiseDebugStock(0).pos.dist({10,2})<1e-12);
    assert(wm.denoiseDebugStock(18).pos.dist({12,3})<1e-12);
    assert(!wm.denoiseDebugStock(18).vel.isValid()&&!wm.denoiseDebugStock(18).body_valid);
    p.M_pos={13,4};wm.M_ball.M_pos={11,2}; // simulated later decision overrides
    assert(wm.denoiseDebugStock(0).pos.dist({10,2})<1e-12);
    assert(wm.denoiseDebugStock(18).pos.dist({12,3})<1e-12);
    assert(wm.ball().pos().dist({11,2})<1e-12&&p.pos().dist({13,4})<1e-12);
    wm.M_opponents_from_self.push_back(&p);
    DebugClient dc;
    assert(dc.open(argv[1],"PacketFixture",6));
    dc.setMaxPacketSize(7800);
    dc.addMessage(std::string(20000,'x'));
    dc.addComment(&p,"%s",std::string(20000,'z').c_str());
    for(int i=0;i<50;++i){dc.addLine({double(i),0},{double(i),1},"#888888");dc.addCircle({double(i),0},.3,"#00bfff");}
    class Dummy : public PlayerAgent { void actionImpl() override {} };
    Dummy dummy;
    dc.writeAll(wm,dummy.effector());
    std::ifstream f(std::string(argv[1])+"/PacketFixture-6.dcl");std::string line;
    std::getline(f,line);std::getline(f,line);
    assert(line.size()<7900&&line.find("CLIPPED")!=std::string::npos);
    std::cout<<"PASS: immutable stock snapshot, invalid-field masks, identity mapping, bounded complete packet\n";
}
