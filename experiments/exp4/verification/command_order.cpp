// White-box regression against production parser, handlers, shuffle and physics.
// -fno-access-control is used only by this fixture, never by the server build.
#include "stadium.h"
#include "player.h"
#include "team.h"
#include "serverparam.h"
#include "playerparam.h"
#include "random.h"
#include <rcss/net/addr.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

static void parse(Player * p, const std::string & text)
{
    std::vector<char> message(text.begin(),text.end());
    message.push_back(0); p->parseMsg(message.data(),message.size());
}
static int peer(rcss::net::Addr & address)
{
    int fd=::socket(AF_INET,SOCK_DGRAM,0);assert(fd>=0);
    sockaddr_in local{};local.sin_family=AF_INET;local.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    assert(::bind(fd,reinterpret_cast<sockaddr*>(&local),sizeof(local))==0);
    socklen_t size=sizeof(local);assert(::getsockname(fd,reinterpret_cast<sockaddr*>(&local),&size)==0);
    address.setHost("127.0.0.1");address.setPort(ntohs(local.sin_port));return fd;
}
int main(int argc,char **argv)
{
    assert(argc==2);
    std::filesystem::create_directories(argv[1]);
    std::vector<std::string> options={"command-order-test","server::port=0","server::coach_port=0","server::olcoach_port=0",
        "server::synch_mode=true","server::randomize_player_commands=true","server::command_order_seed=42",
        "server::random_seed=12345","player::random_seed=54321","player::player_types=1",
        "server::game_logging=false","server::text_logging=false","server::game_log_fixed=true",
        "server::game_log_fixed_name=fixture","server::game_log_dir="+std::string(argv[1]),
        "server::foul_detect_probability=0","server::foul_cycles=1"};
    std::vector<char*> args;for(auto & s:options)args.push_back(s.data());
    assert(ServerParam::init(args.size(),args.data()));
    Stadium stadium;assert(stadium.init());
    rcss::net::Addr a,b;int pa=peer(a),pb=peer(b);
    Player * left=stadium.initPlayer("QueueLeft",19,false,a);
    Player * right=stadium.initPlayer("QueueRight",19,false,b);
    assert(left&&right);
    auto objects=stadium.M_movable_objects;
    auto send_wire=[&](Player * p,const std::string & text) {
        auto destination=p->M_socket.getName();destination.setHost("127.0.0.1");
        int fd=p==left?pa:pb;
        assert(::sendto(fd,text.c_str(),text.size()+1,0,
                       reinterpret_cast<const sockaddr*>(&destination.getAddr()),sizeof(sockaddr_in))>0);
        for(int n=0;n<100&&!p->doneReceived();++n) {
            stadium.doRecvFromClients();
            if(!p->doneReceived())std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        assert(p->doneReceived());
    };
    auto reset=[&] {
        stadium.changePlayMode(PM_BeforeKickOff);
        stadium.changePlayMode(PM_PlayOn);
        stadium.M_time=321;stadium.M_stoppage_time=0;
        stadium.M_movable_objects=objects;
        stadium.moveBall(PVector(0,0),PVector(0,0));
        PVector zero(0,0);double lface=0,rface=M_PI;
        left->place(PVector(-.7,0),lface,zero,zero);
        right->place(PVector(.8,0),rface,zero,zero);
        for(auto * p:{left,right}) {
            p->discardQueuedCommands();p->M_command_done=false;p->M_turn_neck_done=false;
            p->M_done_received=false;p->M_foul_cycles=0;p->M_tackle_cycles=0;p->M_kick_cycles=-1;
            p->M_kick_count=p->M_tackle_count=p->M_turn_count=p->M_turn_neck_count=0;
            p->M_angle_neck=p->M_angle_neck_committed=0;
            p->beginCommandCollection();
        }
        DefaultRNG::seed(98765);
    };
    auto run=[&](bool left_first,int polls,bool queued) {
        reset();ServerParam::instance().M_randomize_player_commands=queued;
        auto first=left_first?left:right;auto second=left_first?right:left;
        const std::string kick="(kick 40 0)(turn_neck 5)(done)";
        const std::string foul="(tackle 0 on)(turn_neck -5)(done)";
        send_wire(first,first==left?kick:foul);
        if(queued)assert(left->M_kick_count==0&&right->M_tackle_count==0);
        for(int i=0;i<polls;++i)stadium.doRecvFromClients();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        send_wire(second,second==left?kick:foul);
        if(queued)assert(stadium.applyQueuedPlayerCommands());
        // Real movement/noise/collision functions, with non-zero default noise.
        stadium.incMovableObjects();
        return std::array<double,8>{stadium.ball().pos().x,stadium.ball().pos().y,
            stadium.ball().vel().x,stadium.ball().vel().y,double(left->M_kick_count),double(right->M_tackle_count),
            left->angleNeckCommitted(),right->angleNeckCommitted()};
    };
    int kick_first=0,foul_first=0;
    for(int seed=0;seed<64;++seed) {
        ServerParam::instance().setCommandOrderSeed(seed);
        auto x=run(true,0,true),y=run(false,43,true);
        assert(x==y); // Arrival order, extra receive polling and delays cannot change results.
        if(x[4]==1)++kick_first;else ++foul_first;
    }
    assert(kick_first>0&&foul_first>0);
    auto early=run(true,0,false),late=run(false,0,false);
    assert(early[4]==1&&late[4]==0); // Legacy mode retains the original order-dependent behavior.
    ServerParam::instance().M_randomize_player_commands=true;
    reset();
    parse(left,"(turn 30)(kick 50 0)(turn_neck 10)(turn_neck 20)(done)");
    parse(right,"(done)");
    assert(stadium.applyQueuedPlayerCommands());
    assert(left->M_turn_count==1&&left->M_kick_count==0&&left->M_turn_neck_count==1);
    reset();
    parse(left,"(done)(kick 100 0)");
    parse(right,"(done)");
    assert(left->queuedCommandCount()==0&&stadium.applyQueuedPlayerCommands()&&left->M_kick_count==0);
    reset();
    parse(left,"(kick 50 0)(done)");
    parse(right,"(tackle 0 on)"); // No done: transactional commit must reject the whole round.
    assert(!stadium.applyQueuedPlayerCommands()&&left->M_kick_count==0&&right->M_tackle_count==0);
    assert(left->queuedCommandCount()==0&&right->queuedCommandCount()==0);
    reset();
    std::string flood;for(int i=0;i<129;++i)flood+="(turn_neck 1)";flood+="(done)";
    parse(left,flood);parse(right,"(done)");
    assert(!stadium.applyQueuedPlayerCommands()&&left->M_turn_neck_count==0);
    reset();
    parse(left,"(kick nope)(done)");parse(right,"(done)");
    assert(left->commandQueueFailed()&&!stadium.applyQueuedPlayerCommands());
    ::close(pa);::close(pb);
    std::cout<<"REGRESSION PASS: 64 seeds x reversed UDP arrival; exact noisy physics; polling isolation; both priorities; legacy; command order; post-done drop; incomplete batch; overflow; malformed input. kick-first="
             <<kick_first<<" foul-first="<<foul_first<<std::endl;
}
