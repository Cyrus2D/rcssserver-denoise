// Regression checks for decision-only state overrides and guarded application.
// Compile with -fno-access-control: white-box setup without a production test API.
#include <rcsc/player/world_model.h>
#include <rcsc/common/player_type.h>
#include <rcsc/common/server_param.h>
#include <rcsc/game_mode.h>
#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    using namespace rcsc;
    auto & sp = ServerParam::instance();
    sp.M_full_view = true;
    sp.M_see_no_fade_unum = true;
    sp.M_see_no_fade_all = sp.M_see_noise_free = false;
    sp.M_player_say_msg_size = 0;
    WorldModel wm;
    PlayerType type;
    wm.M_our_side = LEFT;
    wm.M_client_version = 19;
    wm.M_self.M_player_type = &type;
    wm.M_self.M_unum = 2;
    wm.M_self.M_pos.assign( 0, 0 );
    wm.M_self.M_vel.assign( 0, 0 );
    wm.M_self.M_pos_count = wm.M_self.M_vel_count = wm.M_self.M_face_count = 0;
    wm.M_game_mode.update( "play_on", GameTime(1,0) );
    wm.M_ball.updateAll( {20,0}, {.1,.1}, 0, {20,0}, {.1,.1}, {.3,0}, {.01,.01}, 0 );
    wm.M_teammates.emplace_back();
    auto & p = wm.M_teammates.back();
    p.M_side = LEFT; p.M_unum = 3; p.M_pos.assign( 18, 0 );
    p.M_pos_count = p.M_seen_pos_count = 0;
    p.M_vel.assign( .11, .12 ); p.M_vel_count = 1000;
    p.M_body = 123; p.M_body_count = 1000;
    wm.M_state_raw[0].seen = wm.M_state_raw[3].seen = true;
    wm.M_state_raw[0].distance = 20;
    wm.M_state_raw[3].distance = 18;
    StateDenoiseOptions options;
    options.apply_player = options.apply_ball = true;
    int calls = 0;
    wm.setStateModel( [&]( const StateModelInput & in ) {
        ++calls;
        if ( in.ball ) return std::array<float,8>{22,0,.8f,0,45,0,2,0};
        // Missing stock fields must remain missing despite earlier corrections.
        assert( in.features.back()[10] == 0 && in.features.back()[11] == 0 );
        for ( const auto & f : in.features ) assert( f[10] == 0 && f[11] == 0 );
        return std::array<float,8>{.6f,.2f,0,1,0,0,0,0};
    }, options );
    auto cycle = [&]( int n ) {
        wm.M_time = wm.M_see_player_record_time = GameTime(n,0);
        wm.updateDenoisedState( wm.M_time );
    };
    for ( int n = 1; n <= 7; ++n ) cycle(n);
    assert( calls == 0 );
    cycle(8);
    assert( calls == 2 && wm.exp4Record(0).pos_applied && wm.exp4Record(3).body_applied );
    assert( std::abs( wm.ball().vel().x - .8 ) < 1e-6 && std::abs(p.vel().x-.6)<1e-6 );
    assert( std::abs( p.body().degree()-90 ) < 1e-4 && p.bodyValid() && p.velValid() );
    for ( int n : {1,2,5,20} )
        assert( wm.ball().inertiaPoint(n).dist(inertia_n_step_point(wm.ball().pos(),wm.ball().vel(),n,sp.ballDecay())) < 1e-10 );
    assert( wm.ball().inertiaPoint(1).dist(wm.exp4Record(0).next_pos) > 10 );
    wm.restoreStateModelObservations();
    assert( wm.ball().pos().dist({20,0}) < 1e-12 && std::abs(wm.ball().vel().x-.3)<1e-12 );
    assert( p.vel().dist({.11,.12})<1e-12 && p.velCount()==1000 );
    assert( p.body().degree()==123 && p.bodyCount()==1000 && !p.M_decision_velocity_corrected );
    cycle(9);
    wm.restoreStateModelObservations();
    // Protect near-ball state and near-player velocity, but infer missing body.
    wm.M_state_raw[0].distance = 3;
    wm.M_state_raw[3].distance = 5;
    cycle(10);
    assert( !wm.exp4Record(0).pos_applied && !wm.exp4Record(0).vel_applied );
    assert( wm.ball().pos().dist({20,0}) < 1e-12 && std::abs(wm.ball().vel().x-.3)<1e-12 );
    assert( !wm.exp4Record(3).vel_applied && wm.exp4Record(3).body_applied );
    wm.restoreStateModelObservations();
    // Reported body direction is preserved.
    wm.M_state_raw[3].raw_body = true;
    cycle(11);
    assert( !wm.exp4Record(3).body_applied && p.body().degree()==123 );
    // A missing cycle or unsupported sensor mode invalidates the whole window.
    cycle(13);
    assert( !wm.M_state_denoiser->ready(0) );
    sp.M_see_noise_free = true;
    cycle(14);
    assert( !wm.M_state_denoiser->ready(0) );
    sp.M_see_noise_free = false;
    // Reset mode/component flags, then check shadow never changes decision state.
    options.apply_player = options.apply_ball = false;
    wm.setStateModel( [](const StateModelInput &) { return std::array<float,8>{30,0,1,0,50,0,2,0}; }, options );
    for ( int n=15;n<=22;++n ) cycle(n);
    assert( !wm.exp4Record(0).pos_applied && !wm.exp4Record(0).vel_applied && !wm.exp4Record(3).body_applied );
    assert( wm.ball().pos().dist({20,0}) < 1e-12 && p.body().degree()==123 );
    // Disabling one group suppresses its candidates.
    options.player_enabled=false; options.apply_ball=true;
    wm.setStateModel( [](const StateModelInput &) { return std::array<float,8>{30,0,1,0,50,0,2,0}; }, options );
    for ( int n=23;n<=30;++n ) cycle(n);
    assert( !wm.exp4Record(3).vel.isValid() && wm.exp4Record(0).vel.isValid() );
    wm.M_state_raw[0].distance = 20;
    cycle(31);
    assert( wm.exp4Record(0).pos_applied && wm.ball().pos().dist({30,0})<1e-12 );
    GameMode setplay;
    setplay.update( "kick_off_l", GameTime(31,0) );
    wm.updateGameMode( setplay, GameTime(31,0) );
    assert( wm.ball().pos().dist({20,0})<1e-12 && !wm.M_state_denoiser->ready(0) );
    cycle(32);
    assert( !wm.exp4Record(0).vel.isValid() );
    GameMode playon;
    playon.update( "play_on", GameTime(33,0) );
    wm.updateGameMode( playon, GameTime(33,0) );
    for (int n=33;n<=39;++n) cycle(n);
    assert( !wm.exp4Record(0).vel.isValid() );
    cycle(40);
    assert( wm.exp4Record(0).vel.isValid() );
    wm.restoreStateModelObservations();
    std::cout << "PASS: observation restoration, input masks, physical path, range gates, body preservation, gaps, sensor gate, shadow, component modes and setplay restarts\n";
}
