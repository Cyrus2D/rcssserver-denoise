#include "csv_reader.h"
#include "kalman_state.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>

namespace {
using exp6::Csv;
namespace fs = std::filesystem;

using exp6::Vec;
using exp6::Kalman;

struct Candidate {
    const char *name;
    double accel;
    bool velocity;
};
constexpr Candidate filters[] = {
    {"kf_pos", .20, false},
    {"kf_vel_smooth", .06, true},
    {"kf_vel_medium", .18, true},
    {"kf_vel_quick", .45, true},
};
struct Track {
    long long cycle = -100;
    Vec last, velocity;
    bool velocity_known = false;
    std::array<Kalman, std::size(filters)> estimates;
};
struct Stat {
    uint64_t count = 0, better = 0, worse25 = 0, worse50 = 0;
    double raw = 0., error = 0., max_added = 0.;
    void add(double baseline, double prediction) {
        const double added = prediction-baseline;
        ++count;
        raw += baseline;
        error += prediction;
        better += added < -.01;
        worse25 += added > .25+1.e-9;
        worse50 += added > .5+1.e-9;
        max_added = std::max(max_added,added);
    }
};
struct ObserverFrame { double focus_range, focus_direction; };
constexpr double radians = 3.14159265358979323846/180.;
constexpr std::array<double,6> edges{3,10,20,30,40,60};
constexpr std::array<const char *,7> bands{"0-3","3-10","10-20","20-30","30-40","40-60","60+"};
using Results = std::map<std::tuple<std::string,std::string,std::string>,Stat>;

void score(Results &out, const std::string &target, const char *band, bool near,
           const std::string &method, double stock, double error) {
    out[{target,"all",method}].add(stock,error);
    out[{target,band,method}].add(stock,error);
    if (near) out[{target,"under50",method}].add(stock,error);
}

Vec clamp_correction(Vec candidate, Vec observed, double limit) {
    const Vec delta = candidate-observed;
    const double distance = delta.length();
    return distance>limit ? observed + delta*(limit/distance) : candidate;
}

void scan_observer(const fs::path &directory, Results &results, bool near_safe) {
    Csv agent(directory/"agent_cycle.csv.gz", {
        "cycle","stopped","game_mode","decision_made","focus_dist","focus_dir"});
    std::unordered_map<long long,ObserverFrame> play;
    while (agent.next()) {
        if (agent[2] != "play_on" || !agent.yes(3) || agent[1] != "0") continue;
        const double focus = agent.number(4), direction = agent.number(5);
        play.emplace(agent.key(),ObserverFrame{std::isfinite(focus)?focus:0.,
                                              std::isfinite(direction)?direction:0.});
    }
    Csv object(directory/"objects.csv.gz", {
        "cycle","stopped","object_id","seen","match_method","dist","dir",
        "dist_chg","dir_chg","ls_pos_x","ls_pos_y","ls_rpos_x","ls_rpos_y",
        "ls_vel_x","ls_vel_y","truth_valid","tru_on_field","tru_pos_x","tru_pos_y","true_dist"});
    std::array<Track,23> track{};
    while (object.next()) {
        const auto frame = play.find(object.key());
        if (frame == play.end() || !object.yes(3)) continue;
        const int id = static_cast<int>(object.number(2));
        if (id<0 || id>=static_cast<int>(track.size())) throw std::runtime_error("invalid object id");
        const bool ball = id==0;
        if (!ball && object[4] != "unum") continue;
        const Vec observed{object.number(9),object.number(10)};
        const Vec relative{object.number(11),object.number(12)};
        const double distance = object.number(5);
        if (!observed.valid() || !relative.valid() || !std::isfinite(distance)) continue;
        const double norm = relative.length();
        const Vec radial = norm>0.0001 ? relative*(1./norm) : Vec{1.,0.};
        const Vec tangent{-radial.y,radial.x};
        const Vec velocity{object.number(13),object.number(14)};
        const bool vel_known = velocity.valid() && !object[7].empty() && !object[8].empty();
        const double direction = (object.number(6)-frame->second.focus_direction)*radians;
        const double focus = frame->second.focus_range;
        const double focus_to_target = focus>0. ? std::hypot(distance*std::cos(direction)-focus,
                                                            distance*std::sin(direction)) : distance;
        const double sigma_range = .0125*(distance+focus_to_target);
        // Treat the two angular roundings and self-pose error as sensor variance.
        // The small floor also allows for errors correlated between cycles.
        const double radial_var = sigma_range*sigma_range + .04*.04;
        const double tangent_var = std::pow(.008*distance,2) + .04*.04;
        const auto cycle = static_cast<long long>(object.number(0));
        auto &t = track[id];
        const bool continuous = t.cycle+1==cycle;
        const Vec mean2 = continuous ? (observed+t.last)*.5 : observed;
        const Vec compensated = continuous && t.velocity_known ? (observed+t.last+t.velocity)*.5 : observed;
        for (size_t i=0;i<t.estimates.size();++i) {
            auto &estimate = t.estimates[i];
            const auto &candidate = filters[i];
            if (!continuous) {
                estimate.reset(observed,velocity,radial_var,candidate.velocity && vel_known);
                continue;
            }
            estimate.predict(ball ? candidate.accel*2.5 : candidate.accel);
            estimate.measure(radial,0,observed.dot(radial),radial_var);
            estimate.measure(tangent,0,observed.dot(tangent),tangent_var);
            if (candidate.velocity && vel_known) {
                // Velocity fields share some observation error with position;
                // this working variance is tuned only on validation games.
                const double variance = ball ? .12*.12 : .06*.06;
                estimate.measure({1.,0.},2,velocity.x,variance);
                estimate.measure({0.,1.},2,velocity.y,variance);
            }
        }
        t.cycle = cycle;
        t.last = observed;
        t.velocity = velocity;
        t.velocity_known = vel_known;

        // Truth is read only after all estimates and persistent states are computed.
        if (!object.yes(15) || !object.yes(16)) continue;
        const Vec truth{object.number(17),object.number(18)};
        const double true_distance = object.number(19);
        if (!truth.valid() || !std::isfinite(true_distance)) continue;
        const char *band = bands[std::upper_bound(edges.begin(),edges.end(),true_distance)-edges.begin()];
        const bool near = true_distance < 50.;
        const std::string target = ball?"ball":"player";
        const double stock = (observed-truth).length();
        score(results,target,band,near,"visual",stock,stock);
        score(results,target,band,near,"mean_two",stock,(mean2-truth).length());
        score(results,target,band,near,"velocity_two",stock,(compensated-truth).length());
        for (size_t i=0;i<t.estimates.size();++i) {
            const Vec p = t.estimates[i].pos();
            const std::string method = filters[i].name;
            score(results,target,band,near,method,stock,(p-truth).length());
            if (near_safe && (!filters[i].velocity || i==2)) {
                const Vec gated = distance < 5. ? observed : clamp_correction(p,observed,.25);
                score(results,target,band,near,method+"_far5_cap25",stock,(gated-truth).length());
                if (!filters[i].velocity) {
                    for (const double max_step : {.25,.5}) {
                        const auto clamped = clamp_correction(p,observed,max_step);
                        score(results,target,band,near,method+(max_step==.25?"_cap25":"_cap50"),
                              stock,(clamped-truth).length());
                    }
                }
            }
            if (!filters[i].velocity) continue;
            for (const double max_step : {.25,.5}) {
                const auto clamped = clamp_correction(p,observed,max_step);
                score(results,target,band,near,method+(max_step==.25?"_cap25":"_cap50"),
                      stock,(clamped-truth).length());
            }
        }
    }
}
} // namespace

int main(int argc,char **argv) {
    if (argc<2) return 2;
    try {
        Results pooled;
        const bool near_safe = std::string_view(argv[1]) == "--near-safe";
        for (int i=near_safe?2:1;i<argc;++i) scan_observer(argv[i],pooled,near_safe);
        std::cout << std::setprecision(17);
        for (const auto &[group,m] : pooled) {
            const auto &[target,band,method] = group;
            std::cout << target << '\t' << band << '\t' << method << '\t' << m.count << '\t'
                      << m.raw << '\t' << m.error << '\t' << m.better << '\t' << m.worse25
                      << '\t' << m.worse50 << '\t' << m.max_added << '\n';
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
