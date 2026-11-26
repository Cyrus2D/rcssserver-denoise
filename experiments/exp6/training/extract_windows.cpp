#include "../reporting/csv_reader.h"
#include "../reporting/kalman_state.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace {
using exp6::Csv;
using exp6::Kalman;
using exp6::Vec;
namespace fs = std::filesystem;
constexpr int history = 8;
constexpr int features = 14;
constexpr int label_columns = 6;
constexpr double radians = 3.14159265358979323846 / 180.;

struct Sample {
    Vec pos, vel, radial, kf_delta;
    double distance, sigma;
    bool direct_velocity, teammate;
};
struct Track {
    long long last_cycle = -100;
    int block = -1;
    int length = 0;
    Kalman kf;
    std::array<Sample,history> samples;
};
struct Frame {
    double focus_range, focus_direction;
    int block;
};

void extract(const fs::path &directory, const fs::path &output, const int player_stride,
             const int ball_stride) {
    Csv agent(directory/"agent_cycle.csv.gz", {
        "cycle","stopped","game_mode","decision_made","focus_dist","focus_dir"});
    std::unordered_map<long long,Frame> frames;
    long long previous = -100;
    int block = 0;
    while (agent.next()) {
        if (agent[2]!="play_on" || !agent.yes(3) || agent[1]!="0") {
            previous = -100;
            continue;
        }
        const auto cycle = static_cast<long long>(agent.number(0));
        if (cycle!=previous+1) ++block;
        previous = cycle;
        const double f = agent.number(4), dir = agent.number(5);
        if (!frames.emplace(agent.key(),Frame{std::isfinite(f)?f:0.,
                                             std::isfinite(dir)?dir:0.,block}).second)
            throw std::runtime_error("duplicate PlayOn decision");
    }

    std::ofstream players(output.string()+".player.bin",std::ios::binary);
    std::ofstream balls(output.string()+".ball.bin",std::ios::binary);
    if (!players || !balls) throw std::runtime_error("cannot open output shards");
    Csv object(directory/"objects.csv.gz", {
        "cycle","stopped","object_id","seen","match_method","is_teammate","dist","dir",
        "dist_chg","dir_chg","ls_pos_x","ls_pos_y","ls_rpos_x","ls_rpos_y",
        "ls_vel_x","ls_vel_y","truth_valid","tru_on_field","tru_pos_x","tru_pos_y","true_dist"});
    std::array<Track,23> tracks{};
    while (object.next()) {
        const auto frame = frames.find(object.key());
        if (frame==frames.end() || !object.yes(3)) continue;
        const int id = static_cast<int>(object.number(2));
        if (id<0 || id>=static_cast<int>(tracks.size())) throw std::runtime_error("invalid object id");
        const bool ball = id==0;
        if (!ball && object[4]!="unum") continue;
        const double distance = object.number(6);
        const Vec pos{object.number(10),object.number(11)};
        const Vec relative{object.number(12),object.number(13)};
        if (!pos.valid() || !relative.valid() || !std::isfinite(distance)) continue;
        const double length = relative.length();
        const Vec ray = length>1.e-4 ? relative*(1./length) : Vec{1.,0.};
        const Vec tangent{-ray.y,ray.x};
        const Vec measured_velocity{object.number(14),object.number(15)};
        const bool direct = measured_velocity.valid() && !object[8].empty() && !object[9].empty();
        const Vec velocity = direct ? measured_velocity : Vec{};
        const auto &f = frame->second;
        const double angle = (object.number(7)-f.focus_direction)*radians;
        const double focus_to_target = f.focus_range>0.
            ? std::hypot(distance*std::cos(angle)-f.focus_range,distance*std::sin(angle))
            : distance;
        const double sigma = .0125*(distance+focus_to_target);
        const auto cycle = static_cast<long long>(object.number(0));
        auto &track = tracks[id];
        const bool continuous = track.last_cycle+1==cycle && track.block==f.block;
        // Match the physical filters selected in the earlier study.
        const double radial_var = sigma*sigma + .04*.04;
        const double tangent_var = std::pow(.008*distance,2) + .04*.04;
        if (continuous) {
            track.kf.predict(ball ? .18*2.5 : .20);
            track.kf.measure(ray,0,pos.dot(ray),radial_var);
            track.kf.measure(tangent,0,pos.dot(tangent),tangent_var);
            if (ball && direct) {
                track.kf.measure({1.,0.},2,velocity.x,.12*.12);
                track.kf.measure({0.,1.},2,velocity.y,.12*.12);
            }
        } else {
            track.length = 0;
            track.kf.reset(pos,velocity,radial_var,ball && direct);
        }
        const Vec kf_delta = track.kf.pos()-pos;
        track.samples[cycle%history] = {pos,velocity,ray,kf_delta,distance,sigma,direct,object.yes(5)};
        track.length = std::min(track.length+1,history);
        track.last_cycle = cycle;
        track.block = f.block;

        // Sample at a fixed cycle/identity phase, independently of truth or error.
        const int stride = ball ? ball_stride : player_stride;
        if ((cycle+id)%stride!=0 || !object.yes(16) || !object.yes(17)) continue;
        const Vec truth{object.number(18),object.number(19)};
        const double true_distance = object.number(20);
        if (!truth.valid() || !std::isfinite(true_distance)) continue;
        std::array<float,history*features+label_columns> row{};
        for (int i=history-track.length;i<history;++i) {
            const auto &s = track.samples[(cycle-history+1+i)%history];
            const Vec shift = s.pos-pos;
            // The anchor and the coordinate axes come from the current observation.
            const double values[features] = {
                shift.dot(ray)/3., shift.dot(tangent)/3.,
                s.vel.dot(ray), s.vel.dot(tangent), s.direct_velocity?1.:0.,
                s.distance/50., s.sigma, s.radial.dot(ray), s.radial.dot(tangent),
                s.kf_delta.dot(ray), s.kf_delta.dot(tangent),
                s.teammate?1.:0., 1., static_cast<double>(history-1-i)/history
            };
            for (int j=0;j<features;++j) row[i*features+j] = static_cast<float>(values[j]);
        }
        const Vec target = truth-pos;
        const double labels[label_columns] = {
            target.dot(ray), target.dot(tangent), distance, true_distance,
            kf_delta.dot(ray), kf_delta.dot(tangent)
        };
        for (int i=0;i<label_columns;++i)
            row[history*features+i] = static_cast<float>(labels[i]);
        auto &stream = ball ? balls : players;
        stream.write(reinterpret_cast<const char *>(row.data()),sizeof(row));
    }
    players.close();
    balls.close();
    if (!players || !balls) throw std::runtime_error("failed writing a shard");
}
} // namespace

int main(int argc,char **argv) {
    if (argc!=5) return 2;
    try {
        const int player_stride = std::stoi(argv[3]), ball_stride = std::stoi(argv[4]);
        if (player_stride<1 || ball_stride<1) return 2;
        extract(argv[1],argv[2],player_stride,ball_stride);
    } catch (const std::exception &error) {
        std::cerr << argv[1] << ": " << error.what() << '\n';
        return 1;
    }
}
