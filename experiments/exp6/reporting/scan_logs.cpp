#include "csv_reader.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {
namespace fs = std::filesystem;
using exp6::Csv;
using exp6::missing;
constexpr std::array<double, 6> edges{3, 10, 20, 30, 40, 60};
constexpr double radians = 3.14159265358979323846 / 180.0;


enum Metric {
    Seen, Known, Motion, Body, Head, Range, SignedRange, Bearing, Raw, ExactRange, ExactBearing,
    Relative, Visual, Track, PairedVisual, PairedTrack, VisualVel, TrackVel, RangeChange,
    BearingChange, BodyDirection, HeadDirection, Fresh, SignedLocalizedRange, PairedRawRange,
    LocalizedRange, RawWorse50, RawWorse100, TrackWorse50, TrackWorse100, Z, ZAbs1, ZAbs2, ZAbs3,
    SelfPosition, SelfVelocity, SelfBody, SelfFace, SelfFix, SelfFixFace,
    ReplayStock, ReplayRaw, ReplayImproved, ReplayWorsened25, Count
};
constexpr std::array<const char *, Count> metric_names{
    "seen_share","known_share","motion_share","body_share","head_share","raw_range","signed_range",
    "raw_bearing","raw_position","exact_range_position","exact_bearing_position","visual_relative",
    "visual_global","track_position","paired_visual","paired_track","visual_velocity","track_velocity",
    "n_dist_chg","n_dir_chg","n_body_rel","n_head_rel","fresh_wm","signed_localized_range",
    "paired_raw_range","localized_range","raw_over_50cm","raw_over_100cm","wm_over_50cm","wm_over_100cm",
    "z","z_abs_over_1","z_abs_over_2","z_abs_over_3","self_position","self_velocity","self_body",
    "self_face","self_fix_position","self_fix_face","replay_stock","replay_raw",
    "replay_improved_1cm","replay_worsened_25cm"
};
enum Group { Player, Teammate, Opponent, Ball, Landmark, Self, Groups };
constexpr std::array<const char *, Groups> group_names{"all","teammate","opponent","ball","landmark","self"};

struct Stat {
    uint64_t n = 0;
    double sum = 0;
    double squares = 0;
    void add(double x) {
        if (!std::isfinite(x)) return;
        ++n;
        sum += x;
        squares += x*x;
    }
};
using Metrics = std::array<Stat, Count>;
using ByDistance = std::array<Metrics, 7>;
struct Audit {
    std::array<ByDistance, Groups> stats{};
    std::map<std::string, uint64_t> coverage;
    double distance_rate;
    double focus_rate;
    double landmark_distance_rate;
    double landmark_focus_rate;

    void polar(Metrics &m, double d, double measured, double bearing, double f,
               double r_d, double r_f, double delta) {
        const double range = measured - d;
        m[Range].add(std::abs(range));
        m[SignedRange].add(delta);
        m[Bearing].add(std::abs(bearing));
        const double angle = 2 * std::abs(std::sin(bearing * radians / 2));
        const double pos = std::hypot(range, std::sqrt(std::max(0., measured*d)) * angle);
        if (std::isfinite(range) && std::isfinite(angle)) {
            m[Raw].add(pos);
            m[ExactRange].add(d * angle);
            m[ExactBearing].add(std::abs(range));
            m[RawWorse50].add(pos > .5);
            m[RawWorse100].add(pos > 1.);
        }
        const double sigma = d*r_d + f*r_f;
        if (sigma > 0 && std::isfinite(delta)) {
            const double z = delta / sigma;
            m[Z].add(z);
            m[ZAbs1].add(std::abs(z) > 1);
            m[ZAbs2].add(std::abs(z) > 2);
            m[ZAbs3].add(std::abs(z) > 3);
        }
    }

    void collect(const fs::path &root) {
        std::unordered_set<long long> play;
        Csv agent(root/"agent_cycle.csv.gz", {"cycle","stopped","game_mode","decision_made",
            "truth_valid","see_used","see_count_this_cycle","view_angle_deg","view_quality","say_msg",
            "tru_focus_dist","n_est_pos_d","n_est_vel_d","n_est_body","n_est_face","n_rawfix_pos_d","n_rawfix_face"});
        while (agent.next()) {
            coverage["skipped"] += agent[3] == "0";
            if (agent[2] != "play_on" || !agent.yes(3)) continue;
            ++coverage["play"];
            coverage["truth"] += agent.yes(4);
            coverage["see"] += agent.yes(5);
            coverage["one_see"] += agent.yes(6);
            coverage["full_view"] += agent.number(7) == 360;
            coverage["high_quality"] += agent[8] == "high";
            coverage["say"] += !agent[9].empty();
            if (!agent.yes(4)) continue;
            if (!play.insert(agent.key()).second) throw std::runtime_error("Duplicate PlayOn key");
            const double focus = agent.number(10);
            coverage["focus_labeled"] += std::isfinite(focus);
            coverage["focus_nonzero"] += focus > 0.0001;
            for (int i = 0; i < 6; ++i) stats[Self][0][SelfPosition+i].add(std::abs(agent.number(11+i)));
        }
        Csv hear(root/"hear.csv.gz", {"sender_kind","has_content"});
        while (hear.next()) {
            coverage["teammate_payload"] += hear[0] == "teammate" && hear.yes(1);
        }
        Csv obs(root/"objects.csv.gz", {"cycle","stopped","object_id","truth_valid","tru_on_field",
            "is_teammate","true_dist","true_dir","true_focus_dist","seen","unum_visible","match_method",
            "match_ambiguous","dist","dir","dist_chg","dir_chg","body_rel","head_rel","n_dist",
            "n_dist_chg","n_dir_chg","n_body_rel","n_head_rel","n_ls_rpos_d","n_ls_pos_d","n_trk_pos_d",
            "n_ls_vel_d","n_trk_vel_d","ls_rpos_x","ls_rpos_y","trk_pos_count",
            "ls_pos_x","ls_pos_y","tru_pos_x","tru_pos_y"});
        while (obs.next()) {
            if (!obs.yes(3) || !obs.yes(4) || !play.count(obs.key())) continue;
            const double d = obs.number(6);
            if (!std::isfinite(d)) continue;
            const size_t bin = std::upper_bound(edges.begin(), edges.end(), d) - edges.begin();
            const bool ball = obs[2] == "0", seen = obs.yes(9);
            const bool known = seen && (ball || (obs.yes(10) && obs[11] == "unum"));
            const bool valid = known && !obs.yes(12);
            const size_t target = obs.yes(5) ? Teammate : Opponent;
            Metrics row{};
            row[Seen].add(seen); row[Known].add(known);
            row[Motion].add(known && !obs[15].empty() && !obs[16].empty());
            row[Body].add(known && !obs[17].empty());
            row[Head].add(known && !obs[18].empty());
            const double track = obs.number(26);
            row[Track].add(track);
            row[TrackVel].add(obs.number(28));
            if (std::isfinite(track)) {
                row[TrackWorse50].add(track > .5);
                row[TrackWorse100].add(track > 1.);
                row[Fresh].add(obs[31] == "0");
            }
            if (valid) {
                const double measured = obs.number(13);
                const double bearing = std::remainder(obs.number(14) - obs.number(7), 360.);
                polar(row,d,measured,bearing,obs.number(8),distance_rate,focus_rate,obs.number(19));
                if (std::isfinite(measured)) {
                    coverage["known_ranges"] += !ball;
                    coverage["off_grid_ranges"] += !ball && std::abs(measured*10-std::round(measured*10)) > 1e-7;
                }
                for (int i = 0; i < 4; ++i) row[RangeChange+i].add(std::abs(obs.number(20+i)));
                row[Relative].add(obs.number(24));
                const double visual = obs.number(25);
                row[Visual].add(visual);
                row[VisualVel].add(obs.number(27));
                if (std::isfinite(visual) && std::isfinite(track)) {
                    row[PairedVisual].add(visual);
                    row[PairedTrack].add(track);
                }
                const double rx = obs.number(29), ry = obs.number(30);
                const double radius = std::hypot(rx, ry);
                const double lr = radius - d;
                if (std::isfinite(visual) && std::isfinite(measured) && radius > 0) {
                    // Keep the original estimated origin and direction. Truth
                    // appears only in the final scoring subtraction.
                    const double change = measured / radius - 1.;
                    const double revised = std::hypot((obs.number(32) - obs.number(34)) + change*rx,
                                                      (obs.number(33) - obs.number(35)) + change*ry);
                    if (std::isfinite(revised)) {
                        row[ReplayStock].add(visual);
                        row[ReplayRaw].add(revised);
                        row[ReplayImproved].add(revised < visual - .01);
                        row[ReplayWorsened25].add(revised > visual + .25);
                    }
                }
                if (std::isfinite(lr) && std::isfinite(measured)) {
                    row[SignedLocalizedRange].add(lr);
                    row[LocalizedRange].add(std::abs(lr));
                    row[PairedRawRange].add(std::abs(measured-d));
                }
            }
            auto merge = [&](Group group) {
                auto &to = stats[group][bin];
                for (size_t i=0; i<row.size(); ++i) {
                    to[i].n += row[i].n; to[i].sum += row[i].sum; to[i].squares += row[i].squares;
                }
            };
            if (ball) merge(Ball);
            else { merge(Player); merge(Group(target)); }
        }
        Csv flags(root/"landmarks.csv.gz", {"cycle","stopped","true_dist","dist","true_focus_dist","n_dist","n_dir"});
        while (flags.next()) {
            if (!play.count(flags.key())) continue;
            const double d = flags.number(2);
            if (!std::isfinite(d)) continue;
            const size_t bin = std::upper_bound(edges.begin(), edges.end(), d) - edges.begin();
            auto &m = stats[Landmark][bin];
            polar(m,d,flags.number(3),flags.number(6),flags.number(4),landmark_distance_rate,
                  landmark_focus_rate,flags.number(5));
        }
    }
};
} // namespace

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    try {
        Audit audit{{},{},std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5])};
        for (const auto &entry : fs::directory_iterator(argv[1]))
            if (entry.is_directory() && fs::exists(entry.path()/"meta.json")) audit.collect(entry.path());
        std::cout << std::setprecision(17);
        for (const auto &[name,n] : audit.coverage) std::cout << "coverage\t" << name << '\t' << n << '\n';
        for (size_t group=0; group<Groups; ++group)
            for (size_t bin=0; bin<7; ++bin)
                for (size_t i=0; i<Count; ++i) {
                    const auto &s = audit.stats[group][bin][i];
                    if (s.n) std::cout << group_names[group] << '\t' << bin << '\t' << metric_names[i]
                        << '\t' << s.n << '\t' << s.sum << '\t' << s.squares << '\n';
                }
    } catch (const std::exception &error) {
        std::cerr << argv[1] << ": " << error.what() << '\n';
        return 1;
    }
}

