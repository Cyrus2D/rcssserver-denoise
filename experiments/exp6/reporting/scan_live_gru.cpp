#include "csv_reader.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace {
using exp6::Csv;
namespace fs = std::filesystem;

struct Stats {
    long long rows = 0, applied = 0, better1 = 0, worse1 = 0, worse5 = 0, worse10 = 0, worse25 = 0;
    double stock_sum = 0., decision_sum = 0., max_added = 0.;
    void add(double stock, double decision, bool corrected) {
        const double delta = decision - stock;
        ++rows;
        applied += corrected;
        stock_sum += stock;
        decision_sum += decision;
        better1 += delta < -.01;
        worse1 += delta > .01;
        worse5 += delta > .05;
        worse10 += delta > .10;
        worse25 += delta > .2502; // coordinates in the log are rounded to 0.1 mm
        max_added = std::max(max_added, delta);
    }
};

struct Audit {
    long long rows = 0, out_of_policy = 0, unsafe_shift = 0, over_range_skips = 0, near_unchanged = 0;
    double max_shift = 0.;
};

using Key = std::tuple<char, std::string_view, std::string_view>;
using Metrics = std::map<Key, Stats>;

std::string_view distance_band(double distance) {
    if (distance < 5.) return "0-5";
    if (distance < 10.) return "5-10";
    if (distance < 20.) return "10-20";
    if (distance < 35.) return "20-35";
    if (distance < 50.) return "35-50";
    return "50+";
}

void scan(const fs::path &folder, char side, Metrics &metrics, Audit &audit) {
    Csv agent(folder/"agent_cycle.csv.gz", {"cycle","stopped","game_mode","decision_made","n_est_pos_d"});
    std::unordered_set<long long> play_on;
    while (agent.next()) {
        if (agent[1] != "0" || agent[2] != "play_on" || !agent.yes(3)) continue;
        play_on.insert(agent.key());
        const double error = agent.number(4);
        if (std::isfinite(error)) metrics[{side,"self","all"}].add(error,error,false);
    }

    enum Column { Cycle, Stopped, Id, Truth, OnField, TrueX, TrueY, Distance, Team,
                  Seen, Method, ObservedRange, TrackX, TrackY, StockX, StockY, Applied,
                  PosCount, SeenPosCount };
    Csv object(folder/"objects.csv.gz", {"cycle","stopped","object_id","truth_valid",
        "tru_on_field","tru_pos_x","tru_pos_y","true_dist","is_teammate","seen","match_method",
        "dist","trk_pos_x","trk_pos_y","trk_stock_pos_x","trk_stock_pos_y","trk_decision_pos_applied",
        "trk_pos_count","trk_seen_pos_count"});
    long long frame = -1;
    bool play = false;
    double ball_x = exp6::missing, ball_y = exp6::missing;
    while (object.next()) {
        const auto key = object.key();
        if (key != frame) {
            frame = key;
            play = play_on.count(key) != 0;
            ball_x = ball_y = exp6::missing;
        }
        const bool ball = object[Id] == "0";
        const bool applied = object.yes(Applied);
        const bool named = object.yes(Seen) && object[Method] == "unum";
        const bool fresh = object[PosCount] == "0" && object[SeenPosCount] == "0";
        if (applied) {
            ++audit.rows;
            audit.out_of_policy += side != 'l' || !play || ball || !named || !fresh
                || !(object.number(ObservedRange) >= 5.);
            const double shift = std::hypot(object.number(TrackX)-object.number(StockX),
                                            object.number(TrackY)-object.number(StockY));
            audit.unsafe_shift += !std::isfinite(shift) || shift > .2502;
            audit.max_shift = std::max(audit.max_shift,shift);
        }
        if (!play || !object.yes(Truth) || !object.yes(OnField)) continue;
        const double true_x = object.number(TrueX), true_y = object.number(TrueY);
        const double decision = std::hypot(object.number(TrackX)-true_x,object.number(TrackY)-true_y);
        if (ball) {
            ball_x = true_x;
            ball_y = true_y;
            if (std::isfinite(decision)) metrics[{side,"ball","all"}].add(decision,decision,false);
            continue;
        }
        if (!named || !fresh || !std::isfinite(decision)) continue;
        const double distance = object.number(Distance);
        if (!std::isfinite(distance)) continue;

        if (side == 'l' && !applied) {
            const double observed = object.number(ObservedRange);
            audit.near_unchanged += observed < 5.;
            audit.over_range_skips += observed >= 5.;
        }
        const double stock = applied ? std::hypot(object.number(StockX)-true_x,object.number(StockY)-true_y)
                                     : decision;
        if (!std::isfinite(stock)) throw std::runtime_error("Missing recorded stock position");
        const auto relation = object.yes(Team) ? "teammate" : "opponent";
        const auto band = distance_band(distance);
        metrics[{side,"player","all"}].add(stock,decision,applied);
        metrics[{side,"player",band}].add(stock,decision,applied);
        if (distance < 50.) {
            metrics[{side,"player","under50"}].add(stock,decision,applied);
            metrics[{side,relation,"under50"}].add(stock,decision,applied);
        }
        if (distance < 20. && std::hypot(true_x-ball_x,true_y-ball_y) < 2.) {
            metrics[{side,relation,"within2m_of_ball_under20m"}].add(stock,decision,applied);
        }
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const fs::path game(argv[1]);
    try {
        Metrics metrics;
        std::map<char, Audit> audits;
        for (char side : {'l','r'}) {
            for (int unum = 1; unum <= 11; ++unum)
                scan(game/(std::string(1,side) + '_' + std::to_string(unum)),side,metrics,audits[side]);
        }
        std::cout << std::setprecision(14);
        for (const auto &[side, a] : audits)
            std::cout << "audit\t" << side << '\t' << a.rows << '\t' << a.out_of_policy << '\t'
                      << a.unsafe_shift << '\t' << a.over_range_skips << '\t' << a.near_unchanged
                      << '\t' << a.max_shift << '\n';
        for (const auto &[key, s] : metrics) {
            const auto &[side, target, band] = key;
            std::cout << "metric\t" << side << '\t' << target << '\t' << band << '\t'
                      << s.rows << '\t' << s.applied << '\t' << s.stock_sum << '\t' << s.decision_sum
                      << '\t' << s.better1 << '\t' << s.worse1 << '\t' << s.worse5 << '\t' << s.worse10
                      << '\t' << s.worse25 << '\t' << s.max_added << '\n';
        }
    } catch (const std::exception &e) {
        std::cerr << game << ": " << e.what() << '\n';
        return 1;
    }
}
