#ifndef EXP6_KALMAN_STATE_H
#define EXP6_KALMAN_STATE_H

#include <array>
#include <cmath>

namespace exp6 {

struct Vec {
    double x = 0.0, y = 0.0;
    Vec operator+(Vec other) const { return {x + other.x, y + other.y}; }
    Vec operator-(Vec other) const { return {x - other.x, y - other.y}; }
    Vec operator*(double weight) const { return {x * weight, y * weight}; }
    double length() const { return std::hypot(x, y); }
    double dot(Vec other) const { return x*other.x + y*other.y; }
    bool valid() const { return std::isfinite(x) && std::isfinite(y); }
};

struct Kalman {
    // State is global (x,y,vx,vy), in metres and metres per cycle.
    std::array<double, 4> state{};
    std::array<std::array<double, 4>, 4> cov{};
    Vec pos() const { return {state[0], state[1]}; }
    Vec vel() const { return {state[2], state[3]}; }

    void reset(Vec p, Vec v, double pos_var, bool has_velocity) {
        state = {p.x,p.y,has_velocity?v.x:0.,has_velocity?v.y:0.};
        cov = {};
        cov[0][0] = cov[1][1] = pos_var;
        cov[2][2] = cov[3][3] = has_velocity ? .06*.06 : .45*.45;
    }

    void predict(double accel_std) {
        const double q = accel_std*accel_std;
        state[0] += state[2];
        state[1] += state[3];
        auto previous = cov;
        for (int i=0;i<4;++i) {
            for (int j=0;j<4;++j) {
                cov[i][j] = previous[i][j];
                if (i<2) cov[i][j] += previous[i+2][j];
                if (j<2) cov[i][j] += previous[i][j+2];
                if (i<2 && j<2) cov[i][j] += previous[i+2][j+2];
            }
        }
        for (int i=0;i<2;++i) {
            cov[i][i] += .25*q;
            cov[i][i+2] += .5*q;
            cov[i+2][i] += .5*q;
            cov[i+2][i+2] += q;
        }
    }

    void measure(Vec axis, int offset, double value, double variance) {
        std::array<double,4> projected{};
        for (int i=0;i<4;++i)
            projected[i] = cov[i][offset]*axis.x + cov[i][offset+1]*axis.y;
        const double denom = variance + axis.x*projected[offset] + axis.y*projected[offset+1];
        const double residual = value - (axis.x*state[offset] + axis.y*state[offset+1]);
        for (int i=0;i<4;++i) {
            const double gain = projected[i]/denom;
            state[i] += gain*residual;
            for (int j=0;j<4;++j) cov[i][j] -= gain*projected[j];
        }
    }
};

} // namespace exp6

#endif
