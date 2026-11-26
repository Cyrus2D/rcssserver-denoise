#ifndef EXP6_CSV_READER_H
#define EXP6_CSV_READER_H

#include <zlib.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace exp6 {
inline constexpr double missing = std::numeric_limits<double>::quiet_NaN();

// The logger produces quoted CSV. Numeric tables use the same reader as agent_cycle,
// whose command and label fields can contain commas.
class Csv {
    gzFile file;
    std::array<char, 16384> line{};
    std::vector<int> destination;
    std::vector<std::string_view> cells;

    bool read_line() {
        if (!gzgets(file, line.data(), int(line.size()))) {
            int error;
            const char *message = gzerror(file, &error);
            if (error != Z_OK && error != Z_STREAM_END) throw std::runtime_error(message);
            return false;
        }
        const std::string_view content(line.data());
        if (!content.empty() && content.back() != '\n')
            throw std::runtime_error("CSV line exceeds buffer or is incomplete");
        return true;
    }
public:
    Csv(const std::filesystem::path &path, const std::vector<std::string_view> &columns)
        : file(gzopen(path.c_str(), "rb")), cells(columns.size()) {
        if (!file) throw std::runtime_error("Cannot open " + path.string());
        gzbuffer(file, 256 * 1024);
        if (!read_line()) throw std::runtime_error("Missing header " + path.string());
        std::string_view header(line.data());
        header.remove_suffix(1);
        for (size_t start = 0; start <= header.size();) {
            const size_t end = header.find(',', start);
            const auto col = header.substr(start, end - start);
            const auto it = std::find(columns.begin(), columns.end(), col);
            destination.push_back(it == columns.end() ? -1 : int(it - columns.begin()));
            if (end == header.npos) break;
            start = end + 1;
        }
        for (size_t i = 0; i < columns.size(); ++i)
            if (std::find(destination.begin(), destination.end(), int(i)) == destination.end())
                throw std::runtime_error("Missing column " + std::string(columns[i]) + " in " + path.string());
        while (destination.back() < 0) destination.pop_back();
    }
    ~Csv() { gzclose(file); }
    bool next() {
        if (!read_line()) return false;
        const char *p = line.data();
        for (const int field : destination) {
            const char *start = p;
            if (*p == '"') {
                ++start;
                for (++p; *p; ++p) {
                    if (*p != '"') continue;
                    if (p[1] == '"') { ++p; continue; }
                    break;
                }
                if (field >= 0) cells[field] = {start, size_t(p - start)};
                if (*p != '"') throw std::runtime_error("Unterminated quoted field");
                ++p;
            } else {
                while (*p != ',' && *p != '\n' && *p) ++p;
                if (field >= 0) cells[field] = {start, size_t(p - start)};
            }
            if (*p == ',') ++p;
            else if (*p != '\n') throw std::runtime_error("Malformed CSV record");
        }
        return true;
    }
    std::string_view operator[](size_t i) const { return cells[i]; }
    double number(size_t i) const {
        const auto s = cells[i];
        if (s.empty()) return missing;
        double value = missing;
        const auto parsed = std::from_chars(s.data(), s.data() + s.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != s.data() + s.size())
            throw std::runtime_error("Invalid number " + std::string(s));
        return value;
    }
    bool yes(size_t i) const { return cells[i] == "1"; }
    long long key() const { return static_cast<long long>(number(0)) * (1LL << 20) + static_cast<long long>(number(1)); }
};

} // namespace exp6

#endif
