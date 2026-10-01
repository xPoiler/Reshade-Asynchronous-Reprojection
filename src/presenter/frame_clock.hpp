#pragma once
#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

namespace fw {

// When the game's frames happened, for a game that does not say (no simulation-start marker: the ReShade
// path, without DLSS or FSR). All that is known then is when each frame was presented, and that carries
// the presentation's own unevenness: RE2 presents one frame in five or so 30% late while its camera turns
// by the same amount as in the frames around it - the simulation kept its pace, only the presentation
// stuttered. Divided by the presented interval, such a turn looks like the camera slowing down and then
// speeding up, and a warp extrapolating that speed shakes. So frames are timed at the game's steady pace
// (the median of the recent intervals), kept in step with the presented times slowly (a tenth of the
// difference per frame), never later than the presented time (a frame cannot have happened after it was
// shown) nor more than one interval before it, and started again from the presented time after a gap or a
// skipped frame.
class FrameClock {
public:
    double next(std::uint64_t frame, double presented) {
        const bool consecutive = have_ && frame == frame_ + 1 && presented > presented_ && presented - presented_ < 0.25;
        if (!consecutive) {
            t_ = presented;
            intervals_.clear();
        } else {
            intervals_.push_back(presented - presented_);
            if (intervals_.size() > 15) intervals_.pop_front();
            std::vector<double> sorted(intervals_.begin(), intervals_.end());
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const double pace = sorted[sorted.size() / 2];
            t_ += pace;
            t_ += (presented - t_) * 0.1;
            // (never far from the presented time: a real change of pace is followed within a frame)
            t_ = std::clamp(t_, presented - pace, presented);
        }
        have_ = true;
        frame_ = frame;
        presented_ = presented;
        return t_;
    }
    void reset() { have_ = false; intervals_.clear(); }

private:
    bool have_ = false;
    std::uint64_t frame_ = 0;
    double presented_ = 0, t_ = 0;
    std::deque<double> intervals_;
};

}  // namespace fw
