#pragma once

namespace gfx::alpha {
    class WboitTarget {
    public:
        WboitTarget() = default;
        WboitTarget(const WboitTarget&) = delete;
        WboitTarget& operator=(const WboitTarget&) = delete;
        WboitTarget(WboitTarget&& other) noexcept;
        WboitTarget& operator=(WboitTarget&& other) noexcept;
        ~WboitTarget();

        bool create(int width, int height);
        bool resize(int width, int height);
        void bind() const;
        void clear() const;
        void reset();

        [[nodiscard]] unsigned int fbo() const { return fbo_; }
        [[nodiscard]] unsigned int accumulation() const { return accumulation_; }
        [[nodiscard]] unsigned int revealage() const { return revealage_; }
        [[nodiscard]] int width() const { return width_; }
        [[nodiscard]] int height() const { return height_; }

    private:
        bool allocate();

        unsigned int fbo_ = 0;
        unsigned int accumulation_ = 0;
        unsigned int revealage_ = 0;
        int width_ = 0;
        int height_ = 0;
    };
}
