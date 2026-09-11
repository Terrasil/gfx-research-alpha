#pragma once

namespace gfx::alpha {
    class LayeredWboitTarget {
    public:
        static constexpr int layer_count = 8;

        LayeredWboitTarget() = default;
        LayeredWboitTarget(const LayeredWboitTarget&) = delete;
        LayeredWboitTarget& operator=(const LayeredWboitTarget&) = delete;
        LayeredWboitTarget(LayeredWboitTarget&& other) noexcept;
        LayeredWboitTarget& operator=(LayeredWboitTarget&& other) noexcept;
        ~LayeredWboitTarget();

        bool create(int width, int height);
        bool resize(int width, int height);
        void bind_layers() const;
        void bind_global_revealage() const;
        void clear() const;
        void reset();

        [[nodiscard]] unsigned int accumulation() const { return accumulation_; }
        [[nodiscard]] unsigned int revealage() const { return revealage_; }
        [[nodiscard]] unsigned int global_revealage() const { return global_revealage_; }
        [[nodiscard]] int width() const { return width_; }
        [[nodiscard]] int height() const { return height_; }

    private:
        bool allocate();

        unsigned int layered_fbo_ = 0;
        unsigned int global_fbo_ = 0;
        unsigned int accumulation_ = 0;
        unsigned int revealage_ = 0;
        unsigned int global_revealage_ = 0;
        int width_ = 0;
        int height_ = 0;
    };
}
