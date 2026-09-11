#include <gfx/alpha/wboit_target.hpp>

#include <cstdio>
#include <utility>

#include <glad/gl.h>

namespace gfx::alpha {
    WboitTarget::WboitTarget(WboitTarget&& other) noexcept { *this = std::move(other); }

    WboitTarget& WboitTarget::operator=(WboitTarget&& other) noexcept {
        if (this == &other) return *this;
        reset();
        fbo_ = std::exchange(other.fbo_, 0);
        accumulation_ = std::exchange(other.accumulation_, 0);
        revealage_ = std::exchange(other.revealage_, 0);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
        return *this;
    }

    WboitTarget::~WboitTarget() { reset(); }

    bool WboitTarget::create(int width, int height) {
        reset();
        if (width <= 0 || height <= 0) return false;
        width_ = width;
        height_ = height;
        return allocate();
    }

    bool WboitTarget::resize(int width, int height) {
        if (width == width_ && height == height_ && fbo_) return true;
        return create(width, height);
    }

    bool WboitTarget::allocate() {
        glCreateFramebuffers(1, &fbo_);
        glCreateTextures(GL_TEXTURE_2D, 1, &accumulation_);
        glCreateTextures(GL_TEXTURE_2D, 1, &revealage_);
        glTextureStorage2D(accumulation_, 1, GL_RGBA16F, width_, height_);
        glTextureStorage2D(revealage_, 1, GL_R16F, width_, height_);
        for (const GLuint texture : {accumulation_, revealage_}) {
            glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, accumulation_, 0);
        glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT1, revealage_, 0);
        const GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glNamedFramebufferDrawBuffers(fbo_, 2, buffers);
        const GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            std::fprintf(stderr, "WBOIT framebuffer incomplete: 0x%x\n", status);
            return false;
        }
        return true;
    }

    void WboitTarget::bind() const { glBindFramebuffer(GL_FRAMEBUFFER, fbo_); }

    void WboitTarget::clear() const {
        const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float one[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        glClearNamedFramebufferfv(fbo_, GL_COLOR, 0, zero);
        glClearNamedFramebufferfv(fbo_, GL_COLOR, 1, one);
    }

    void WboitTarget::reset() {
        if (accumulation_) glDeleteTextures(1, &accumulation_);
        if (revealage_) glDeleteTextures(1, &revealage_);
        if (fbo_) glDeleteFramebuffers(1, &fbo_);
        fbo_ = accumulation_ = revealage_ = 0;
        width_ = height_ = 0;
    }
}
