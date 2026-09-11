#include <gfx/alpha/layered_wboit_target.hpp>

#include <cstdio>
#include <utility>

#include <glad/gl.h>

namespace gfx::alpha {
    LayeredWboitTarget::LayeredWboitTarget(LayeredWboitTarget&& other) noexcept { *this = std::move(other); }

    LayeredWboitTarget& LayeredWboitTarget::operator=(LayeredWboitTarget&& other) noexcept {
        if (this == &other) return *this;
        reset();
        layered_fbo_ = std::exchange(other.layered_fbo_, 0);
        global_fbo_ = std::exchange(other.global_fbo_, 0);
        accumulation_ = std::exchange(other.accumulation_, 0);
        revealage_ = std::exchange(other.revealage_, 0);
        global_revealage_ = std::exchange(other.global_revealage_, 0);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
        return *this;
    }

    LayeredWboitTarget::~LayeredWboitTarget() { reset(); }

    bool LayeredWboitTarget::create(int width, int height) {
        reset();
        if (width <= 0 || height <= 0) return false;
        width_ = width;
        height_ = height;
        return allocate();
    }

    bool LayeredWboitTarget::resize(int width, int height) {
        if (width == width_ && height == height_ && layered_fbo_ && global_fbo_) return true;
        return create(width, height);
    }

    bool LayeredWboitTarget::allocate() {
        glCreateFramebuffers(1, &layered_fbo_);
        glCreateFramebuffers(1, &global_fbo_);
        glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &accumulation_);
        glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &revealage_);
        glCreateTextures(GL_TEXTURE_2D, 1, &global_revealage_);

        glTextureStorage3D(accumulation_, 1, GL_RGBA16F, width_, height_, layer_count);
        glTextureStorage3D(revealage_, 1, GL_R16F, width_, height_, layer_count);
        glTextureStorage2D(global_revealage_, 1, GL_R16F, width_, height_);

        for (const GLuint texture : {accumulation_, revealage_, global_revealage_}) {
            glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        glNamedFramebufferTexture(layered_fbo_, GL_COLOR_ATTACHMENT0, accumulation_, 0);
        glNamedFramebufferTexture(layered_fbo_, GL_COLOR_ATTACHMENT1, revealage_, 0);
        const GLenum layered_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glNamedFramebufferDrawBuffers(layered_fbo_, 2, layered_buffers);

        glNamedFramebufferTexture(global_fbo_, GL_COLOR_ATTACHMENT0, global_revealage_, 0);
        glNamedFramebufferDrawBuffer(global_fbo_, GL_COLOR_ATTACHMENT0);

        const GLenum layered_status = glCheckNamedFramebufferStatus(layered_fbo_, GL_FRAMEBUFFER);
        const GLenum global_status = glCheckNamedFramebufferStatus(global_fbo_, GL_FRAMEBUFFER);
        if (layered_status != GL_FRAMEBUFFER_COMPLETE || global_status != GL_FRAMEBUFFER_COMPLETE) {
            std::fprintf(stderr, "Layered WBOIT framebuffer incomplete: layered=0x%x global=0x%x\n", layered_status, global_status);
            return false;
        }
        return true;
    }

    void LayeredWboitTarget::bind_layers() const { glBindFramebuffer(GL_FRAMEBUFFER, layered_fbo_); }
    void LayeredWboitTarget::bind_global_revealage() const { glBindFramebuffer(GL_FRAMEBUFFER, global_fbo_); }

    void LayeredWboitTarget::clear() const {
        const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const float one[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        glClearTexImage(accumulation_, 0, GL_RGBA, GL_FLOAT, zero);
        glClearTexImage(revealage_, 0, GL_RED, GL_FLOAT, one);
        glClearTexImage(global_revealage_, 0, GL_RED, GL_FLOAT, one);
    }

    void LayeredWboitTarget::reset() {
        if (accumulation_) glDeleteTextures(1, &accumulation_);
        if (revealage_) glDeleteTextures(1, &revealage_);
        if (global_revealage_) glDeleteTextures(1, &global_revealage_);
        if (layered_fbo_) glDeleteFramebuffers(1, &layered_fbo_);
        if (global_fbo_) glDeleteFramebuffers(1, &global_fbo_);
        layered_fbo_ = global_fbo_ = 0;
        accumulation_ = revealage_ = global_revealage_ = 0;
        width_ = height_ = 0;
    }
}
