#include <gfx/alpha/layered_wboit_target.hpp>
#include <gfx/alpha/textured_obj_scene.hpp>
#include <gfx/alpha/wboit_target.hpp>

#include <gfx/research/application.hpp>
#include <gfx/research/assets.hpp>
#include <gfx/research/framebuffer.hpp>
#include <gfx/research/fullscreen_triangle.hpp>
#include <gfx/research/gpu_timer.hpp>
#include <gfx/research/model.hpp>
#include <gfx/research/orbit_camera.hpp>
#include <gfx/research/primitives.hpp>
#include <gfx/research/screenshot.hpp>
#include <gfx/research/shader.hpp>
#include <gfx/research/texture_2d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace {
    constexpr int METHOD_RAW_DITHER = 0;
    constexpr int METHOD_OCCUPANCY_2X2 = 1;
    constexpr int METHOD_BOX = 2;
    constexpr int METHOD_GAUSSIAN = 3;
    constexpr int METHOD_POSITION_AWARE = 4;
    constexpr int METHOD_TWO_COMPONENT_LINEAR = 5;
    constexpr int METHOD_CONTROL_GUIDED_ANISOTROPIC = 6;
    constexpr int METHOD_FULL_DEFERRED_LIGHT_CONTROL = 7;
    constexpr int METHOD_WBOIT = 8;
    constexpr int METHOD_LAYERED_WBOIT = 9;
    constexpr int METHOD_CLASSIC_FORWARD_ALPHA = 10;
    constexpr int METHOD_SORTED_FORWARD_ALPHA = 11;
    constexpr int METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION = 12;
    constexpr int METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION = 13;
    constexpr int METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC = 14;
    constexpr int METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE = 15;
    constexpr int METHOD_DEPTH_PEEL_REFERENCE = 16;

    constexpr int RODD_ABLATE_NONE = 0;
    constexpr int RODD_ABLATE_ORDINAL = 1 << 0;
    constexpr int RODD_ABLATE_CENSORED_NEGATIVES = 1 << 1;
    constexpr int RODD_ABLATE_GEOMETRY = 1 << 2;
    constexpr int RODD_ABLATE_RANK_CONDITIONING = 1 << 3;
    constexpr int RODD_ABLATE_MOMENTS = 1 << 4;
    constexpr int RODD_ABLATE_DIRECTIONAL_PAIRS = 1 << 5;
    constexpr int RODD_ABLATE_CONFIDENCE_GUARD = 1 << 6;

    constexpr int MAX_TRANSPARENT_LAYERS = 16;
    constexpr int MAX_SCENE_LAYERS = MAX_TRANSPARENT_LAYERS + 1;
    constexpr int REFERENCE_PEEL_LAYERS = 24;

    constexpr std::array<const char*, 17> METHOD_LABELS = {
        "Raw coverage dither / screen-door",
        "2x2 occupancy box",
        "Local box reconstruction",
        "Gaussian reconstruction",
        "Joint bilateral / G-buffer guided reconstruction",
        "Two-component local affine reconstruction",
        "RODD experimental deferred reconstruction",
        "Raw mixed deferred control (same G-buffer, no reconstruction)",
        "Weighted blended OIT",
        "Layered weighted OIT (8 bins)",
        "Classic forward alpha (fixed draw order)",
        "Object-sorted forward alpha (baseline, not reference)",
        "Deferred attribute reconstruction (1 layer)",
        "Deferred sparse-layer K=2 legacy (post-lit resolve)",
        "Deferred sparse-layer K=2 isotropic + direct composite",
        "Deferred sparse-layer K=2 EDGE-AWARE + direct (proposed)",
        "Depth-peeling alpha reference (24 layers)"
    };

    constexpr std::array<const char*, 17> METHOD_IDS = {
        "raw-dither",
        "occupancy-2x2",
        "box-reconstruction",
        "gaussian-reconstruction",
        "joint-bilateral-gbuffer",
        "two-component-affine",
        "rodd-experimental",
        "mixed-deferred-raw-control",
        "wboit",
        "layered-wboit-8",
        "classic-forward-fixed-order",
        "object-sorted-forward",
        "deferred-attribute-reconstruction",
        "deferred-sparse-layer-2-legacy",
        "deferred-sparse-layer-2-isotropic-direct",
        "deferred-sparse-layer-2-edge-aware",
        "depth-peeling-reference-24"
    };

    constexpr std::array<const char*, 4> ALPHA_MODE_LABELS = {
        "Constant",
        "Horizontal gradient",
        "Leaf-like cutout",
        "High-frequency field"
    };

    constexpr std::array<const char*, 4> ALPHA_MODE_IDS = {
        "constant",
        "gradient",
        "leaf",
        "high-frequency"
    };

    constexpr std::array<const char*, 5> DITHER_LABELS = {
        "Bayer 4x4",
        "Stable hash",
        "RODD candidate 4x4",
        "Bayer 8x8",
        "Bayer 16x16"
    };

    constexpr std::array<const char*, 5> DITHER_IDS = {
        "bayer4",
        "hash",
        "rodd4-candidate",
        "bayer8",
        "bayer16"
    };

    constexpr std::array<const char*, 6> COVERAGE_ENCODING_LABELS = {
        "Identity",
        "Foreground biased",
        "Background biased",
        "Complementary co-designed +/- bias",
        "Balanced toward 0.5 (experimental)",
        "High-alpha power reveal (experimental)"
    };

    constexpr std::array<const char*, 6> COVERAGE_ENCODING_IDS = {
        "identity",
        "foreground-biased",
        "background-biased",
        "complementary-codesign",
        "balanced-logit-companding",
        "power-alpha-companding"
    };

    const char* roddVariantId(const int mask) {
        switch (mask) {
            case RODD_ABLATE_ORDINAL: return "no-ordinal";
            case RODD_ABLATE_CENSORED_NEGATIVES: return "no-censored-negatives";
            case RODD_ABLATE_GEOMETRY: return "no-geometry-gating";
            case RODD_ABLATE_RANK_CONDITIONING: return "no-rank-conditioning";
            case RODD_ABLATE_MOMENTS: return "no-moment-reproduction";
            case RODD_ABLATE_DIRECTIONAL_PAIRS: return "no-directional-pairs";
            case RODD_ABLATE_CONFIDENCE_GUARD: return "no-confidence-guard";
            default: return "full";
        }
    }

    constexpr std::array<const char*, 7> TRANSPARENT_GEOMETRY_LABELS = {
        "Glass panel",
        "Sphere",
        "Leaf card",
        "Depth step",
        "Mixed scene",
        "Angled plates",
        "Layer stack"
    };

    constexpr std::array<const char*, 7> TRANSPARENT_GEOMETRY_IDS = {
        "panel",
        "sphere",
        "leaf-card",
        "depth-step",
        "mixed",
        "angled-plates",
        "layer-stack"
    };

    constexpr glm::vec3 SPONZA_GROUND_POSITION = {0.0f, 0.0f, 0.0f};
    constexpr float SPONZA_TARGET_SIZE = 14.0f;
    constexpr float SPONZA_YAW_RADIANS = 0.0f;

    // Orbit around the visual center of Sponza. The default position matches the
    // useful interior view near (3.75, 1.09, 0.12) while always looking inward.
    constexpr glm::vec3 SPONZA_ORBIT_CENTER = {0.0f, 1.35f, 0.0f};
    constexpr float DEFAULT_CAMERA_DISTANCE = 4.05f;
    constexpr float DEFAULT_CAMERA_YAW = 1.54f;
    constexpr float DEFAULT_CAMERA_PITCH = -0.06f;
    constexpr float DEFAULT_CAMERA_FOV = 54.0f;
    constexpr float DEFAULT_CAMERA_NEAR = 0.03f;
    constexpr float DEFAULT_CAMERA_FAR = 250.0f;
    constexpr float CAMERA_MOUSE_SENSITIVITY = 0.0022f;
    constexpr float CAMERA_KEY_ORBIT_SPEED = 0.95f;
    constexpr float CAMERA_KEY_ZOOM_SPEED = 2.2f;
    constexpr float CAMERA_FAST_MULTIPLIER = 3.5f;
    constexpr float CAMERA_MIN_DISTANCE = 1.25f;
    constexpr float CAMERA_MAX_DISTANCE = 7.0f;

    struct LaunchOptions {
        bool benchmarkSuite = false;
        bool benchmarkQuick = false;
        bool benchmarkPaper = false;
        bool benchmarkEdgeDecision = false;
        std::filesystem::path benchmarkOutput = "results/benchmark";
    };

    struct BenchmarkCase {
        std::string pairId;
        std::string sweep;
        int method = METHOD_CONTROL_GUIDED_ANISOTROPIC;
        int repeat = 0;
        int targetWidth = 1920;
        int targetHeight = 1080;
        int lightCount = 16;
        int alphaMode = 0;
        int ditherMode = 0;
        int coverageEncodingMode = 0;
        float coverageBias = 1.15f;
        int kernelRadius = 2;
        int roddAblationMask = RODD_ABLATE_NONE;
        int transparentGeometry = 0;
        float cameraYawOffset = 0.0f;
        float objectOffsetX = 0.0f;
        int transparentLayers = 4;
        float alpha = 0.5f;
        float depthSigma = 0.35f;
        float albedoSigma = 0.12f;
        float wboitDepthExponent = 2.0f;
        float layeredDepthSpan = 12.0f;
        int warmupFrames = 60;
        int sampleFrames = 300;
        bool quality = false;
        bool capture = false;
        bool captureStages = false;
        bool recordOverdraw = false;
        bool recordDeferredLighting = false;
    };

    struct TimingStats {
        double mean = 0.0;
        double median = 0.0;
        double stddev = 0.0;
        double p95 = 0.0;
        double minimum = 0.0;
        double maximum = 0.0;
    };

    struct ImageMetrics {
        double rmse = 0.0;
        double relativeRmse = 0.0;
        double psnr = 0.0;
        double maximumAbsolute = 0.0;
        double gradientRmse = 0.0;
        double gradientRelativeRmse = 0.0;
        double gradientTransfer = 1.0;
        double globalSsim = 1.0;
        double orderedPhaseResidualRatio = 0.0;
    };

    struct AlphaMetrics {
        double rmse = 0.0;
        double mae = 0.0;
        double maximumAbsolute = 0.0;
        std::size_t samples = 0;
    };

    struct AttributeMetrics {
        double albedoRmse = std::numeric_limits<double>::quiet_NaN();
        double normalMeanDegrees = std::numeric_limits<double>::quiet_NaN();
        double normalMedianDegrees = std::numeric_limits<double>::quiet_NaN();
        double normalP95Degrees = std::numeric_limits<double>::quiet_NaN();
        double roughnessRmse = std::numeric_limits<double>::quiet_NaN();
        double roughnessMae = std::numeric_limits<double>::quiet_NaN();
        double positionRmse = std::numeric_limits<double>::quiet_NaN();
        double positionMax = std::numeric_limits<double>::quiet_NaN();
        double alphaRmse = std::numeric_limits<double>::quiet_NaN();
        double alphaMae = std::numeric_limits<double>::quiet_NaN();
        double alphaMax = std::numeric_limits<double>::quiet_NaN();
        double matchRecall = 0.0;
        double falsePositiveRatio = 0.0;
        std::size_t matchedSamples = 0;
        std::size_t referenceSamples = 0;
        std::size_t predictedSamples = 0;
        std::size_t wrongSurfaceSamples = 0;
    };

    struct OverdrawMetrics {
        double meanInputFragments = 0.0;
        double meanSurvivingFragments = 0.0;
        unsigned int maxInputFragments = 0;
        unsigned int maxSurvivingFragments = 0;
        double survivalRatio = 0.0;
        double coveredPixelRatio = 0.0;
    };

    glm::mat4 fittedModelTransform(
        const gfx::research::Model& model,
        const glm::vec3& groundPosition,
        const float targetSize,
        const float yawRadians
    ) {
        const glm::vec3 minimum = model.bounds_min();
        const glm::vec3 maximum = model.bounds_max();
        const glm::vec3 extent = maximum - minimum;
        const float largestExtent = std::max({extent.x, extent.y, extent.z, 1e-5f});
        const float scale = targetSize / largestExtent;
        const glm::vec3 anchor(
            (minimum.x + maximum.x) * 0.5f,
            minimum.y,
            (minimum.z + maximum.z) * 0.5f
        );

        glm::mat4 transform(1.0f);
        transform = glm::translate(transform, groundPosition);
        transform = glm::rotate(transform, yawRadians, glm::vec3(0.0f, 1.0f, 0.0f));
        transform = glm::scale(transform, glm::vec3(scale));
        transform = glm::translate(transform, -anchor);
        return transform;
    }

    glm::mat4 fittedSceneTransform(
        const gfx::alpha::TexturedObjScene& scene,
        const glm::vec3& groundPosition,
        const float targetSize,
        const float yawRadians
    ) {
        const glm::vec3 minimum = scene.bounds_min();
        const glm::vec3 maximum = scene.bounds_max();
        const glm::vec3 extent = maximum - minimum;
        const float largestExtent = std::max({extent.x, extent.y, extent.z, 1e-5f});
        const float scale = targetSize / largestExtent;
        const glm::vec3 anchor(
            (minimum.x + maximum.x) * 0.5f,
            minimum.y,
            (minimum.z + maximum.z) * 0.5f
        );

        glm::mat4 transform(1.0f);
        transform = glm::translate(transform, groundPosition);
        transform = glm::rotate(transform, yawRadians, glm::vec3(0.0f, 1.0f, 0.0f));
        transform = glm::scale(transform, glm::vec3(scale));
        transform = glm::translate(transform, -anchor);
        return transform;
    }

    std::string scalarToken(const float value) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << value;
        std::string result = stream.str();
        std::replace(result.begin(), result.end(), '-', 'm');
        std::replace(result.begin(), result.end(), '.', 'p');
        return result;
    }

    const char* glString(const GLenum name) {
        const auto* value = glGetString(name);
        return value ? reinterpret_cast<const char*>(value) : "unknown";
    }

    TimingStats calculateStats(const std::vector<double>& values) {
        TimingStats result;
        if (values.empty()) return result;
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        result.minimum = sorted.front();
        result.maximum = sorted.back();
        result.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
        const std::size_t middle = sorted.size() / 2;
        result.median = sorted.size() % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) * 0.5 : sorted[middle];
        double variance = 0.0;
        for (const double value : sorted) {
            const double d = value - result.mean;
            variance += d * d;
        }
        variance /= static_cast<double>(sorted.size());
        result.stddev = std::sqrt(variance);
        const std::size_t p95Index = std::min(
            sorted.size() - 1,
            static_cast<std::size_t>(std::ceil(static_cast<double>(sorted.size()) * 0.95)) - 1
        );
        result.p95 = sorted[p95Index];
        return result;
    }

    std::vector<float> readTextureRgba(const unsigned int texture, const int width, const int height) {
        std::vector<float> rgba(static_cast<std::size_t>(width) * height * 4);
        glGetTextureImage(texture, 0, GL_RGBA, GL_FLOAT, static_cast<GLsizei>(rgba.size() * sizeof(float)), rgba.data());
        return rgba;
    }

    std::vector<unsigned int> readTextureR32U(const unsigned int texture, const int width, const int height) {
        std::vector<unsigned int> values(static_cast<std::size_t>(width) * height);
        glGetTextureImage(texture, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, static_cast<GLsizei>(values.size() * sizeof(unsigned int)), values.data());
        return values;
    }

    OverdrawMetrics calculateOverdrawMetrics(const std::vector<unsigned int>& input, const std::vector<unsigned int>& survivors) {
        OverdrawMetrics result;
        if (input.size() != survivors.size() || input.empty()) return result;
        unsigned long long sumInput = 0;
        unsigned long long sumSurvivors = 0;
        std::size_t covered = 0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            const unsigned int in = input[i];
            const unsigned int sv = survivors[i];
            sumInput += in;
            sumSurvivors += sv;
            result.maxInputFragments = std::max(result.maxInputFragments, in);
            result.maxSurvivingFragments = std::max(result.maxSurvivingFragments, sv);
            if (in != 0u) ++covered;
        }
        const double pixelCount = static_cast<double>(input.size());
        result.meanInputFragments = static_cast<double>(sumInput) / pixelCount;
        result.meanSurvivingFragments = static_cast<double>(sumSurvivors) / pixelCount;
        result.survivalRatio = sumInput == 0 ? 0.0 : static_cast<double>(sumSurvivors) / static_cast<double>(sumInput);
        result.coveredPixelRatio = static_cast<double>(covered) / pixelCount;
        return result;
    }

    bool saveTexturePfm(const std::filesystem::path& path, const unsigned int texture, const int width, const int height) {
        if (!texture || width <= 0 || height <= 0) return false;
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        const std::vector<float> rgba = readTextureRgba(texture, width, height);
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) return false;
        stream << "PF\n" << width << ' ' << height << "\n-1.0\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t index = (static_cast<std::size_t>(y) * width + x) * 4;
                stream.write(reinterpret_cast<const char*>(&rgba[index]), sizeof(float) * 3);
            }
        }
        return stream.good();
    }

    bool saveTextureChannelPfm(
        const std::filesystem::path& path,
        const unsigned int texture,
        const int width,
        const int height,
        const int channel
    ) {
        if (!texture || width <= 0 || height <= 0 || channel < 0 || channel > 3) return false;
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        const std::vector<float> rgba = readTextureRgba(texture, width, height);
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) return false;
        stream << "Pf\n" << width << ' ' << height << "\n-1.0\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t index = (static_cast<std::size_t>(y) * width + x) * 4 + static_cast<std::size_t>(channel);
                stream.write(reinterpret_cast<const char*>(&rgba[index]), sizeof(float));
            }
        }
        return stream.good();
    }

    ImageMetrics calculateImageMetrics(const std::vector<float>& image, const std::vector<float>& reference) {
        ImageMetrics result;
        if (image.size() != reference.size() || image.empty()) {
            result.rmse = result.relativeRmse = result.psnr = result.maximumAbsolute = std::numeric_limits<double>::quiet_NaN();
            result.gradientRmse = result.gradientRelativeRmse = result.gradientTransfer = std::numeric_limits<double>::quiet_NaN();
            result.globalSsim = result.orderedPhaseResidualRatio = std::numeric_limits<double>::quiet_NaN();
            return result;
        }

        long double squaredError = 0.0;
        long double squaredReference = 0.0;
        double peak = 1e-12;
        double maximumAbsolute = 0.0;
        std::size_t channels = 0;
        for (std::size_t i = 0; i + 3 < image.size(); i += 4) {
            for (int c = 0; c < 3; ++c) {
                const double a = image[i + static_cast<std::size_t>(c)];
                const double b = reference[i + static_cast<std::size_t>(c)];
                const double d = a - b;
                squaredError += d * d;
                squaredReference += b * b;
                peak = std::max(peak, std::abs(b));
                maximumAbsolute = std::max(maximumAbsolute, std::abs(d));
                ++channels;
            }
        }
        const double mse = static_cast<double>(squaredError / std::max<std::size_t>(channels, 1));
        const double refMse = static_cast<double>(squaredReference / std::max<std::size_t>(channels, 1));
        result.rmse = std::sqrt(mse);
        result.relativeRmse = result.rmse / std::max(std::sqrt(refMse), 1e-12);
        result.psnr = mse <= 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10((peak * peak) / mse);
        result.maximumAbsolute = maximumAbsolute;
        return result;
    }

    void calculateGradientMetrics(const std::vector<float>& image, const std::vector<float>& reference, const int width, const int height, ImageMetrics& result) {
        if (image.size() != reference.size() || width < 2 || height < 2) {
            result.gradientRmse = result.gradientRelativeRmse = result.gradientTransfer = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        long double squaredError = 0.0;
        long double squaredReference = 0.0;
        std::size_t values = 0;
        const auto sample = [&](const std::vector<float>& data, const int x, const int y, const int c) {
            return static_cast<double>(data[(static_cast<std::size_t>(y) * width + x) * 4 + c]);
        };
        for (int y = 0; y < height - 1; ++y) {
            for (int x = 0; x < width - 1; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const double ax = sample(image, x + 1, y, c) - sample(image, x, y, c);
                    const double ay = sample(image, x, y + 1, c) - sample(image, x, y, c);
                    const double rx = sample(reference, x + 1, y, c) - sample(reference, x, y, c);
                    const double ry = sample(reference, x, y + 1, c) - sample(reference, x, y, c);
                    const double dx = ax - rx;
                    const double dy = ay - ry;
                    squaredError += dx * dx + dy * dy;
                    squaredReference += rx * rx + ry * ry;
                    values += 2;
                }
            }
        }
        const double mse = static_cast<double>(squaredError / std::max<std::size_t>(values, 1));
        const double refMse = static_cast<double>(squaredReference / std::max<std::size_t>(values, 1));
        result.gradientRmse = std::sqrt(mse);
        result.gradientRelativeRmse = result.gradientRmse / std::max(std::sqrt(refMse), 1e-12);

        long double methodGradientEnergy = 0.0;
        for (int y = 0; y < height - 1; ++y) {
            for (int x = 0; x < width - 1; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const double ax = sample(image, x + 1, y, c) - sample(image, x, y, c);
                    const double ay = sample(image, x, y + 1, c) - sample(image, x, y, c);
                    methodGradientEnergy += ax * ax + ay * ay;
                }
            }
        }
        result.gradientTransfer = std::sqrt(static_cast<double>(methodGradientEnergy / std::max(squaredReference, 1e-24L)));
    }

    double calculateGlobalLuminanceSsim(const std::vector<float>& image, const std::vector<float>& reference) {
        if (image.size() != reference.size() || image.empty()) return std::numeric_limits<double>::quiet_NaN();
        long double meanA = 0.0;
        long double meanB = 0.0;
        double peak = 1e-6;
        std::size_t samples = 0;
        for (std::size_t i = 0; i + 3 < image.size(); i += 4) {
            const double a = 0.2126 * image[i] + 0.7152 * image[i + 1] + 0.0722 * image[i + 2];
            const double b = 0.2126 * reference[i] + 0.7152 * reference[i + 1] + 0.0722 * reference[i + 2];
            meanA += a;
            meanB += b;
            peak = std::max(peak, std::abs(b));
            ++samples;
        }
        if (samples < 2) return 1.0;
        meanA /= samples;
        meanB /= samples;
        long double varA = 0.0;
        long double varB = 0.0;
        long double covariance = 0.0;
        for (std::size_t i = 0; i + 3 < image.size(); i += 4) {
            const long double a = 0.2126L * image[i] + 0.7152L * image[i + 1] + 0.0722L * image[i + 2];
            const long double b = 0.2126L * reference[i] + 0.7152L * reference[i + 1] + 0.0722L * reference[i + 2];
            const long double da = a - meanA;
            const long double db = b - meanB;
            varA += da * da;
            varB += db * db;
            covariance += da * db;
        }
        const long double denom = static_cast<long double>(samples - 1);
        varA /= denom;
        varB /= denom;
        covariance /= denom;
        const long double c1 = (0.01L * peak) * (0.01L * peak);
        const long double c2 = (0.03L * peak) * (0.03L * peak);
        const long double numerator = (2.0L * meanA * meanB + c1) * (2.0L * covariance + c2);
        const long double denominator = (meanA * meanA + meanB * meanB + c1) * (varA + varB + c2);
        return denominator > 0.0L ? static_cast<double>(numerator / denominator) : 1.0;
    }

    double calculateOrderedPhaseResidualRatio(
        const std::vector<float>& image,
        const std::vector<float>& reference,
        const int width,
        const int height,
        const int ditherMode
    ) {
        if (ditherMode == 1 || image.size() != reference.size() || width <= 0 || height <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const int patternSize = ditherMode == 3 ? 8 : (ditherMode == 4 ? 16 : 4);
        const int phaseTotal = patternSize * patternSize;
        std::vector<long double> phaseSum(static_cast<std::size_t>(phaseTotal), 0.0L);
        std::vector<std::size_t> phaseCount(static_cast<std::size_t>(phaseTotal), 0);
        long double totalSquared = 0.0;
        std::size_t total = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 4;
                const long double a = 0.2126L * image[i] + 0.7152L * image[i + 1] + 0.0722L * image[i + 2];
                const long double b = 0.2126L * reference[i] + 0.7152L * reference[i + 1] + 0.0722L * reference[i + 2];
                const long double error = a - b;
                const int phase = (y % patternSize) * patternSize + (x % patternSize);
                phaseSum[static_cast<std::size_t>(phase)] += error;
                ++phaseCount[static_cast<std::size_t>(phase)];
                totalSquared += error * error;
                ++total;
            }
        }
        if (total == 0 || totalSquared <= 1e-24L) return 0.0;
        long double phaseEnergy = 0.0;
        for (std::size_t phase = 0; phase < phaseSum.size(); ++phase) {
            if (phaseCount[phase] == 0) continue;
            const long double mean = phaseSum[phase] / static_cast<long double>(phaseCount[phase]);
            phaseEnergy += mean * mean * static_cast<long double>(phaseCount[phase]);
        }
        return std::clamp(static_cast<double>(phaseEnergy / totalSquared), 0.0, 1.0);
    }

    AlphaMetrics calculateAlphaMetrics(const std::vector<float>& estimateMeta, const std::vector<float>& referenceAlpha) {
        AlphaMetrics result;
        if (estimateMeta.size() != referenceAlpha.size() || estimateMeta.empty()) {
            result.rmse = result.mae = result.maximumAbsolute = std::numeric_limits<double>::quiet_NaN();
            return result;
        }
        long double squared = 0.0;
        long double absolute = 0.0;
        for (std::size_t i = 0; i + 3 < estimateMeta.size(); i += 4) {
            const double reference = referenceAlpha[i];
            const double support = estimateMeta[i + 1];
            if (reference <= 1e-6 && support < 0.5) continue;
            const double estimate = estimateMeta[i];
            const double error = estimate - reference;
            squared += error * error;
            absolute += std::abs(error);
            result.maximumAbsolute = std::max(result.maximumAbsolute, std::abs(error));
            ++result.samples;
        }
        if (result.samples == 0) return result;
        result.rmse = std::sqrt(static_cast<double>(squared / result.samples));
        result.mae = static_cast<double>(absolute / result.samples);
        return result;
    }

    AttributeMetrics calculateAttributeMetrics(
        const std::vector<float>& reconstructedAlbedo,
        const std::vector<float>& reconstructedNormal,
        const std::vector<float>& reconstructedPosition,
        const std::vector<float>& reconstructedControl,
        const std::vector<float>& referenceAlbedo,
        const std::vector<float>& referenceNormal,
        const std::vector<float>& referencePosition,
        const std::vector<float>& referenceControl
    ) {
        AttributeMetrics result;
        const std::size_t size = reconstructedAlbedo.size();
        if (size == 0 || reconstructedNormal.size() != size || reconstructedPosition.size() != size
            || reconstructedControl.size() != size || referenceAlbedo.size() != size
            || referenceNormal.size() != size || referencePosition.size() != size || referenceControl.size() != size) {
            return result;
        }

        long double albedoSquared = 0.0;
        long double roughnessSquared = 0.0;
        long double roughnessAbsolute = 0.0;
        long double positionSquared = 0.0;
        long double alphaSquared = 0.0;
        long double alphaAbsolute = 0.0;
        std::vector<double> normalAngles;
        normalAngles.reserve(size / 16);
        std::size_t falsePositive = 0;

        for (std::size_t i = 0; i + 3 < size; i += 4) {
            const bool referenceValid = referencePosition[i + 3] > 0.5 && referenceControl[i + 3] > 0.5;
            const bool reconstructedValid = reconstructedPosition[i + 3] > 0.5 && reconstructedControl[i + 3] > 0.5;
            if (referenceValid) ++result.referenceSamples;
            if (reconstructedValid) ++result.predictedSamples;
            if (!referenceValid) {
                if (reconstructedValid) ++falsePositive;
                continue;
            }
            if (!reconstructedValid) continue;

            const int referenceKey = static_cast<int>(std::floor(referenceControl[i + 3] + 0.5f));
            const int reconstructedKey = static_cast<int>(std::floor(reconstructedControl[i + 3] + 0.5f));
            if (referenceKey != reconstructedKey) {
                ++result.wrongSurfaceSamples;
                continue;
            }

            ++result.matchedSamples;
            for (int c = 0; c < 3; ++c) {
                const double d = static_cast<double>(reconstructedAlbedo[i + static_cast<std::size_t>(c)])
                    - static_cast<double>(referenceAlbedo[i + static_cast<std::size_t>(c)]);
                albedoSquared += d * d;
            }

            const double roughnessError = static_cast<double>(reconstructedAlbedo[i + 3])
                - static_cast<double>(referenceAlbedo[i + 3]);
            roughnessSquared += roughnessError * roughnessError;
            roughnessAbsolute += std::abs(roughnessError);

            const double px = static_cast<double>(reconstructedPosition[i]) - static_cast<double>(referencePosition[i]);
            const double py = static_cast<double>(reconstructedPosition[i + 1]) - static_cast<double>(referencePosition[i + 1]);
            const double pz = static_cast<double>(reconstructedPosition[i + 2]) - static_cast<double>(referencePosition[i + 2]);
            const double positionError = std::sqrt(px * px + py * py + pz * pz);
            positionSquared += positionError * positionError;
            result.positionMax = std::isfinite(result.positionMax) ? std::max(result.positionMax, positionError) : positionError;

            const double alphaError = static_cast<double>(reconstructedControl[i + 1])
                - static_cast<double>(referenceControl[i + 1]);
            alphaSquared += alphaError * alphaError;
            alphaAbsolute += std::abs(alphaError);
            result.alphaMax = std::isfinite(result.alphaMax) ? std::max(result.alphaMax, std::abs(alphaError)) : std::abs(alphaError);

            double rnx = static_cast<double>(reconstructedNormal[i]) * 2.0 - 1.0;
            double rny = static_cast<double>(reconstructedNormal[i + 1]) * 2.0 - 1.0;
            double rnz = static_cast<double>(reconstructedNormal[i + 2]) * 2.0 - 1.0;
            double enx = static_cast<double>(referenceNormal[i]) * 2.0 - 1.0;
            double eny = static_cast<double>(referenceNormal[i + 1]) * 2.0 - 1.0;
            double enz = static_cast<double>(referenceNormal[i + 2]) * 2.0 - 1.0;
            const double rLength = std::sqrt(rnx * rnx + rny * rny + rnz * rnz);
            const double eLength = std::sqrt(enx * enx + eny * eny + enz * enz);
            if (rLength > 1e-12 && eLength > 1e-12) {
                rnx /= rLength;
                rny /= rLength;
                rnz /= rLength;
                enx /= eLength;
                eny /= eLength;
                enz /= eLength;
                const double cosine = std::clamp(rnx * enx + rny * eny + rnz * enz, -1.0, 1.0);
                normalAngles.push_back(std::acos(cosine) * 57.29577951308232);
            }
        }

        result.matchRecall = result.referenceSamples == 0 ? 1.0
            : static_cast<double>(result.matchedSamples) / static_cast<double>(result.referenceSamples);
        result.falsePositiveRatio = result.predictedSamples == 0 ? 0.0
            : static_cast<double>(falsePositive) / static_cast<double>(result.predictedSamples);
        if (result.matchedSamples == 0) return result;

        const long double n = static_cast<long double>(result.matchedSamples);
        result.albedoRmse = std::sqrt(static_cast<double>(albedoSquared / (3.0L * n)));
        result.roughnessRmse = std::sqrt(static_cast<double>(roughnessSquared / n));
        result.roughnessMae = static_cast<double>(roughnessAbsolute / n);
        result.positionRmse = std::sqrt(static_cast<double>(positionSquared / n));
        result.alphaRmse = std::sqrt(static_cast<double>(alphaSquared / n));
        result.alphaMae = static_cast<double>(alphaAbsolute / n);

        if (!normalAngles.empty()) {
            std::sort(normalAngles.begin(), normalAngles.end());
            result.normalMeanDegrees = std::accumulate(normalAngles.begin(), normalAngles.end(), 0.0)
                / static_cast<double>(normalAngles.size());
            const std::size_t middle = normalAngles.size() / 2;
            result.normalMedianDegrees = normalAngles.size() % 2 == 0
                ? 0.5 * (normalAngles[middle - 1] + normalAngles[middle])
                : normalAngles[middle];
            const std::size_t p95 = std::min(
                normalAngles.size() - 1,
                static_cast<std::size_t>(std::ceil(static_cast<double>(normalAngles.size()) * 0.95)) - 1
            );
            result.normalP95Degrees = normalAngles[p95];
        }
        return result;
    }

    class AlphaResearch final : public gfx::research::Application {
    public:
        explicit AlphaResearch(LaunchOptions options) :
            Application({
                .title = "gfx-research-alpha",
                .width = 1600,
                .height = 900,
                .vsync = false
            }),
            launchOptions_(std::move(options)) {}

    private:
        bool on_init() override {
            root_ = GFX_RESEARCH_SOURCE_DIR;
            if (!loadShaders()) return false;
            if (!createTargets(config().width, config().height)) return false;

            plane_ = gfx::research::make_plane(20.0f, 8);
            card_ = gfx::research::make_plane(1.0f, 1);
            sphere_ = gfx::research::make_uv_sphere(1.0f, 72, 36);
            sponzaLoaded_ = sponza_.load(gfx::research::model_path(gfx::research::ModelAsset::Sponza));
            dragonLoaded_ = dragon_.load(gfx::research::model_path(gfx::research::ModelAsset::Dragon));
            suzanneLoaded_ = suzanne_.load(gfx::research::model_path(gfx::research::ModelAsset::Suzanne));
            teapotLoaded_ = teapot_.load(gfx::research::model_path(gfx::research::ModelAsset::Teapot));
            curtainTextureLoaded_ = curtainTexture_.load(gfx::research::asset_path("models/textures/sponza_curtain_blue_diff.tga"), true, true);
            foliageTextureLoaded_ = foliageTexture_.load(gfx::research::asset_path("models/textures/vase_plant.tga"), true, true);

            resetCamera(false);

            glEnable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            glDisable(GL_BLEND);

            renderWidth_ = config().width;
            renderHeight_ = config().height;
            if (launchOptions_.benchmarkEdgeDecision) startBenchmarkSuite(false, false, false, true);
            else if (launchOptions_.benchmarkPaper) startBenchmarkSuite(true, true);
            else if (launchOptions_.benchmarkSuite || launchOptions_.benchmarkQuick) startBenchmarkSuite(!launchOptions_.benchmarkQuick);
            return true;
        }

        void on_resize(const int width, const int height) override {
            if (benchmarkRunning_) return;
            createTargets(width, height);
            renderWidth_ = width;
            renderHeight_ = height;
        }

        void on_shutdown() override {
            if (overdrawInputTexture_) glDeleteTextures(1, &overdrawInputTexture_);
            if (overdrawSurvivorTexture_) glDeleteTextures(1, &overdrawSurvivorTexture_);
            overdrawInputTexture_ = 0;
            overdrawSurvivorTexture_ = 0;
        }

        bool loadShaders() {
            const gfx::research::ShaderFile opaqueFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/opaque_gbuffer.frag"}
            };
            const gfx::research::ShaderFile mixedOpaqueFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/opaque_mixed_gbuffer.frag"}
            };
            const gfx::research::ShaderFile mixedTransparentFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/transparent_mixed_gbuffer.frag"}
            };
            const gfx::research::ShaderFile mixedReconstructFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/reconstruct_mixed.frag"}
            };
            const gfx::research::ShaderFile attributeReconstructFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/reconstruct_transparent_attributes.frag"}
            };
            const gfx::research::ShaderFile attributeEdgeReconstructFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/reconstruct_transparent_attributes_edge.frag"}
            };
            const gfx::research::ShaderFile attributeReferenceFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/transparent_attribute_reference.frag"}
            };
            const gfx::research::ShaderFile attributeCompositeFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/composite_reconstructed_transparency.frag"}
            };
            const gfx::research::ShaderFile twoLayerDirectCompositeFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/composite_reconstructed_layers_direct.frag"}
            };
            const gfx::research::ShaderFile twoLayerCompositeFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/composite_reconstructed_layers.frag"}
            };
            const gfx::research::ShaderFile rawDitherFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/raw_dither.frag"}
            };
            const gfx::research::ShaderFile supportFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/layered_wboit_support.frag"}
            };
            const gfx::research::ShaderFile lightingFiles[] = {
                {gfx::research::ShaderStage::Compute, root_ / "shaders/deferred_lighting_full.comp"}
            };
            const gfx::research::ShaderFile forwardFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/forward_reference.frag"}
            };
            const gfx::research::ShaderFile depthPeelFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/depth_peel.frag"}
            };
            const gfx::research::ShaderFile peelAccumulateFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/depth_peel_accumulate.frag"}
            };
            const gfx::research::ShaderFile peelResolveFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/depth_peel_resolve.frag"}
            };
            const gfx::research::ShaderFile wboitFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/transparent_wboit.frag"}
            };
            const gfx::research::ShaderFile wboitResolveFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/wboit_resolve.frag"}
            };
            const gfx::research::ShaderFile layeredFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Geometry, root_ / "shaders/layered_wboit.geom"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/transparent_layered_wboit.frag"}
            };
            const gfx::research::ShaderFile globalRevealFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/global_reveal.frag"}
            };
            const gfx::research::ShaderFile layeredResolveFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/layered_wboit_resolve.frag"}
            };
            const gfx::research::ShaderFile alphaReferenceFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/alpha_reference.frag"}
            };
            const gfx::research::ShaderFile overdrawFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/scene.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/transparent_overdraw.frag"}
            };
            const gfx::research::ShaderFile displayFiles[] = {
                {gfx::research::ShaderStage::Vertex, root_ / "shaders/fullscreen.vert"},
                {gfx::research::ShaderStage::Fragment, root_ / "shaders/display.frag"}
            };
            return opaqueShader_.load(opaqueFiles)
                && mixedOpaqueGBufferShader_.load(mixedOpaqueFiles)
                && mixedTransparentGBufferShader_.load(mixedTransparentFiles)
                && mixedReconstructShader_.load(mixedReconstructFiles)
                && attributeReconstructShader_.load(attributeReconstructFiles)
                && attributeEdgeReconstructShader_.load(attributeEdgeReconstructFiles)
                && attributeReferenceShader_.load(attributeReferenceFiles)
                && attributeCompositeShader_.load(attributeCompositeFiles)
                && twoLayerCompositeShader_.load(twoLayerCompositeFiles)
                && twoLayerDirectCompositeShader_.load(twoLayerDirectCompositeFiles)
                && rawDitherShader_.load(rawDitherFiles)
                && layeredSupportShader_.load(supportFiles)
                && deferredLightingShader_.load(lightingFiles)
                && forwardShader_.load(forwardFiles)
                && depthPeelShader_.load(depthPeelFiles)
                && depthPeelAccumulateShader_.load(peelAccumulateFiles)
                && depthPeelResolveShader_.load(peelResolveFiles)
                && wboitShader_.load(wboitFiles)
                && wboitResolveShader_.load(wboitResolveFiles)
                && layeredWboitShader_.load(layeredFiles)
                && globalRevealShader_.load(globalRevealFiles)
                && layeredResolveShader_.load(layeredResolveFiles)
                && alphaReferenceShader_.load(alphaReferenceFiles)
                && overdrawShader_.load(overdrawFiles)
                && displayShader_.load(displayFiles);
        }

        bool createTargets(const int width, const int height) {
            if (width <= 0 || height <= 0) return false;
            const gfx::research::ColorAttachmentDesc opaqueGBufferColors[] = {
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST}
            };
            const gfx::research::ColorAttachmentDesc mixedGBufferColors[] = {
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST}
            };
            const gfx::research::ColorAttachmentDesc hdrColor[] = {
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST}
            };
            const gfx::research::ColorAttachmentDesc supportColor[] = {
                {GL_R32F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA8, GL_NEAREST, GL_NEAREST}
            };
            const gfx::research::ColorAttachmentDesc resultColors[] = {
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST}
            };

            return opaqueGBuffer_.create(width, height, opaqueGBufferColors, true, GL_DEPTH_COMPONENT32F)
                && opaqueLit_.create(width, height, hdrColor, false)
                && layeredSupportTarget_.create(width, height, supportColor, true, GL_DEPTH_COMPONENT32F)
                && mixedGBuffer_.create(width, height, mixedGBufferColors, true, GL_DEPTH_COMPONENT32F)
                && mixedLit_.create(width, height, hdrColor, false)
                && reconstructedTransparentGBuffer_.create(width, height, mixedGBufferColors, false)
                && reconstructedTransparentLit_.create(width, height, hdrColor, false)
                && reconstructedTransparentLayer1GBuffer_.create(width, height, mixedGBufferColors, false)
                && reconstructedTransparentLayer1Lit_.create(width, height, hdrColor, false)
                && resultTarget_.create(width, height, resultColors, true, GL_DEPTH_COMPONENT32F)
                && referenceTarget_.create(width, height, hdrColor, false)
                && depthPeelA_.create(width, height, hdrColor, true, GL_DEPTH_COMPONENT32F)
                && depthPeelB_.create(width, height, hdrColor, true, GL_DEPTH_COMPONENT32F)
                && depthPeelAccum_.create(width, height, hdrColor, false)
                && alphaReferenceTarget_.create(width, height, hdrColor, true, GL_DEPTH_COMPONENT32F)
                && wboitTarget_.create(width, height)
                && layeredWboitTarget_.create(width, height)
                && createOverdrawTargets(width, height);
        }


        

        

        

        bool createOverdrawTargets(const int width, const int height) {
            auto recreate = [&](unsigned int& texture) {
                if (texture) glDeleteTextures(1, &texture);
                glCreateTextures(GL_TEXTURE_2D, 1, &texture);
                glTextureStorage2D(texture, 1, GL_R32UI, width, height);
                glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            };
            recreate(overdrawInputTexture_);
            recreate(overdrawSurvivorTexture_);
            return overdrawInputTexture_ != 0 && overdrawSurvivorTexture_ != 0;
        }

        void setObjectUniforms(
            gfx::research::Shader& shader,
            const glm::mat4& model,
            const glm::mat4& viewProjection,
            const glm::vec3& color,
            const float roughness,
            const float alpha,
            const int alphaMode
        ) {
            shader.bind();
            shader.set_mat4("uModel", model);
            shader.set_mat4("uMvp", viewProjection * model);
            shader.set_vec3("uEye", camera_.position());
            shader.set_vec3("uBaseColor", color);
            shader.set_float("uRoughness", roughness);
            shader.set_float("uAlpha", alpha);
            shader.set_int("uAlphaMode", alphaMode);
            shader.set_int("uLightCount", lightCount_);
            shader.set_int("uUseBaseTexture", 0);
            shader.set_uint("uDitherSeed", 0u);
        }

        void drawOpaqueMesh(
            gfx::research::Shader& shader,
            const gfx::research::Mesh& mesh,
            const glm::mat4& model,
            const glm::vec3& color,
            const float roughness,
            const int pattern,
            const glm::mat4& viewProjection
        ) {
            setObjectUniforms(shader, model, viewProjection, color, roughness, 1.0f, 0);
            shader.set_int("uPattern", pattern);
            mesh.draw();
        }

        void drawOpaqueModel(
            gfx::research::Shader& shader,
            const gfx::research::Model& modelData,
            const glm::mat4& model,
            const glm::vec3& color,
            const float roughness,
            const int pattern,
            const glm::mat4& viewProjection
        ) {
            setObjectUniforms(shader, model, viewProjection, color, roughness, 1.0f, 0);
            shader.set_int("uPattern", pattern);
            modelData.draw();
        }

        void drawOpaqueScene(gfx::research::Shader& shader, const glm::mat4& viewProjection) {
            if (!showSponza_) {
                drawOpaqueMesh(
                    shader,
                    plane_,
                    glm::translate(glm::mat4(1.0f), {0.0f, -0.02f, 0.0f}),
                    {0.16f, 0.17f, 0.19f},
                    0.58f,
                    1,
                    viewProjection
                );
            }

            if (showSponza_ && sponzaLoaded_) {
                const glm::mat4 model = fittedSceneTransform(sponza_, SPONZA_GROUND_POSITION, SPONZA_TARGET_SIZE, SPONZA_YAW_RADIANS);
                shader.bind();
                shader.set_mat4("uModel", model);
                shader.set_mat4("uMvp", viewProjection * model);
                shader.set_vec3("uEye", camera_.position());
                shader.set_int("uPattern", 0);
                sponza_.draw(shader);
            }
            if (dragonLoaded_) {
                drawOpaqueModel(
                    shader,
                    dragon_,
                    fittedModelTransform(dragon_, {-2.55f, 0.0f, -1.35f}, 1.35f, 0.35f),
                    {0.30f, 0.12f, 0.08f},
                    0.28f,
                    0,
                    viewProjection
                );
            }
            if (suzanneLoaded_) {
                drawOpaqueModel(
                    shader,
                    suzanne_,
                    fittedModelTransform(suzanne_, {2.45f, 0.0f, -1.45f}, 1.25f, -0.35f),
                    {0.10f, 0.30f, 0.18f},
                    0.34f,
                    0,
                    viewProjection
                );
            }
            if (teapotLoaded_) {
                drawOpaqueModel(
                    shader,
                    teapot_,
                    fittedModelTransform(teapot_, {0.0f, 0.0f, -2.85f}, 1.10f, 0.0f),
                    {0.18f, 0.16f, 0.34f},
                    0.22f,
                    0,
                    viewProjection
                );
            }
        }

        template<typename DrawFn>
        void drawTransparentObjects(gfx::research::Shader& shader, const glm::mat4& viewProjection, DrawFn&& draw, const int onlyObject = -1) {
            int objectIndex = 0;
            const auto submit = [&](const gfx::research::Mesh& mesh, glm::mat4 model, glm::vec3 color, float roughness, float alpha, int alphaMode, const gfx::research::Texture2D* texture, unsigned int ditherSeed) {
                const int currentObject = objectIndex++;
                if (onlyObject >= 0 && currentObject != onlyObject) return;
                model = glm::translate(glm::mat4(1.0f), transparentMotionOffset_) * model;
                setObjectUniforms(shader, model, viewProjection, color, roughness, alpha, alphaMode);
                shader.set_int("uDitherMode", ditherMode_);
                shader.set_int("uCoverageEncodingMode", coverageEncodingMode_);
                shader.set_float("uCoverageBias", coverageBias_);
                shader.set_uint("uDitherSeed", ditherSeed);
                shader.set_uint("uSurfaceId", static_cast<unsigned int>(currentObject + 1));
                const bool useTexture = texture && texture->id() != 0;
                shader.set_int("uUseBaseTexture", useTexture ? 1 : 0);
                if (useTexture) {
                    texture->bind(3);
                }
                draw(mesh);
            };

            const auto verticalCard = [&](const glm::vec3& position, const glm::vec2& size, const float yawAroundY, const glm::vec3& color, const float roughness, const float alpha, const int alphaMode, const gfx::research::Texture2D* texture, unsigned int ditherSeed) {
                glm::mat4 model(1.0f);
                model = glm::translate(model, position);
                // The card mesh starts in the XZ plane. Rotate it upright first, then
                // apply the user-controlled yaw around the world Y axis.
                model = glm::rotate(model, yawAroundY, glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::rotate(model, 1.57079632679f, glm::vec3(1.0f, 0.0f, 0.0f));
                model = glm::scale(model, glm::vec3(size.x, 1.0f, size.y));
                submit(card_, model, color, roughness, alpha, alphaMode, texture, ditherSeed);
            };

            const gfx::research::Texture2D* curtain = curtainTextureLoaded_ ? &curtainTexture_ : nullptr;
            const gfx::research::Texture2D* foliage = foliageTextureLoaded_ ? &foliageTexture_ : nullptr;

            if (transparentGeometry_ == 0 || transparentGeometry_ == 4) {
                verticalCard({1.45f, 1.45f, 0.04f}, {3.2f, 2.6f}, 1.57079632679f, {0.06f, 0.82f, 1.00f}, 0.08f, alpha_, alphaMode_, curtain, 11u);
            }

            if (transparentGeometry_ == 1 || transparentGeometry_ == 4) {
                glm::mat4 model(1.0f);
                model = glm::translate(model, {1.60f, 1.30f, 0.02f});
                model = glm::scale(model, glm::vec3(0.72f));
                submit(sphere_, model, {1.00f, 0.28f, 0.06f}, 0.16f, alpha_, alphaMode_, nullptr, 23u);
            }

            if (transparentGeometry_ == 2 || transparentGeometry_ == 4) {
                verticalCard({1.10f, 1.30f, -0.85f}, {1.55f, 2.00f}, 1.20f, {0.12f, 1.00f, 0.28f}, 0.46f, alpha_, 2, foliage, 37u);
            }

            if (transparentGeometry_ == 3) {
                verticalCard({2.05f, 1.35f, 0.22f}, {1.80f, 2.25f}, 1.57079632679f, {0.00f, 0.75f, 1.00f}, 0.12f, alpha_, alphaMode_, curtain, 41u);
                verticalCard({1.30f, 1.35f, -0.10f}, {1.80f, 2.25f}, 1.57079632679f, {1.00f, 0.24f, 0.00f}, 0.28f, alpha_, alphaMode_, curtain, 53u);
            }

            if (transparentGeometry_ == 4) {
                verticalCard({1.78f, 1.18f, 0.42f}, {1.20f, 1.45f}, 0.78539816339f, {0.00f, 0.75f, 1.00f}, 0.12f, alpha_, alphaMode_, curtain, 41u);
                verticalCard({1.18f, 1.34f, -0.34f}, {1.05f, 1.35f}, 1.57079632679f, {1.00f, 0.24f, 0.00f}, 0.28f, alpha_, alphaMode_, curtain, 53u);
                for (int i = 0; i < 5; ++i) {
                    verticalCard(
                        {1.15f + 0.32f * static_cast<float>(i % 2), 0.82f + 0.42f * static_cast<float>(i), -0.85f - 0.22f * static_cast<float>(i)},
                        {0.92f, 0.92f},
                        -0.25f + 0.13f * static_cast<float>(i),
                        {0.18f, 1.00f, 0.22f},
                        0.46f,
                        alpha_,
                        2,
                        foliage,
                        101u + static_cast<unsigned int>(i)
                    );
                }
            }

            if (transparentGeometry_ == 5) {
                verticalCard({1.86f, 1.18f, 0.46f}, {1.25f, 1.50f}, 0.78539816339f, {0.00f, 0.75f, 1.00f}, 0.10f, alpha_, alphaMode_, curtain, 141u);
                verticalCard({1.12f, 1.32f, -0.36f}, {1.05f, 1.35f}, 1.57079632679f, {1.00f, 0.24f, 0.00f}, 0.24f, alpha_, alphaMode_, curtain, 153u);
            }

            if (transparentGeometry_ == 6) {
                const int count = std::clamp(transparentLayers_, 1, MAX_TRANSPARENT_LAYERS);
                const float half = 0.5f * static_cast<float>(count - 1);
                for (int i = count - 1; i >= 0; --i) {
                    const float fi = static_cast<float>(i);
                    const float centered = fi - half;
                    const glm::vec3 color(
                        0.35f + 0.55f * (0.5f + 0.5f * std::sin(fi * 1.7f)),
                        0.30f + 0.60f * (0.5f + 0.5f * std::sin(fi * 2.3f + 1.0f)),
                        0.35f + 0.55f * (0.5f + 0.5f * std::sin(fi * 1.1f + 2.0f))
                    );
                    verticalCard(
                        {1.58f + 0.04f * centered, 1.28f + 0.015f * centered, 0.12f - 0.095f * fi},
                        {1.10f, 1.24f},
                        1.57079632679f,
                        color,
                        0.18f + 0.02f * fi,
                        alpha_,
                        alphaMode_,
                        (i & 1) == 0 ? curtain : nullptr,
                        200u + static_cast<unsigned int>(i)
                    );
                }
            }
        }

        std::vector<glm::vec3> transparentObjectCenters() const {
            std::vector<glm::vec3> centers;
            if (transparentGeometry_ == 0 || transparentGeometry_ == 4) centers.push_back({1.45f, 1.45f, 0.04f});
            if (transparentGeometry_ == 1 || transparentGeometry_ == 4) centers.push_back({1.60f, 1.30f, 0.02f});
            if (transparentGeometry_ == 2 || transparentGeometry_ == 4) centers.push_back({1.10f, 1.30f, -0.85f});
            if (transparentGeometry_ == 3) {
                centers.push_back({2.05f, 1.35f, 0.22f});
                centers.push_back({1.30f, 1.35f, -0.10f});
            }
            if (transparentGeometry_ == 4) {
                centers.push_back({1.78f, 1.18f, 0.42f});
                centers.push_back({1.18f, 1.34f, -0.34f});
                for (int i = 0; i < 5; ++i) {
                    centers.push_back({
                        1.15f + 0.32f * static_cast<float>(i % 2),
                        0.82f + 0.42f * static_cast<float>(i),
                        -0.85f - 0.22f * static_cast<float>(i)
                    });
                }
            }
            if (transparentGeometry_ == 5) {
                centers.push_back({1.86f, 1.18f, 0.46f});
                centers.push_back({1.12f, 1.32f, -0.36f});
            }
            if (transparentGeometry_ == 6) {
                const int count = std::clamp(transparentLayers_, 1, MAX_TRANSPARENT_LAYERS);
                const float half = 0.5f * static_cast<float>(count - 1);
                for (int i = count - 1; i >= 0; --i) {
                    const float fi = static_cast<float>(i);
                    const float centered = fi - half;
                    centers.push_back({1.58f + 0.04f * centered, 1.28f + 0.015f * centered, 0.12f - 0.095f * fi});
                }
            }
            for (glm::vec3& center : centers) center += transparentMotionOffset_;
            return centers;
        }

        std::vector<int> transparentBackToFrontOrder() const {
            const std::vector<glm::vec3> centers = transparentObjectCenters();
            std::vector<int> order(centers.size());
            std::iota(order.begin(), order.end(), 0);
            const glm::vec3 eye = camera_.position();
            std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
                const glm::vec3 da = centers[static_cast<std::size_t>(a)] - eye;
                const glm::vec3 db = centers[static_cast<std::size_t>(b)] - eye;
                return glm::dot(da, da) > glm::dot(db, db);
            });
            return order;
        }

        void clearLightingTarget(gfx::research::Framebuffer& target) {
            target.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            const float clear[] = {0.0f, 0.0f, 0.0f, 0.0f};
            glClearBufferfv(GL_COLOR, 0, clear);
            gfx::research::Framebuffer::bind_default();
        }

        void bindDeferredLighting(
            gfx::research::Framebuffer& gBuffer,
            gfx::research::Framebuffer& output,
            const bool
        ) {
            deferredLightingShader_.bind();
            glBindTextureUnit(0, gBuffer.color(0));
            glBindTextureUnit(1, gBuffer.color(1));
            glBindTextureUnit(2, gBuffer.color(2));
            glBindImageTexture(0, output.color(0), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            deferredLightingShader_.set_int("uAlbedo", 0);
            deferredLightingShader_.set_int("uNormal", 1);
            deferredLightingShader_.set_int("uPosition", 2);
            deferredLightingShader_.set_vec3("uEye", camera_.position());
            deferredLightingShader_.set_int("uLightCount", lightCount_);
        }

        void renderFullRateDeferredLighting(gfx::research::Framebuffer& gBuffer, gfx::research::Framebuffer& output) {
            clearLightingTarget(output);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            bindDeferredLighting(gBuffer, output, false);

            constexpr unsigned int localSize = 256u;
            constexpr unsigned int maxGroupsX = 65535u;
            const std::size_t pixelCount = static_cast<std::size_t>(renderWidth_) * static_cast<std::size_t>(renderHeight_);
            const unsigned int totalGroups = static_cast<unsigned int>((pixelCount + localSize - 1u) / localSize);
            if (totalGroups > 0u) {
                const unsigned int groupsX = std::min(totalGroups, maxGroupsX);
                const unsigned int groupsY = (totalGroups + groupsX - 1u) / groupsX;
                glDispatchCompute(groupsX, groupsY, 1u);
            }
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        }

        

        

        

        

        

        

        void renderOpaque(const glm::mat4& viewProjection) {
            opaqueGBuffer_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
            const float clearAlbedo[] = {0.02f, 0.025f, 0.035f, 1.0f};
            const float clearNormal[] = {0.5f, 0.5f, 1.0f, 0.0f};
            const float clearPosition[] = {0.0f, 0.0f, 0.0f, 0.0f};
            glClearBufferfv(GL_COLOR, 0, clearAlbedo);
            glClearBufferfv(GL_COLOR, 1, clearNormal);
            glClearBufferfv(GL_COLOR, 2, clearPosition);
            glClear(GL_DEPTH_BUFFER_BIT);
            drawOpaqueScene(opaqueShader_, viewProjection);
            renderFullRateDeferredLighting(opaqueGBuffer_, opaqueLit_);
        }

        void renderLayeredSupport(const glm::mat4& viewProjection, const int onlyObject = -1) {
            glBlitNamedFramebuffer(
                opaqueGBuffer_.id(), layeredSupportTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_DEPTH_BUFFER_BIT, GL_NEAREST
            );
            layeredSupportTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            const float clearSupport[] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float clearGuide[] = {0.0f, 0.0f, 0.0f, 0.0f};
            glClearBufferfv(GL_COLOR, 0, clearSupport);
            glClearBufferfv(GL_COLOR, 1, clearGuide);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
            drawTransparentObjects(layeredSupportShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); }, onlyObject);
        }


        

        void renderMixedDitheredGBuffer(const glm::mat4& viewProjection, const bool preserveOpaqueAttributes = false) {
            // This is the proposed representation.
            //
            // Opaque and dithered-alpha geometry write the SAME ordinary 2D deferred
            // attachments and the SAME depth attachment. There is no separate
            // transparent G-buffer and no layered scene array in this path.
            //
            // A transparent fragment that fails the screen-door test executes discard,
            // therefore it writes neither depth nor any G-buffer attachment. The opaque
            // or farther surviving scene sample already present at that pixel remains
            // coherent in albedo, normal, position and depth.
            mixedGBuffer_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);

            const GLenum buffers[] = {
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
                GL_COLOR_ATTACHMENT3
            };
            glNamedFramebufferDrawBuffers(mixedGBuffer_.id(), 4, buffers);

            const float clearAlbedo[] = {0.02f, 0.025f, 0.035f, 1.0f};
            const float clearNormal[] = {0.5f, 0.5f, 1.0f, 0.0f};
            const float clearPosition[] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float clearControl[] = {1.0f, 1.0f, 0.0f, 0.0f};
            glClearBufferfv(GL_COLOR, 0, clearAlbedo);
            glClearBufferfv(GL_COLOR, 1, clearNormal);
            glClearBufferfv(GL_COLOR, 2, clearPosition);
            glClearBufferfv(GL_COLOR, 3, clearControl);
            glClear(GL_DEPTH_BUFFER_BIT);

            // 1) Ordinary opaque scene.
            drawOpaqueScene(mixedOpaqueGBufferShader_, viewProjection);

            // The attribute-space methods need the exact opaque background under pixels
            // later overwritten by transparent survivors. Preserve only the unlit opaque
            // attributes here; ALL lighting remains deferred until after transparent
            // attribute reconstruction. This avoids a duplicate opaque geometry pass.
            if (preserveOpaqueAttributes) {
                for (int attachment = 0; attachment < 3; ++attachment) {
                    glNamedFramebufferReadBuffer(mixedGBuffer_.id(), GL_COLOR_ATTACHMENT0 + attachment);
                    glNamedFramebufferDrawBuffer(opaqueGBuffer_.id(), GL_COLOR_ATTACHMENT0 + attachment);
                    glBlitNamedFramebuffer(
                        mixedGBuffer_.id(), opaqueGBuffer_.id(),
                        0, 0, renderWidth_, renderHeight_,
                        0, 0, renderWidth_, renderHeight_,
                        GL_COLOR_BUFFER_BIT, GL_NEAREST
                    );
                }
                glBlitNamedFramebuffer(
                    mixedGBuffer_.id(), opaqueGBuffer_.id(),
                    0, 0, renderWidth_, renderHeight_,
                    0, 0, renderWidth_, renderHeight_,
                    GL_DEPTH_BUFFER_BIT, GL_NEAREST
                );
                const GLenum opaqueBuffers[] = {
                    GL_COLOR_ATTACHMENT0,
                    GL_COLOR_ATTACHMENT1,
                    GL_COLOR_ATTACHMENT2
                };
                glNamedFramebufferDrawBuffers(opaqueGBuffer_.id(), 3, opaqueBuffers);
                glNamedFramebufferDrawBuffers(mixedGBuffer_.id(), 4, buffers);
                mixedGBuffer_.bind();
            }

            // 2) Dithered-alpha objects go straight into the very same MRT/depth.
            // Depth writes intentionally stay enabled: the final G-buffer is a single
            // visibility sample per pixel selected from the complete scene.
            mixedTransparentGBufferShader_.bind();
            drawTransparentObjects(
                mixedTransparentGBufferShader_,
                viewProjection,
                [](const gfx::research::Mesh& mesh) { mesh.draw(); }
            );

            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        }

        void renderMixedReconstruction(const int methodOverride = -1) {
            const GLenum resultBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(resultTarget_.id(), 2, resultBuffers);
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);

            mixedReconstructShader_.bind();
            glBindTextureUnit(0, mixedLit_.color(0));
            glBindTextureUnit(1, mixedGBuffer_.color(0));
            glBindTextureUnit(2, mixedGBuffer_.color(1));
            glBindTextureUnit(3, mixedGBuffer_.color(2));
            glBindTextureUnit(4, mixedGBuffer_.color(3));
            mixedReconstructShader_.set_int("uLit", 0);
            mixedReconstructShader_.set_int("uAlbedo", 1);
            mixedReconstructShader_.set_int("uNormal", 2);
            mixedReconstructShader_.set_int("uPosition", 3);
            mixedReconstructShader_.set_int("uControl", 4);
            mixedReconstructShader_.set_int("uMethod", methodOverride >= 0 ? methodOverride : method_);
            mixedReconstructShader_.set_int("uKernelRadius", kernelRadius_);
            mixedReconstructShader_.set_int("uDitherMode", ditherMode_);
            mixedReconstructShader_.set_int("uCoverageEncodingMode", coverageEncodingMode_);
            mixedReconstructShader_.set_float("uCoverageBias", coverageBias_);
            mixedReconstructShader_.set_int("uRoddAblationMask", roddAblationMask_);
            mixedReconstructShader_.set_float("uDepthSigma", depthSigma_);
            mixedReconstructShader_.set_float("uAlbedoSigma", albedoSigma_);
            mixedReconstructShader_.set_float("uDetailStrength", detailStrength_);
            fullscreen_.draw();

            glDepthMask(GL_TRUE);
        }

        


        

        

        void renderReconstructedTransparentAttributes(
            gfx::research::Framebuffer& output,
            const int layerIndex,
            const bool edgeAware
        ) {
            const GLenum drawBuffers[] = {
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
                GL_COLOR_ATTACHMENT3
            };
            glNamedFramebufferDrawBuffers(output.id(), 4, drawBuffers);
            output.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);

            const float clearAlbedo[] = {0.0f, 0.0f, 0.0f, 0.5f};
            const float clearNormal[] = {0.5f, 0.5f, 1.0f, 0.0f};
            const float clearPosition[] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float clearControl[] = {0.0f, 0.0f, 0.0f, 0.0f};
            glClearBufferfv(GL_COLOR, 0, clearAlbedo);
            glClearBufferfv(GL_COLOR, 1, clearNormal);
            glClearBufferfv(GL_COLOR, 2, clearPosition);
            glClearBufferfv(GL_COLOR, 3, clearControl);

            gfx::research::Shader& shader = edgeAware ? attributeEdgeReconstructShader_ : attributeReconstructShader_;
            shader.bind();
            glBindTextureUnit(0, mixedGBuffer_.color(0));
            glBindTextureUnit(1, mixedGBuffer_.color(1));
            glBindTextureUnit(2, mixedGBuffer_.color(2));
            glBindTextureUnit(3, mixedGBuffer_.color(3));
            shader.set_int("uMixedAlbedo", 0);
            shader.set_int("uMixedNormal", 1);
            shader.set_int("uMixedPosition", 2);
            shader.set_int("uMixedControl", 3);
            shader.set_int("uKernelRadius", kernelRadius_);
            shader.set_int("uDitherMode", ditherMode_);
            shader.set_int("uCoverageEncodingMode", coverageEncodingMode_);
            shader.set_float("uCoverageBias", coverageBias_);
            shader.set_float("uDepthSigma", depthSigma_);
            shader.set_float("uAlbedoSigma", albedoSigma_);
            shader.set_int("uLayerIndex", layerIndex);
            fullscreen_.draw();
            glDepthMask(GL_TRUE);
        }

        void renderAttributeComposite() {
            const GLenum resultBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(resultTarget_.id(), 2, resultBuffers);
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);

            attributeCompositeShader_.bind();
            glBindTextureUnit(0, opaqueLit_.color(0));
            glBindTextureUnit(1, reconstructedTransparentLit_.color(0));
            glBindTextureUnit(2, reconstructedTransparentGBuffer_.color(2));
            glBindTextureUnit(3, reconstructedTransparentGBuffer_.color(3));
            attributeCompositeShader_.set_int("uOpaqueLit", 0);
            attributeCompositeShader_.set_int("uTransparentLit", 1);
            attributeCompositeShader_.set_int("uTransparentPosition", 2);
            attributeCompositeShader_.set_int("uTransparentControl", 3);
            fullscreen_.draw();
            glDepthMask(GL_TRUE);
        }

        void renderTwoLayerAttributeComposite(const bool directComposite) {
            const GLenum resultBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(resultTarget_.id(), 2, resultBuffers);
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);

            gfx::research::Shader& shader = directComposite ? twoLayerDirectCompositeShader_ : twoLayerCompositeShader_;
            shader.bind();
            glBindTextureUnit(0, opaqueLit_.color(0));
            glBindTextureUnit(1, reconstructedTransparentLit_.color(0));
            glBindTextureUnit(2, reconstructedTransparentGBuffer_.color(2));
            glBindTextureUnit(3, reconstructedTransparentGBuffer_.color(3));
            glBindTextureUnit(4, reconstructedTransparentLayer1Lit_.color(0));
            glBindTextureUnit(5, reconstructedTransparentLayer1GBuffer_.color(2));
            glBindTextureUnit(6, reconstructedTransparentLayer1GBuffer_.color(3));
            shader.set_int("uOpaqueLit", 0);
            shader.set_int("uFrontLit", 1);
            shader.set_int("uFrontPosition", 2);
            shader.set_int("uFrontControl", 3);
            shader.set_int("uBackLit", 4);
            shader.set_int("uBackPosition", 5);
            shader.set_int("uBackControl", 6);
            fullscreen_.draw();
            glDepthMask(GL_TRUE);
        }

        void renderDeferredAttributeReconstruction(const glm::mat4& viewProjection) {
            renderMixedDitheredGBuffer(viewProjection, true);
            renderReconstructedTransparentAttributes(reconstructedTransparentGBuffer_, 0, false);
            const bool timeLighting = benchmarkRunning_ && currentBenchmarkCase().recordDeferredLighting;
            if (timeLighting) deferredLightingTimer_.begin();
            renderFullRateDeferredLighting(opaqueGBuffer_, opaqueLit_);
            renderFullRateDeferredLighting(reconstructedTransparentGBuffer_, reconstructedTransparentLit_);
            if (timeLighting) deferredLightingTimer_.end();
            renderAttributeComposite();
        }

        void renderDeferredTwoLayerReconstruction(
            const glm::mat4& viewProjection,
            const bool edgeAware,
            const bool directComposite
        ) {
            renderMixedDitheredGBuffer(viewProjection, true);
            renderReconstructedTransparentAttributes(reconstructedTransparentGBuffer_, 0, edgeAware);
            renderReconstructedTransparentAttributes(reconstructedTransparentLayer1GBuffer_, 1, edgeAware);

            // No fragment has been lit before this point. Opaque, front transparent and
            // reconstructed back transparent attributes are now shaded as post-processes.
            const bool timeLighting = benchmarkRunning_ && currentBenchmarkCase().recordDeferredLighting;
            if (timeLighting) deferredLightingTimer_.begin();
            renderFullRateDeferredLighting(opaqueGBuffer_, opaqueLit_);
            renderFullRateDeferredLighting(reconstructedTransparentGBuffer_, reconstructedTransparentLit_);
            renderFullRateDeferredLighting(reconstructedTransparentLayer1GBuffer_, reconstructedTransparentLayer1Lit_);
            if (timeLighting) deferredLightingTimer_.end();

            renderTwoLayerAttributeComposite(directComposite);
        }

        void renderFullDeferredLightControl(const glm::mat4& viewProjection) {
            // Same mixed G-buffer and the same deferred lighting as proposed, but no
            // de-dithering reconstruction. This isolates the post-lighting decoder.
            renderMixedDitheredGBuffer(viewProjection);
            const bool timeLighting = benchmarkRunning_ && currentBenchmarkCase().recordDeferredLighting;
            if (timeLighting) deferredLightingTimer_.begin();
            renderFullRateDeferredLighting(mixedGBuffer_, mixedLit_);
            if (timeLighting) deferredLightingTimer_.end();
            renderMixedReconstruction(METHOD_RAW_DITHER);
        }

        void renderRawDither(const glm::mat4& viewProjection) {
            glNamedFramebufferReadBuffer(opaqueLit_.id(), GL_COLOR_ATTACHMENT0);
            glNamedFramebufferDrawBuffer(resultTarget_.id(), GL_COLOR_ATTACHMENT0);
            glBlitNamedFramebuffer(
                opaqueLit_.id(), resultTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_COLOR_BUFFER_BIT, GL_NEAREST
            );
            glBlitNamedFramebuffer(
                opaqueGBuffer_.id(), resultTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_DEPTH_BUFFER_BIT, GL_NEAREST
            );
            const float clearMeta[] = {0.0f, 0.0f, 0.0f, 0.0f};
            glClearNamedFramebufferfv(resultTarget_.id(), GL_COLOR, 1, clearMeta);
            const GLenum resultBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(resultTarget_.id(), 2, resultBuffers);
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
            rawDitherShader_.bind();
            rawDitherShader_.set_int("uOpaqueDepth", 0);
            glBindTextureUnit(0, opaqueGBuffer_.depth());
            drawTransparentObjects(rawDitherShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });
        }

        void renderClassicForward(const glm::mat4& viewProjection) {
            glNamedFramebufferReadBuffer(opaqueLit_.id(), GL_COLOR_ATTACHMENT0);
            glNamedFramebufferDrawBuffer(resultTarget_.id(), GL_COLOR_ATTACHMENT0);
            glBlitNamedFramebuffer(
                opaqueLit_.id(), resultTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_COLOR_BUFFER_BIT, GL_NEAREST
            );
            glBlitNamedFramebuffer(
                opaqueGBuffer_.id(), resultTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_DEPTH_BUFFER_BIT, GL_NEAREST
            );
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            forwardShader_.bind();
            glBindTextureUnit(0, layeredSupportTarget_.color(0));
            glBindTextureUnit(1, opaqueGBuffer_.depth());
            forwardShader_.set_int("uNearestOnly", 0);

            // Deliberately no transparency sorting: this is the conventional
            // fixed draw-order alpha-blending baseline. Its result depends on the
            // application's submission order whenever transparent fragments overlap.
            drawTransparentObjects(forwardShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });

            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);
        }

        void renderSortedForward(const glm::mat4& viewProjection) {
            glNamedFramebufferReadBuffer(opaqueLit_.id(), GL_COLOR_ATTACHMENT0);
            glNamedFramebufferDrawBuffer(resultTarget_.id(), GL_COLOR_ATTACHMENT0);
            glBlitNamedFramebuffer(
                opaqueLit_.id(), resultTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_COLOR_BUFFER_BIT, GL_NEAREST
            );
            glBlitNamedFramebuffer(
                opaqueGBuffer_.id(), resultTarget_.id(),
                0, 0, renderWidth_, renderHeight_,
                0, 0, renderWidth_, renderHeight_,
                GL_DEPTH_BUFFER_BIT, GL_NEAREST
            );
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            forwardShader_.bind();
            glBindTextureUnit(0, layeredSupportTarget_.color(0));
            glBindTextureUnit(1, opaqueGBuffer_.depth());
            forwardShader_.set_int("uNearestOnly", 0);
            const std::vector<int> order = transparentBackToFrontOrder();
            for (const int object : order) {
                drawTransparentObjects(forwardShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); }, object);
            }
            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);
        }

        void renderDepthPeeledReference(const glm::mat4& viewProjection) {
            const float clearColor[] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float clearDepth = 1.0f;

            glClearNamedFramebufferfv(depthPeelAccum_.id(), GL_COLOR, 0, clearColor);

            for (int layer = 0; layer < REFERENCE_PEEL_LAYERS; ++layer) {
                gfx::research::Framebuffer& current = (layer & 1) == 0 ? depthPeelA_ : depthPeelB_;
                gfx::research::Framebuffer& previous = (layer & 1) == 0 ? depthPeelB_ : depthPeelA_;

                glClearNamedFramebufferfv(current.id(), GL_COLOR, 0, clearColor);
                glClearNamedFramebufferfv(current.id(), GL_DEPTH, 0, &clearDepth);
                current.bind();
                glViewport(0, 0, renderWidth_, renderHeight_);
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
                glDisable(GL_BLEND);

                depthPeelShader_.bind();
                glBindTextureUnit(1, opaqueGBuffer_.depth());
                glBindTextureUnit(2, layer == 0 ? opaqueGBuffer_.depth() : previous.depth());
                depthPeelShader_.set_int("uOpaqueDepth", 1);
                depthPeelShader_.set_int("uPrevDepth", 2);
                depthPeelShader_.set_int("uFirstLayer", layer == 0 ? 1 : 0);
                drawTransparentObjects(depthPeelShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });

                depthPeelAccum_.bind();
                glViewport(0, 0, renderWidth_, renderHeight_);
                glDisable(GL_DEPTH_TEST);
                glDepthMask(GL_FALSE);
                glEnable(GL_BLEND);
                glBlendEquation(GL_FUNC_ADD);
                glBlendFuncSeparate(GL_ONE_MINUS_DST_ALPHA, GL_ONE, GL_ONE_MINUS_DST_ALPHA, GL_ONE);
                depthPeelAccumulateShader_.bind();
                glBindTextureUnit(0, current.color(0));
                depthPeelAccumulateShader_.set_int("uLayer", 0);
                fullscreen_.draw();
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }

            referenceTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);
            depthPeelResolveShader_.bind();
            glBindTextureUnit(0, depthPeelAccum_.color(0));
            glBindTextureUnit(1, opaqueLit_.color(0));
            depthPeelResolveShader_.set_int("uAccum", 0);
            depthPeelResolveShader_.set_int("uOpaque", 1);
            fullscreen_.draw();
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);
        }

        void renderWboit(const glm::mat4& viewProjection) {
            wboitTarget_.clear();
            wboitTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendEquationi(0, GL_FUNC_ADD);
            glBlendEquationi(1, GL_FUNC_ADD);
            glBlendFunci(0, GL_ONE, GL_ONE);
            glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
            wboitShader_.bind();
            glBindTextureUnit(0, opaqueGBuffer_.depth());
            wboitShader_.set_float("uDepthExponent", wboitDepthExponent_);
            drawTransparentObjects(wboitShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });
            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);

            const GLenum resultBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(resultTarget_.id(), 2, resultBuffers);
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            const float clearColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
            const float clearMeta[] = {0.0f, 0.0f, 0.0f, 1.0f};
            glClearBufferfv(GL_COLOR, 0, clearColor);
            glClearBufferfv(GL_COLOR, 1, clearMeta);
            wboitResolveShader_.bind();
            glBindTextureUnit(0, opaqueLit_.color(0));
            glBindTextureUnit(1, wboitTarget_.accumulation());
            glBindTextureUnit(2, wboitTarget_.revealage());
            fullscreen_.draw();
        }

        void renderLayeredWboit(const glm::mat4& viewProjection) {
            renderLayeredSupport(viewProjection);
            layeredWboitTarget_.clear();

            layeredWboitTarget_.bind_global_revealage();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
            globalRevealShader_.bind();
            glBindTextureUnit(0, opaqueGBuffer_.depth());
            drawTransparentObjects(globalRevealShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });
            glDisable(GL_BLEND);

            layeredWboitTarget_.bind_layers();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_BLEND);
            glBlendEquationi(0, GL_FUNC_ADD);
            glBlendEquationi(1, GL_FUNC_ADD);
            glBlendFunci(0, GL_ONE, GL_ONE);
            glBlendFunci(1, GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
            layeredWboitShader_.bind();
            glBindTextureUnit(0, opaqueGBuffer_.depth());
            glBindTextureUnit(1, layeredSupportTarget_.color(0));
            layeredWboitShader_.set_float("uDepthExponent", wboitDepthExponent_);
            layeredWboitShader_.set_float("uNearPlane", 0.03f);
            layeredWboitShader_.set_float("uFarPlane", 250.0f);
            layeredWboitShader_.set_float("uLayerDepthSpan", layeredDepthSpan_);
            layeredWboitShader_.set_int("uBinCount", 8);
            drawTransparentObjects(layeredWboitShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });
            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);

            const GLenum resultBuffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
            glNamedFramebufferDrawBuffers(resultTarget_.id(), 2, resultBuffers);
            resultTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            const float clearColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
            const float clearMeta[] = {0.0f, 0.0f, 0.0f, 1.0f};
            glClearBufferfv(GL_COLOR, 0, clearColor);
            glClearBufferfv(GL_COLOR, 1, clearMeta);
            layeredResolveShader_.bind();
            glBindTextureUnit(0, opaqueLit_.color(0));
            glBindTextureUnit(1, layeredWboitTarget_.accumulation());
            glBindTextureUnit(2, layeredWboitTarget_.revealage());
            glBindTextureUnit(3, layeredWboitTarget_.global_revealage());
            layeredResolveShader_.set_int("uBinCount", 8);
            fullscreen_.draw();
        }

        void renderAlphaReference(const glm::mat4& viewProjection) {
            alphaReferenceTarget_.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            const float clear[] = {0.0f, 0.0f, 0.0f, 0.0f};
            glClearBufferfv(GL_COLOR, 0, clear);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            alphaReferenceShader_.bind();
            glBindTextureUnit(0, opaqueGBuffer_.depth());
            alphaReferenceShader_.set_int("uClipOpaque", 1);
            const std::vector<int> order = transparentBackToFrontOrder();
            for (const int object : order) {
                drawTransparentObjects(alphaReferenceShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); }, object);
            }
            glDisable(GL_BLEND);
            glDepthMask(GL_TRUE);
        }

        bool ensureAttributeReferenceTargets() {
            if (attributeReferenceWidth_ == renderWidth_ && attributeReferenceHeight_ == renderHeight_) return true;
            const gfx::research::ColorAttachmentDesc referenceColors[] = {
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST},
                {GL_RGBA16F, GL_NEAREST, GL_NEAREST}
            };
            const bool ok = attributeReferenceFront_.create(renderWidth_, renderHeight_, referenceColors, true, GL_DEPTH_COMPONENT32F)
                && attributeReferenceBack_.create(renderWidth_, renderHeight_, referenceColors, true, GL_DEPTH_COMPONENT32F);
            if (ok) {
                attributeReferenceWidth_ = renderWidth_;
                attributeReferenceHeight_ = renderHeight_;
            }
            return ok;
        }

        void renderAttributeReferenceLayer(
            gfx::research::Framebuffer& output,
            const unsigned int previousDepth,
            const bool firstLayer,
            const glm::mat4& viewProjection
        ) {
            const GLenum drawBuffers[] = {
                GL_COLOR_ATTACHMENT0,
                GL_COLOR_ATTACHMENT1,
                GL_COLOR_ATTACHMENT2,
                GL_COLOR_ATTACHMENT3
            };
            glNamedFramebufferDrawBuffers(output.id(), 4, drawBuffers);
            const float clearAlbedo[] = {0.0f, 0.0f, 0.0f, 0.5f};
            const float clearNormal[] = {0.5f, 0.5f, 1.0f, 0.0f};
            const float clearPosition[] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float clearControl[] = {0.0f, 0.0f, 0.0f, 0.0f};
            const float clearDepth = 1.0f;
            glClearNamedFramebufferfv(output.id(), GL_COLOR, 0, clearAlbedo);
            glClearNamedFramebufferfv(output.id(), GL_COLOR, 1, clearNormal);
            glClearNamedFramebufferfv(output.id(), GL_COLOR, 2, clearPosition);
            glClearNamedFramebufferfv(output.id(), GL_COLOR, 3, clearControl);
            glClearNamedFramebufferfv(output.id(), GL_DEPTH, 0, &clearDepth);

            output.bind();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);

            attributeReferenceShader_.bind();
            glBindTextureUnit(1, opaqueGBuffer_.depth());
            glBindTextureUnit(2, previousDepth);
            attributeReferenceShader_.set_int("uOpaqueDepth", 1);
            attributeReferenceShader_.set_int("uPrevDepth", 2);
            attributeReferenceShader_.set_int("uFirstLayer", firstLayer ? 1 : 0);
            drawTransparentObjects(attributeReferenceShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });
            glDepthFunc(GL_LESS);
        }

        void renderAttributeReferences(const glm::mat4& viewProjection) {
            if (!ensureAttributeReferenceTargets()) return;
            renderAttributeReferenceLayer(attributeReferenceFront_, opaqueGBuffer_.depth(), true, viewProjection);
            renderAttributeReferenceLayer(attributeReferenceBack_, attributeReferenceFront_.depth(), false, viewProjection);
            glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
        }

        void renderOverdraw(const glm::mat4& viewProjection) {
            const unsigned int zero = 0u;
            glClearTexImage(overdrawInputTexture_, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
            glClearTexImage(overdrawSurvivorTexture_, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
            gfx::research::Framebuffer::bind_default();
            glViewport(0, 0, renderWidth_, renderHeight_);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);
            overdrawShader_.bind();
            glBindTextureUnit(0, opaqueGBuffer_.depth());
            glBindImageTexture(0, overdrawInputTexture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
            glBindImageTexture(1, overdrawSurvivorTexture_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);
            drawTransparentObjects(overdrawShader_, viewProjection, [](const gfx::research::Mesh& mesh) { mesh.draw(); });
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
            glDepthMask(GL_TRUE);
        }

        void renderProposed(const glm::mat4& viewProjection) {
            renderMixedDitheredGBuffer(viewProjection);

            // Lighting is deliberately AFTER the complete mixed G-buffer exists.
            // It shades exactly one selected scene sample per pixel, regardless of
            // whether that sample came from opaque or dithered-alpha geometry.
            const bool timeLighting = benchmarkRunning_ && currentBenchmarkCase().recordDeferredLighting;
            if (timeLighting) deferredLightingTimer_.begin();
            renderFullRateDeferredLighting(mixedGBuffer_, mixedLit_);
            if (timeLighting) deferredLightingTimer_.end();

            // Only after lighting do the reconstruction variants remove the visible
            // screen-door pattern and recover a continuous image.
            renderMixedReconstruction();
        }

        unsigned int selectedOutputTexture() const {
            return method_ == METHOD_DEPTH_PEEL_REFERENCE ? referenceTarget_.color(0) : resultTarget_.color(0);
        }

        void displaySelected(const int windowWidth, const int windowHeight) {
            gfx::research::Framebuffer::bind_default();
            glViewport(0, 0, windowWidth, windowHeight);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_BLEND);
            displayShader_.bind();
            glBindTextureUnit(0, selectedOutputTexture());
            displayShader_.set_int("uImage", 0);
            displayShader_.set_int("uToneMap", 1);
            fullscreen_.draw();
        }

        void resetCamera(const bool releaseMouse = true) {
            camera_.set_target(SPONZA_ORBIT_CENTER);
            camera_.set_distance(DEFAULT_CAMERA_DISTANCE);
            camera_.set_yaw(DEFAULT_CAMERA_YAW);
            camera_.set_pitch(DEFAULT_CAMERA_PITCH);
            camera_.set_fov(DEFAULT_CAMERA_FOV);
            camera_.set_clip(DEFAULT_CAMERA_NEAR, DEFAULT_CAMERA_FAR);
            if (releaseMouse && cameraMouseCaptured_) {
                if (glfwRawMouseMotionSupported()) glfwSetInputMode(window(), GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                glfwSetInputMode(window(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                cameraMouseCaptured_ = false;
            }
        }

        void updateCamera(const gfx::research::FrameInfo& frame) {
            if (benchmarkRunning_) return;

            ImGuiIO& io = ImGui::GetIO();
            const bool rightMouseDown = glfwGetMouseButton(window(), GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
            if (rightMouseDown && !io.WantCaptureMouse) {
                double x = 0.0;
                double y = 0.0;
                glfwGetCursorPos(window(), &x, &y);
                if (!cameraMouseCaptured_) {
                    cameraMouseCaptured_ = true;
                    cameraLastMouseX_ = x;
                    cameraLastMouseY_ = y;
                    glfwSetInputMode(window(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    if (glfwRawMouseMotionSupported()) glfwSetInputMode(window(), GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
                } else {
                    const float dx = static_cast<float>(x - cameraLastMouseX_);
                    const float dy = static_cast<float>(y - cameraLastMouseY_);
                    cameraLastMouseX_ = x;
                    cameraLastMouseY_ = y;
                    camera_.yaw() -= dx * CAMERA_MOUSE_SENSITIVITY;
                    camera_.pitch() = std::clamp(camera_.pitch() - dy * CAMERA_MOUSE_SENSITIVITY, -1.39626f, 1.39626f);
                }
            } else if (cameraMouseCaptured_) {
                if (glfwRawMouseMotionSupported()) glfwSetInputMode(window(), GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
                glfwSetInputMode(window(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                cameraMouseCaptured_ = false;
            }

            if (!io.WantCaptureMouse && std::abs(io.MouseWheel) > 1e-6f) {
                const float zoomFactor = std::exp(-io.MouseWheel * 0.12f);
                camera_.distance() = std::clamp(camera_.distance() * zoomFactor, CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE);
            }

            if (io.WantCaptureKeyboard) return;
            if (glfwGetKey(window(), GLFW_KEY_R) == GLFW_PRESS) {
                if (!cameraResetPressed_) resetCamera();
                cameraResetPressed_ = true;
            } else {
                cameraResetPressed_ = false;
            }

            const float dt = std::min(frame.delta_seconds, 0.05f);
            const float fast = glfwGetKey(window(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? CAMERA_FAST_MULTIPLIER : 1.0f;
            const float orbitStep = CAMERA_KEY_ORBIT_SPEED * fast * dt;
            const float zoomStep = CAMERA_KEY_ZOOM_SPEED * fast * dt;

            if (glfwGetKey(window(), GLFW_KEY_A) == GLFW_PRESS) camera_.yaw() += orbitStep;
            if (glfwGetKey(window(), GLFW_KEY_D) == GLFW_PRESS) camera_.yaw() -= orbitStep;
            if (glfwGetKey(window(), GLFW_KEY_Q) == GLFW_PRESS) camera_.pitch() = std::clamp(camera_.pitch() - orbitStep, -1.39626f, 1.39626f);
            if (glfwGetKey(window(), GLFW_KEY_E) == GLFW_PRESS) camera_.pitch() = std::clamp(camera_.pitch() + orbitStep, -1.39626f, 1.39626f);
            if (glfwGetKey(window(), GLFW_KEY_W) == GLFW_PRESS) camera_.distance() = std::max(CAMERA_MIN_DISTANCE, camera_.distance() - zoomStep);
            if (glfwGetKey(window(), GLFW_KEY_S) == GLFW_PRESS) camera_.distance() = std::min(CAMERA_MAX_DISTANCE, camera_.distance() + zoomStep);
        }

        void on_frame(const gfx::research::FrameInfo& frame) override {
            prepareBenchmarkFrame();
            updateCamera(frame);
            const int windowWidth = frame.width;
            const int windowHeight = frame.height;
            const float aspect = static_cast<float>(renderWidth_) / static_cast<float>(renderHeight_);
            const glm::mat4 viewProjection = camera_.view_projection(aspect);

            const bool unifiedSceneMethod =
                (method_ >= METHOD_OCCUPANCY_2X2 && method_ <= METHOD_FULL_DEFERRED_LIGHT_CONTROL)
                || method_ == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                || method_ == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                || method_ == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                || method_ == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
            const bool needReference =
                (benchmarkRunning_ && currentBenchmarkCase().quality) || method_ == METHOD_DEPTH_PEEL_REFERENCE;
            const bool needAlphaReference =
                benchmarkRunning_ && currentBenchmarkCase().quality;
            const bool needAttributeReference = benchmarkRunning_ && currentBenchmarkCase().quality
                && (method_ == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                    || method_ == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                    || method_ == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                    || method_ == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
            const bool needOverdrawDepth =
                benchmarkRunning_ && currentBenchmarkCase().recordOverdraw && !overdrawMeasuredThisCase_;

            // Normal interactive proposed rendering has no separate opaque G-buffer/light
            // pass at all. The legacy opaque path is evaluated only when a baseline or
            // benchmark reference explicitly requires it.
            if (!unifiedSceneMethod || needReference || needAlphaReference || needAttributeReference || needOverdrawDepth) {
                renderOpaque(viewProjection);
            }

            if (needReference && method_ != METHOD_DEPTH_PEEL_REFERENCE) renderDepthPeeledReference(viewProjection);
            if (needAlphaReference) renderAlphaReference(viewProjection);
            if (needAttributeReference) renderAttributeReferences(viewProjection);
            if (benchmarkRunning_ && currentBenchmarkCase().recordOverdraw && !overdrawMeasuredThisCase_) {
                renderOverdraw(viewProjection);
                const std::vector<unsigned int> input = readTextureR32U(overdrawInputTexture_, renderWidth_, renderHeight_);
                const std::vector<unsigned int> survivors = readTextureR32U(overdrawSurvivorTexture_, renderWidth_, renderHeight_);
                currentOverdraw_ = calculateOverdrawMetrics(input, survivors);
                overdrawMeasuredThisCase_ = true;
            }

            gpuTimer_.begin();
            if (method_ == METHOD_DEPTH_PEEL_REFERENCE) renderDepthPeeledReference(viewProjection);
            else if (method_ == METHOD_CLASSIC_FORWARD_ALPHA) renderClassicForward(viewProjection);
            else if (method_ == METHOD_SORTED_FORWARD_ALPHA) renderSortedForward(viewProjection);
            else if (method_ == METHOD_RAW_DITHER) renderRawDither(viewProjection);
            else if (method_ == METHOD_WBOIT) renderWboit(viewProjection);
            else if (method_ == METHOD_LAYERED_WBOIT) renderLayeredWboit(viewProjection);
            else if (method_ == METHOD_FULL_DEFERRED_LIGHT_CONTROL) renderFullDeferredLightControl(viewProjection);
            else if (method_ == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION) renderDeferredAttributeReconstruction(viewProjection);
            else if (method_ == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION) renderDeferredTwoLayerReconstruction(viewProjection, false, false);
            else if (method_ == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC) renderDeferredTwoLayerReconstruction(viewProjection, false, true);
            else if (method_ == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE) renderDeferredTwoLayerReconstruction(viewProjection, true, true);
            else renderProposed(viewProjection);
            gpuTimer_.end();

            displaySelected(windowWidth, windowHeight);

            if (benchmarkRunning_) updateBenchmarkFrame();
        }

        void on_gui(const gfx::research::FrameInfo& frame) override {
            ImGui::Begin("Deferred transparency baseline study");
            ImGui::Text("%.1f FPS | CPU %.3f ms | transparency GPU %.3f ms", fps(), cpu_frame_milliseconds(), gpuTimer_.milliseconds());
            ImGui::Separator();
            ImGui::Combo("Method", &method_, METHOD_LABELS.data(), static_cast<int>(METHOD_LABELS.size()));
            if (method_ == METHOD_CLASSIC_FORWARD_ALPHA) ImGui::TextDisabled("Classic fixed draw order: standard alpha blending with no transparency sorting.");
            if (method_ == METHOD_SORTED_FORWARD_ALPHA) ImGui::TextDisabled("Object sorting is view-dependent and is NOT ground truth for intersections.");
            if (method_ == METHOD_DEPTH_PEEL_REFERENCE) ImGui::TextDisabled("Per-pixel depth peeling reference, up to 24 transparent layers.");
            if (method_ == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION) ImGui::TextDisabled("Single-layer attribute-space decoder: reconstruct material attributes first, then light and composite.");
            if (method_ == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION) ImGui::TextDisabled("Legacy K=2: isotropic reconstruction plus the v3.1 post-lighting neighborhood resolve.");
            if (method_ == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC) ImGui::TextDisabled("Ablation: previous isotropic attribute decoder, but direct composition after deferred lighting (no post-lit blur).");
            if (method_ == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE) ImGui::TextDisabled("Proposed v3.2: censored K=2 + anisotropic edge support + affine position/material reproduction + direct final composition.");
            ImGui::Combo("Dither", &ditherMode_, DITHER_LABELS.data(), static_cast<int>(DITHER_LABELS.size()));
            ImGui::Combo("Coverage encoding", &coverageEncodingMode_, COVERAGE_ENCODING_LABELS.data(), static_cast<int>(COVERAGE_ENCODING_LABELS.size()));
            if (coverageEncodingMode_ >= 1 && coverageEncodingMode_ <= 3) ImGui::SliderFloat("Coverage bias", &coverageBias_, 0.0f, 2.0f, "%.2f");
            if (coverageEncodingMode_ == 4) ImGui::SliderFloat("Logit compression", &coverageBias_, 0.0f, 1.5f, "%.2f");
            if (coverageEncodingMode_ == 5) ImGui::SliderFloat("Power exponent p", &coverageBias_, 1.0f, 3.0f, "%.2f");
            if (coverageEncodingMode_ == 3) ImGui::TextDisabled("+/- logit bias gathers both foreground and exposed-background samples; decoder returns physical alpha.");
            if (coverageEncodingMode_ == 4) ImGui::TextDisabled("Experimental measurement preconditioning; Identity is the baseline-study default.");
            if (coverageEncodingMode_ == 5) ImGui::TextDisabled("High-alpha-only reveal: identity for alpha <= 0.5; above 0.5 a pivoted power transform exposes deeper samples and is inverted before composition.");
            ImGui::Combo("Alpha field", &alphaMode_, ALPHA_MODE_LABELS.data(), static_cast<int>(ALPHA_MODE_LABELS.size()));
            ImGui::Combo("Transparent geometry", &transparentGeometry_, TRANSPARENT_GEOMETRY_LABELS.data(), static_cast<int>(TRANSPARENT_GEOMETRY_LABELS.size()));
            ImGui::SliderFloat("Alpha", &alpha_, 0.0f, 1.0f, "%.3f");
            ImGui::SliderInt("Kernel radius", &kernelRadius_, 1, 3);
            ImGui::SliderFloat("Depth sigma", &depthSigma_, 0.03f, 2.0f, "%.3f");
            ImGui::SliderFloat("Albedo edge sigma", &albedoSigma_, 0.02f, 0.50f, "%.3f");
            ImGui::SliderFloat("Detail restoration", &detailStrength_, 0.0f, 0.8f, "%.2f");
            ImGui::SliderFloat("WBOIT depth exponent", &wboitDepthExponent_, 1.0f, 8.0f, "%.2f");
            ImGui::SliderFloat("Layered OIT depth span", &layeredDepthSpan_, 1.0f, 20.0f, "%.2f");
            if (transparentGeometry_ == 6) ImGui::SliderInt("Transparent layers", &transparentLayers_, 1, MAX_TRANSPARENT_LAYERS);
            ImGui::SliderInt("Point lights", &lightCount_, 1, 256);
            ImGui::Checkbox("Sponza", &showSponza_);
            ImGui::Separator();
            const glm::vec3 cameraPosition = camera_.position();
            ImGui::Text("Camera: %.2f, %.2f, %.2f", cameraPosition.x, cameraPosition.y, cameraPosition.z);
            ImGui::SliderFloat("Camera distance", &camera_.distance(), CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE, "%.2f");
            ImGui::SliderAngle("Camera yaw", &camera_.yaw(), -180.0f, 180.0f);
            ImGui::SliderAngle("Camera pitch", &camera_.pitch(), -80.0f, 80.0f);
            ImGui::Text("Orbit center: %.2f, %.2f, %.2f", SPONZA_ORBIT_CENTER.x, SPONZA_ORBIT_CENTER.y, SPONZA_ORBIT_CENTER.z);
            ImGui::TextWrapped("Camera controls: RMB + mouse orbits around Sponza; mouse wheel or W/S zooms; A/D orbits horizontally; Q/E orbits vertically; Shift is faster; R resets the view.");
            if (ImGui::Button("Reset camera")) resetCamera();
            if (sponzaLoaded_) ImGui::Text("Sponza: %zu submeshes, %zu diffuse textures", sponza_.submesh_count(), sponza_.texture_count());
            ImGui::Separator();
            ImGui::TextWrapped("Publication-candidate v3.2: depth peeling is the quality reference. The proposed K=2 path models deeper-layer observations as censored, reconstructs transparent G-buffer attributes with anisotropic edge-aware support and local affine reproduction, then performs deferred lighting only after reconstruction. The previous isotropic K=2 path is retained as the direct ablation; WBOIT/layered WBOIT and forward alpha remain baselines.");
            ImGui::TextWrapped("Benchmark output now also includes overdraw/depth-complexity statistics: input transparent fragments per pixel and coverage-surviving sparse samples per pixel.");
            ImGui::Separator();
            if (!benchmarkRunning_) {
                if (ImGui::Button("Run known-baselines suite")) startBenchmarkSuite(false, false, true);
                if (ImGui::Button("Run v3.2 edge GO/NO-GO")) startBenchmarkSuite(false, false, false, true);
                if (ImGui::Button("Run quick benchmark suite")) startBenchmarkSuite(false);
                ImGui::SameLine();
                if (ImGui::Button("Run publication decision suite")) startBenchmarkSuite(true, true);
                if (ImGui::Button("Run full benchmark suite")) startBenchmarkSuite(true);
            } else {
                ImGui::Text("Benchmark case %zu / %zu", benchmarkCaseIndex_ + 1, benchmarkCases_.size());
                ImGui::TextWrapped("%s", currentBenchmarkCase().pairId.c_str());
                if (ImGui::Button("Stop suite")) stopBenchmarkSuite(false);
            }
            ImGui::Text("Window: %d x %d", frame.width, frame.height);
            ImGui::End();
        }

        void startBenchmarkSuite(
            const bool full,
            const bool paperOnly = false,
            const bool baselineOnly = false,
            const bool edgeDecisionOnly = false
        ) {
            resetCamera();
            benchmarkCases_ = edgeDecisionOnly
                ? makeEdgeDecisionBenchmarkCases()
                : (baselineOnly ? makeBaselineBenchmarkCases() : (paperOnly ? makePaperBenchmarkCases() : makeBenchmarkCases(full)));
            if (benchmarkCases_.empty()) return;
            benchmarkOutput_ = launchOptions_.benchmarkOutput;
            std::filesystem::create_directories(benchmarkOutput_ / "captures");
            benchmarkRaw_.open(benchmarkOutput_ / "timing_raw.csv", std::ios::trunc);
            benchmarkSummary_.open(benchmarkOutput_ / "timing_summary.csv", std::ios::trunc);
            benchmarkQuality_.open(benchmarkOutput_ / "quality.csv", std::ios::trunc);
            benchmarkAttributeQuality_.open(benchmarkOutput_ / "attribute_quality.csv", std::ios::trunc);
            benchmarkOverdraw_.open(benchmarkOutput_ / "overdraw.csv", std::ios::trunc);
            benchmarkManifest_.open(benchmarkOutput_ / "manifest.txt", std::ios::trunc);
            if (!benchmarkRaw_ || !benchmarkSummary_ || !benchmarkQuality_ || !benchmarkAttributeQuality_ || !benchmarkOverdraw_ || !benchmarkManifest_) {
                std::cerr << "Could not create benchmark output in " << benchmarkOutput_ << '\n';
                stopBenchmarkSuite(false);
                return;
            }

            benchmarkRaw_ << "pair_id,sweep,repeat,method,rodd_variant,width,height,lights,alpha_mode,dither,coverage_encoding,coverage_bias,kernel_radius,geometry,layers,alpha,depth_sigma,albedo_sigma,wboit_depth_exponent,layer_span,sample,gpu_ms,deferred_lighting_ms\n";
            benchmarkSummary_ << "pair_id,sweep,repeat,method,rodd_variant,width,height,lights,alpha_mode,dither,coverage_encoding,coverage_bias,kernel_radius,geometry,layers,alpha,depth_sigma,albedo_sigma,wboit_depth_exponent,layer_span,mean_ms,median_ms,stddev_ms,p95_ms,min_ms,max_ms,deferred_lighting_mean_ms,deferred_lighting_median_ms,deferred_lighting_stddev_ms,deferred_lighting_p95_ms,deferred_lighting_min_ms,deferred_lighting_max_ms\n";
            benchmarkQuality_ << "pair_id,sweep,method,rodd_variant,width,height,lights,alpha_mode,dither,coverage_encoding,coverage_bias,kernel_radius,geometry,layers,alpha,depth_sigma,albedo_sigma,wboit_depth_exponent,layer_span,color_rmse,color_relative_rmse,color_psnr,color_max_abs,global_ssim,gradient_rmse,gradient_relative_rmse,gradient_transfer,ordered_phase_residual_ratio,coverage_rmse,coverage_mae,coverage_max_abs,alpha_rmse,alpha_mae,alpha_max_abs,alpha_samples\n";
            benchmarkAttributeQuality_ << "pair_id,sweep,method,layer,width,height,lights,alpha,dither,coverage_encoding,kernel_radius,geometry,reference_samples,predicted_samples,matched_samples,match_recall,false_positive_ratio,wrong_surface_samples,albedo_rmse,normal_mean_deg,normal_median_deg,normal_p95_deg,roughness_rmse,roughness_mae,position_rmse,position_max,alpha_rmse,alpha_mae,alpha_max\n";
            benchmarkOverdraw_ << "pair_id,sweep,method,rodd_variant,width,height,lights,alpha_mode,dither,coverage_encoding,coverage_bias,kernel_radius,geometry,layers,alpha,input_mean_fragments,input_max_fragments,coverage_surviving_mean_fragments,coverage_surviving_max_fragments,coverage_survival_ratio,covered_pixel_ratio\n";
            const glm::vec3 cameraPosition = camera_.position();
            benchmarkManifest_
                << "study=censored-sparse-layer-edge-aware-v3.2\n"
                << "gpu_vendor=" << glString(GL_VENDOR) << '\n'
                << "gpu_renderer=" << glString(GL_RENDERER) << '\n'
                << "opengl_version=" << glString(GL_VERSION) << '\n'
                << "glsl_version=" << glString(GL_SHADING_LANGUAGE_VERSION) << '\n'
                << "camera_position=" << cameraPosition.x << ',' << cameraPosition.y << ',' << cameraPosition.z << '\n'
                << "camera_target=" << SPONZA_ORBIT_CENTER.x << ',' << SPONZA_ORBIT_CENTER.y << ',' << SPONZA_ORBIT_CENTER.z << '\n'
                << "camera_distance=" << camera_.distance() << '\n'
                << "camera_yaw_radians=" << camera_.yaw() << '\n'
                << "camera_pitch_radians=" << camera_.pitch() << '\n'
                << "camera_fov_degrees=" << camera_.fov() << '\n'
                << "camera_near=" << DEFAULT_CAMERA_NEAR << '\n'
                << "camera_far=" << DEFAULT_CAMERA_FAR << '\n'
                << "source_alpha_stored_in_proposed_gbuffer=0;benchmark-alpha-reference-is-a-separate-validation-pass\n"
                << "proposed_hidden_support_pass=0\n"
                << "legacy_opaque_gbuffer=3xRGBA16F+DEPTH32F(reference-and-baselines-only)\n"
                << "proposed_scene_gbuffer=one-ordinary-2D-4xRGBA16F+DEPTH32F(albedo_roughness,normal_valid,world_position_valid,selected_sample_control)\n"
                << "proposed_scene_gbuffer_scope=opaque-and-dithered-alpha-geometry-write-the-same-MRT-and-depth-before-any-lighting\n"
                << "proposed_geometry_framebuffer_switches_between_opaque_and_transparent=0\n"
                << "proposed_layer_count_buffer=0\n"
                << "selected_sample_control=RGBA16F(depth,0,0,surface-id);decoder-requires-only-depth-surface-id-pixel-phase-pattern-and-known-coverage-transform\n"
                << "layered_wboit_front_support=R32F+RGBA8+DEPTH32F(baseline-only;not-used-by-proposed)\n"
                << "dither_modes=bayer4,stable-hash,rodd4-candidate,bayer8,bayer16;ordered-pattern-phase-is-spatial-translation;proposed-thresholds-are-reproducible-from-pixel-and-surface-id\n"
                << "coverage_encoding_modes=identity,foreground-biased,background-biased,complementary-codesign,balanced-logit-companding;all-nonidentity-transforms-are-invertible-before-physical-alpha-reconstruction\n"
                << "kernel_radii=1,2,3\n"
                << "quality_reference=front-to-back-depth-peeling;max_reference_layers=" << REFERENCE_PEEL_LAYERS << ";classic-forward-and-object-sorted-forward-are-baselines-not-ground-truth\n"
                << "methods=depth-peeling-reference-24,classic-forward-fixed-order,object-sorted-forward,wboit,layered-wboit-8,raw-dither,occupancy-2x2,box-reconstruction,gaussian-reconstruction,joint-bilateral-gbuffer,two-component-affine,rodd-experimental,mixed-deferred-raw-control,deferred-attribute-reconstruction,deferred-sparse-layer-2-legacy,deferred-sparse-layer-2-isotropic-direct,deferred-sparse-layer-2-edge-aware\n"
                << "proposed_pipeline=edge-aware-k2;opaque-reference-plus-dithered-transparent-mixed-gbuffer->censored-two-layer-attribute-reconstruction->deferred-lighting-of-reconstructed-layers->confidence-guided-composition\n"
                << "reconstruction_proposed=censored-known-threshold-alpha-inference+surface-id-and-depth-gating+anisotropic-edge-support+local-affine-position-and-material-reproduction+normal-renormalization+geometry-consistency-rejection\n"
                << "rodd_ablation_bits=1:no-ordinal,2:no-censored-negatives,4:no-geometry-gating,8:no-rank-conditioning,16:no-moment-reproduction,32:no-directional-pairs,64:no-confidence-guard\n"
                << "quality_metrics=linear-hdr-rmse,relative-rmse,psnr,global-luma-ssim,gradient-rmse,gradient-transfer,ordered-phase-residual-ratio,coverage-error,continuous-alpha-error;attribute_quality_csv=albedo-rmse,normal-angular-mean-median-p95,roughness-rmse-mae,position-rmse-max,alpha-rmse-mae-max,surface-match-recall,false-positive-ratio\n"
                << "continuous_alpha_metric_scope=single-nonoverlapping-transparent-geometry(panel,sphere,leaf-card);multi-object-and-layer-stack-alpha-metrics=nan\n"
                << "coverage_estimator=physical-alpha-hat-with-measurement-normalized-geometric-support-applied-only-near-silhouettes;alpha_hat=phase-anchored-bayer8-interval-when-consistent-otherwise-local-ordinal-active-set-fit-without-source-alpha-values\n"
                << "detail_strength_default=0.0\n"
                << "control_pipeline=same-shared-dithered-gbuffer->same-deferred-lighting->no-reconstruction\n"
                << "sparse_lighting_backend=none;visibility-sparsity-is-already-encoded-in-the-single-gbuffer-sample\n"
                << "deferred_lighting_timer=one-full-frame-deferred-lighting-pass-over-the-final-mixed-gbuffer\n"
                << "albedo_edge_sigma_default=0.12\n"
                << "wboit_default_depth_exponent=2\n"
                << "layered_wboit_bins=8\n"
                << "layered_wboit_default_depth_span=12\n"
                << "scene=textured-sponza-interior-with-inserted-transparent-test-geometry\n"
                << "overdraw_metrics=input_fragments_per_pixel,coverage_surviving_fragments_per_pixel\n"
                << "temporal_sequences=subpixel-camera-yaw,subpixel-transparent-object-x;temporal-analysis=reference-matched-residual-delta-and-variance;history_ghosting_not_applicable=1\n"
                << "sponza_target_size=" << SPONZA_TARGET_SIZE << '\n'
                << "sponza_submeshes=" << sponza_.submesh_count() << '\n'
                << "sponza_textures=" << sponza_.texture_count() << '\n'
                << "curtain_texture_loaded=" << curtainTextureLoaded_ << '\n'
                << "foliage_texture_loaded=" << foliageTextureLoaded_ << '\n'
                << "full_suite=" << (full && !paperOnly ? 1 : 0) << '\n'
                << "paper_suite=" << (paperOnly ? 1 : 0) << '\n'
                << "baseline_only_suite=" << (baselineOnly ? 1 : 0) << '\n'
                << "sponza_loaded=" << sponzaLoaded_ << '\n'
                << "dragon_loaded=" << dragonLoaded_ << '\n'
                << "suzanne_loaded=" << suzanneLoaded_ << '\n'
                << "teapot_loaded=" << teapotLoaded_ << '\n';

            benchmarkCaseIndex_ = 0;
            benchmarkRunning_ = true;
            benchmarkCaseApplied_ = false;
            benchmarkSamples_.clear();
            benchmarkDeferredLightingSamples_.clear();
            std::cout << "Benchmark suite: " << benchmarkCases_.size() << " cases -> " << benchmarkOutput_ << '\n';
        }

        void stopBenchmarkSuite(const bool completed) {
            if (benchmarkRaw_.is_open()) benchmarkRaw_.close();
            if (benchmarkSummary_.is_open()) benchmarkSummary_.close();
            if (benchmarkQuality_.is_open()) benchmarkQuality_.close();
            if (benchmarkAttributeQuality_.is_open()) benchmarkAttributeQuality_.close();
            if (benchmarkOverdraw_.is_open()) benchmarkOverdraw_.close();
            if (benchmarkManifest_.is_open()) {
                benchmarkManifest_ << "completed=" << (completed ? 1 : 0) << '\n';
                benchmarkManifest_.close();
            }
            benchmarkRunning_ = false;
            benchmarkCaseApplied_ = false;
            transparentMotionOffset_ = glm::vec3(0.0f);
            resetCamera(false);
            if (completed && (launchOptions_.benchmarkSuite || launchOptions_.benchmarkQuick || launchOptions_.benchmarkPaper
                || launchOptions_.benchmarkEdgeDecision)) {
                glfwSetWindowShouldClose(window(), GLFW_TRUE);
            }
            benchmarkSamples_.clear();
            benchmarkDeferredLightingSamples_.clear();
            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(window(), &width, &height);
            if (width > 0 && height > 0) {
                createTargets(width, height);
                renderWidth_ = width;
                renderHeight_ = height;
            }
        }

        const BenchmarkCase& currentBenchmarkCase() const {
            static const BenchmarkCase fallback{};
            if (!benchmarkRunning_ || benchmarkCaseIndex_ >= benchmarkCases_.size()) return fallback;
            return benchmarkCases_[benchmarkCaseIndex_];
        }

        void prepareBenchmarkFrame() {
            if (!benchmarkRunning_ || benchmarkCaseIndex_ >= benchmarkCases_.size()) return;
            if (benchmarkCaseApplied_) return;
            const BenchmarkCase& test = currentBenchmarkCase();
            resetCamera(false);
            camera_.yaw() += test.cameraYawOffset;
            transparentMotionOffset_ = glm::vec3(test.objectOffsetX, 0.0f, 0.0f);
            method_ = test.method;
            lightCount_ = test.lightCount;
            alphaMode_ = test.alphaMode;
            ditherMode_ = test.ditherMode;
            coverageEncodingMode_ = test.coverageEncodingMode;
            coverageBias_ = test.coverageBias;
            kernelRadius_ = test.kernelRadius;
            roddAblationMask_ = test.roddAblationMask;
            transparentGeometry_ = test.transparentGeometry;
            alpha_ = test.alpha;
            depthSigma_ = test.depthSigma;
            albedoSigma_ = test.albedoSigma;
            transparentLayers_ = test.transparentLayers;
            wboitDepthExponent_ = test.wboitDepthExponent;
            layeredDepthSpan_ = test.layeredDepthSpan;
            if (renderWidth_ != test.targetWidth || renderHeight_ != test.targetHeight) {
                renderWidth_ = test.targetWidth;
                renderHeight_ = test.targetHeight;
                createTargets(renderWidth_, renderHeight_);
            }
            benchmarkWarmupRemaining_ = test.warmupFrames;
            benchmarkSampleIndex_ = 0;
            benchmarkSamples_.clear();
            benchmarkDeferredLightingSamples_.clear();
            overdrawMeasuredThisCase_ = false;
            currentOverdraw_ = {};
            benchmarkCaseApplied_ = true;
        }

        void updateBenchmarkFrame() {
            if (!benchmarkRunning_ || !benchmarkCaseApplied_) return;
            if (benchmarkWarmupRemaining_ > 0) {
                --benchmarkWarmupRemaining_;
                return;
            }
            if (!gpuTimer_.has_result()) return;

            const BenchmarkCase test = currentBenchmarkCase();
            const double milliseconds = gpuTimer_.milliseconds();
            const double deferredLightingMilliseconds = test.recordDeferredLighting && deferredLightingTimer_.has_result()
                ? deferredLightingTimer_.milliseconds()
                : std::numeric_limits<double>::quiet_NaN();
            benchmarkSamples_.push_back(milliseconds);
            if (std::isfinite(deferredLightingMilliseconds)) benchmarkDeferredLightingSamples_.push_back(deferredLightingMilliseconds);
            benchmarkRaw_
                << test.pairId << ',' << test.sweep << ',' << test.repeat << ',' << METHOD_IDS[test.method] << ',' << roddVariantId(test.roddAblationMask) << ','
                << test.targetWidth << ',' << test.targetHeight << ',' << test.lightCount << ','
                << ALPHA_MODE_IDS[test.alphaMode] << ',' << DITHER_IDS[test.ditherMode] << ',' << COVERAGE_ENCODING_IDS[test.coverageEncodingMode] << ',' << test.coverageBias << ','
                << test.kernelRadius << ',' << TRANSPARENT_GEOMETRY_IDS[test.transparentGeometry] << ',' << test.transparentLayers << ','
                << std::setprecision(8) << test.alpha << ',' << test.depthSigma << ',' << test.albedoSigma << ',' << test.wboitDepthExponent << ',' << test.layeredDepthSpan << ','
                << benchmarkSampleIndex_ << ',' << std::setprecision(12) << milliseconds << ',' << deferredLightingMilliseconds << '\n';
            ++benchmarkSampleIndex_;
            if (benchmarkSampleIndex_ < test.sampleFrames) return;

            const TimingStats stats = calculateStats(benchmarkSamples_);
            TimingStats lightingStats;
            if (!benchmarkDeferredLightingSamples_.empty()) lightingStats = calculateStats(benchmarkDeferredLightingSamples_);
            else {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                lightingStats.mean = lightingStats.median = lightingStats.stddev = lightingStats.p95 = lightingStats.minimum = lightingStats.maximum = nan;
            }
            benchmarkSummary_
                << test.pairId << ',' << test.sweep << ',' << test.repeat << ',' << METHOD_IDS[test.method] << ',' << roddVariantId(test.roddAblationMask) << ','
                << test.targetWidth << ',' << test.targetHeight << ',' << test.lightCount << ','
                << ALPHA_MODE_IDS[test.alphaMode] << ',' << DITHER_IDS[test.ditherMode] << ',' << COVERAGE_ENCODING_IDS[test.coverageEncodingMode] << ',' << test.coverageBias << ','
                << test.kernelRadius << ',' << TRANSPARENT_GEOMETRY_IDS[test.transparentGeometry] << ',' << test.transparentLayers << ','
                << std::setprecision(8) << test.alpha << ',' << test.depthSigma << ',' << test.albedoSigma << ',' << test.wboitDepthExponent << ',' << test.layeredDepthSpan << ','
                << std::setprecision(12) << stats.mean << ',' << stats.median << ',' << stats.stddev << ','
                << stats.p95 << ',' << stats.minimum << ',' << stats.maximum << ','
                << lightingStats.mean << ',' << lightingStats.median << ',' << lightingStats.stddev << ','
                << lightingStats.p95 << ',' << lightingStats.minimum << ',' << lightingStats.maximum << '\n';

            if (test.quality) {
                writeQuality(test);
                writeAttributeQuality(test);
            }
            if (test.recordOverdraw && overdrawMeasuredThisCase_) writeOverdraw(test, currentOverdraw_);
            if (test.capture) writeCapture(test);

            ++benchmarkCaseIndex_;
            benchmarkCaseApplied_ = false;
            overdrawMeasuredThisCase_ = false;
            currentOverdraw_ = {};
            if (benchmarkCaseIndex_ >= benchmarkCases_.size()) {
                std::cout << "Benchmark suite completed.\n";
                stopBenchmarkSuite(true);
            }
        }

        void writeQuality(const BenchmarkCase& test) {
            ImageMetrics imageMetrics;
            AlphaMetrics coverageMetrics;
            AlphaMetrics alphaMetrics;
            const bool continuousAlphaReferenceValid = test.transparentLayers == 1 && test.transparentGeometry >= 0 && test.transparentGeometry <= 2;
            if (test.method == METHOD_DEPTH_PEEL_REFERENCE) {
                imageMetrics.rmse = 0.0;
                imageMetrics.relativeRmse = 0.0;
                imageMetrics.psnr = std::numeric_limits<double>::infinity();
                imageMetrics.maximumAbsolute = 0.0;
                imageMetrics.globalSsim = 1.0;
                imageMetrics.gradientRmse = 0.0;
                imageMetrics.gradientRelativeRmse = 0.0;
                imageMetrics.gradientTransfer = 1.0;
                imageMetrics.orderedPhaseResidualRatio = 0.0;
                coverageMetrics.rmse = coverageMetrics.mae = coverageMetrics.maximumAbsolute = 0.0;
                coverageMetrics.samples = static_cast<std::size_t>(renderWidth_) * static_cast<std::size_t>(renderHeight_);
                if (continuousAlphaReferenceValid) {
                    alphaMetrics.rmse = alphaMetrics.mae = alphaMetrics.maximumAbsolute = 0.0;
                    alphaMetrics.samples = coverageMetrics.samples;
                } else {
                    alphaMetrics.rmse = alphaMetrics.mae = alphaMetrics.maximumAbsolute = std::numeric_limits<double>::quiet_NaN();
                    alphaMetrics.samples = 0;
                }
            } else {
                const std::vector<float> image = readTextureRgba(resultTarget_.color(0), renderWidth_, renderHeight_);
                const std::vector<float> reference = readTextureRgba(referenceTarget_.color(0), renderWidth_, renderHeight_);
                const std::vector<float> alphaMeta = readTextureRgba(resultTarget_.color(1), renderWidth_, renderHeight_);
                const std::vector<float> alphaReference = readTextureRgba(alphaReferenceTarget_.color(0), renderWidth_, renderHeight_);
                imageMetrics = calculateImageMetrics(image, reference);
                calculateGradientMetrics(image, reference, renderWidth_, renderHeight_, imageMetrics);
                imageMetrics.globalSsim = calculateGlobalLuminanceSsim(image, reference);
                imageMetrics.orderedPhaseResidualRatio = calculateOrderedPhaseResidualRatio(image, reference, renderWidth_, renderHeight_, test.ditherMode);

                if (test.method == METHOD_CLASSIC_FORWARD_ALPHA || test.method == METHOD_SORTED_FORWARD_ALPHA) {
                    // These forward baselines only write the blended color target. They do
                    // not produce the deferred alpha/coverage metadata attachment, so alpha
                    // metrics would be fabricated if we read it. Compare their RGB/HDR
                    // result against depth peeling and report alpha metrics as unavailable.
                    coverageMetrics.rmse = coverageMetrics.mae = coverageMetrics.maximumAbsolute = std::numeric_limits<double>::quiet_NaN();
                    coverageMetrics.samples = 0;
                    alphaMetrics.rmse = alphaMetrics.mae = alphaMetrics.maximumAbsolute = std::numeric_limits<double>::quiet_NaN();
                    alphaMetrics.samples = 0;
                } else {
                    coverageMetrics = calculateAlphaMetrics(alphaMeta, alphaReference);
                    std::vector<float> continuousAlphaMeta = alphaMeta;
                    if (test.method == METHOD_CONTROL_GUIDED_ANISOTROPIC || test.method == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                        || test.method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                        || test.method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                        || test.method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE) {
                        for (std::size_t i = 0; i + 3 < continuousAlphaMeta.size(); i += 4) continuousAlphaMeta[i] = continuousAlphaMeta[i + 3];
                    }
                    if (continuousAlphaReferenceValid) {
                        alphaMetrics = calculateAlphaMetrics(continuousAlphaMeta, alphaReference);
                    } else {
                        alphaMetrics.rmse = alphaMetrics.mae = alphaMetrics.maximumAbsolute = std::numeric_limits<double>::quiet_NaN();
                        alphaMetrics.samples = 0;
                    }
                }
            }
            benchmarkQuality_
                << test.pairId << ',' << test.sweep << ',' << METHOD_IDS[test.method] << ',' << roddVariantId(test.roddAblationMask) << ',' << test.targetWidth << ',' << test.targetHeight << ','
                << test.lightCount << ',' << ALPHA_MODE_IDS[test.alphaMode] << ',' << DITHER_IDS[test.ditherMode] << ',' << COVERAGE_ENCODING_IDS[test.coverageEncodingMode] << ',' << test.coverageBias << ','
                << test.kernelRadius << ',' << TRANSPARENT_GEOMETRY_IDS[test.transparentGeometry] << ',' << test.transparentLayers << ','
                << std::setprecision(8) << test.alpha << ',' << test.depthSigma << ',' << test.albedoSigma << ',' << test.wboitDepthExponent << ',' << test.layeredDepthSpan << ','
                << std::setprecision(12) << imageMetrics.rmse << ',' << imageMetrics.relativeRmse << ',' << imageMetrics.psnr << ','
                << imageMetrics.maximumAbsolute << ',' << imageMetrics.globalSsim << ',' << imageMetrics.gradientRmse << ',' << imageMetrics.gradientRelativeRmse << ','
                << imageMetrics.gradientTransfer << ',' << imageMetrics.orderedPhaseResidualRatio << ','
                << coverageMetrics.rmse << ',' << coverageMetrics.mae << ',' << coverageMetrics.maximumAbsolute << ','
                << alphaMetrics.rmse << ',' << alphaMetrics.mae << ',' << alphaMetrics.maximumAbsolute << ',' << alphaMetrics.samples << '\n';
        }

        void writeAttributeQuality(const BenchmarkCase& test) {
            if (attributeReferenceWidth_ != renderWidth_ || attributeReferenceHeight_ != renderHeight_) return;
            const bool oneLayer = test.method == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION;
            const bool twoLayer = test.method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                || test.method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                || test.method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
            if (!oneLayer && !twoLayer) return;

            const auto writeLayer = [&](const int layer, gfx::research::Framebuffer& reconstructed, gfx::research::Framebuffer& reference) {
                const AttributeMetrics metrics = calculateAttributeMetrics(
                    readTextureRgba(reconstructed.color(0), renderWidth_, renderHeight_),
                    readTextureRgba(reconstructed.color(1), renderWidth_, renderHeight_),
                    readTextureRgba(reconstructed.color(2), renderWidth_, renderHeight_),
                    readTextureRgba(reconstructed.color(3), renderWidth_, renderHeight_),
                    readTextureRgba(reference.color(0), renderWidth_, renderHeight_),
                    readTextureRgba(reference.color(1), renderWidth_, renderHeight_),
                    readTextureRgba(reference.color(2), renderWidth_, renderHeight_),
                    readTextureRgba(reference.color(3), renderWidth_, renderHeight_)
                );
                benchmarkAttributeQuality_
                    << test.pairId << ',' << test.sweep << ',' << METHOD_IDS[test.method] << ',' << layer << ','
                    << test.targetWidth << ',' << test.targetHeight << ',' << test.lightCount << ','
                    << std::setprecision(8) << test.alpha << ',' << DITHER_IDS[test.ditherMode] << ','
                    << COVERAGE_ENCODING_IDS[test.coverageEncodingMode] << ',' << test.kernelRadius << ','
                    << TRANSPARENT_GEOMETRY_IDS[test.transparentGeometry] << ','
                    << metrics.referenceSamples << ',' << metrics.predictedSamples << ',' << metrics.matchedSamples << ','
                    << std::setprecision(12) << metrics.matchRecall << ',' << metrics.falsePositiveRatio << ','
                    << metrics.wrongSurfaceSamples << ',' << metrics.albedoRmse << ','
                    << metrics.normalMeanDegrees << ',' << metrics.normalMedianDegrees << ',' << metrics.normalP95Degrees << ','
                    << metrics.roughnessRmse << ',' << metrics.roughnessMae << ','
                    << metrics.positionRmse << ',' << metrics.positionMax << ','
                    << metrics.alphaRmse << ',' << metrics.alphaMae << ',' << metrics.alphaMax << '\n';
            };

            writeLayer(0, reconstructedTransparentGBuffer_, attributeReferenceFront_);
            if (twoLayer) writeLayer(1, reconstructedTransparentLayer1GBuffer_, attributeReferenceBack_);
        }

        void writeOverdraw(const BenchmarkCase& test, const OverdrawMetrics& overdraw) {
            benchmarkOverdraw_
                << test.pairId << ',' << test.sweep << ',' << METHOD_IDS[test.method] << ',' << roddVariantId(test.roddAblationMask) << ',' << test.targetWidth << ',' << test.targetHeight << ','
                << test.lightCount << ',' << ALPHA_MODE_IDS[test.alphaMode] << ',' << DITHER_IDS[test.ditherMode] << ',' << COVERAGE_ENCODING_IDS[test.coverageEncodingMode] << ',' << test.coverageBias << ','
                << test.kernelRadius << ',' << TRANSPARENT_GEOMETRY_IDS[test.transparentGeometry] << ',' << test.transparentLayers << ','
                << std::setprecision(8) << test.alpha << ',' << std::setprecision(12)
                << overdraw.meanInputFragments << ',' << overdraw.maxInputFragments << ','
                << overdraw.meanSurvivingFragments << ',' << overdraw.maxSurvivingFragments << ','
                << overdraw.survivalRatio << ',' << overdraw.coveredPixelRatio << '\n';
        }

        void saveAttachmentCapture(
            gfx::research::Framebuffer& framebuffer,
            const int attachment,
            const unsigned int texture,
            const std::filesystem::path& png,
            const std::filesystem::path& pfm
        ) {
            framebuffer.bind();
            glReadBuffer(GL_COLOR_ATTACHMENT0 + attachment);
            gfx::research::save_framebuffer_png(png, renderWidth_, renderHeight_);
            saveTexturePfm(pfm, texture, renderWidth_, renderHeight_);
        }

        void writeStageCaptures(const BenchmarkCase& test) {
            if (!test.captureStages) return;
            const std::string base = test.pairId + "-pipeline-";

            if (test.method == METHOD_CONTROL_GUIDED_ANISOTROPIC) {
                saveAttachmentCapture(
                    mixedGBuffer_, 0, mixedGBuffer_.color(0),
                    benchmarkOutput_ / "captures" / (base + "01-gbuffer-albedo-roughness.png"),
                    benchmarkOutput_ / "captures" / (base + "01-gbuffer-albedo-roughness.pfm")
                );
                saveAttachmentCapture(
                    mixedGBuffer_, 1, mixedGBuffer_.color(1),
                    benchmarkOutput_ / "captures" / (base + "02-gbuffer-normal.png"),
                    benchmarkOutput_ / "captures" / (base + "02-gbuffer-normal.pfm")
                );
                saveAttachmentCapture(
                    mixedGBuffer_, 2, mixedGBuffer_.color(2),
                    benchmarkOutput_ / "captures" / (base + "03-gbuffer-world-position.png"),
                    benchmarkOutput_ / "captures" / (base + "03-gbuffer-world-position.pfm")
                );
                saveAttachmentCapture(
                    mixedGBuffer_, 3, mixedGBuffer_.color(3),
                    benchmarkOutput_ / "captures" / (base + "04-selected-sample-control.png"),
                    benchmarkOutput_ / "captures" / (base + "04-selected-sample-control.pfm")
                );
                saveAttachmentCapture(
                    mixedLit_, 0, mixedLit_.color(0),
                    benchmarkOutput_ / "captures" / (base + "05-deferred-lit-mixed-scene.png"),
                    benchmarkOutput_ / "captures" / (base + "05-deferred-lit-mixed-scene.pfm")
                );
                saveAttachmentCapture(
                    resultTarget_, 0, resultTarget_.color(0),
                    benchmarkOutput_ / "captures" / (base + "06-post-lighting-reconstruction.png"),
                    benchmarkOutput_ / "captures" / (base + "06-post-lighting-reconstruction.pfm")
                );
                saveAttachmentCapture(
                    resultTarget_, 1, resultTarget_.color(1),
                    benchmarkOutput_ / "captures" / (base + "07-reconstruction-meta.png"),
                    benchmarkOutput_ / "captures" / (base + "07-reconstruction-meta.pfm")
                );
                return;
            }

            if (test.method != METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                && test.method != METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                && test.method != METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE) return;

            saveAttachmentCapture(
                mixedGBuffer_, 3, mixedGBuffer_.color(3),
                benchmarkOutput_ / "captures" / (base + "01-sparse-selected-samples.png"),
                benchmarkOutput_ / "captures" / (base + "01-sparse-selected-samples.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentGBuffer_, 0, reconstructedTransparentGBuffer_.color(0),
                benchmarkOutput_ / "captures" / (base + "02-front-albedo-roughness.png"),
                benchmarkOutput_ / "captures" / (base + "02-front-albedo-roughness.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentGBuffer_, 1, reconstructedTransparentGBuffer_.color(1),
                benchmarkOutput_ / "captures" / (base + "03-front-normal.png"),
                benchmarkOutput_ / "captures" / (base + "03-front-normal.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentGBuffer_, 3, reconstructedTransparentGBuffer_.color(3),
                benchmarkOutput_ / "captures" / (base + "04-front-alpha-support.png"),
                benchmarkOutput_ / "captures" / (base + "04-front-alpha-support.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentLayer1GBuffer_, 0, reconstructedTransparentLayer1GBuffer_.color(0),
                benchmarkOutput_ / "captures" / (base + "05-back-albedo-roughness.png"),
                benchmarkOutput_ / "captures" / (base + "05-back-albedo-roughness.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentLayer1GBuffer_, 3, reconstructedTransparentLayer1GBuffer_.color(3),
                benchmarkOutput_ / "captures" / (base + "06-back-alpha-support.png"),
                benchmarkOutput_ / "captures" / (base + "06-back-alpha-support.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentLit_, 0, reconstructedTransparentLit_.color(0),
                benchmarkOutput_ / "captures" / (base + "07-front-lit-after-reconstruction.png"),
                benchmarkOutput_ / "captures" / (base + "07-front-lit-after-reconstruction.pfm")
            );
            saveAttachmentCapture(
                reconstructedTransparentLayer1Lit_, 0, reconstructedTransparentLayer1Lit_.color(0),
                benchmarkOutput_ / "captures" / (base + "08-back-lit-after-reconstruction.png"),
                benchmarkOutput_ / "captures" / (base + "08-back-lit-after-reconstruction.pfm")
            );
            saveAttachmentCapture(
                resultTarget_, 0, resultTarget_.color(0),
                benchmarkOutput_ / "captures" / (base + "09-final-composite.png"),
                benchmarkOutput_ / "captures" / (base + "09-final-composite.pfm")
            );
            saveAttachmentCapture(
                resultTarget_, 1, resultTarget_.color(1),
                benchmarkOutput_ / "captures" / (base + "10-final-meta.png"),
                benchmarkOutput_ / "captures" / (base + "10-final-meta.pfm")
            );
            saveAttachmentCapture(
                attributeReferenceFront_, 0, attributeReferenceFront_.color(0),
                benchmarkOutput_ / "captures" / (base + "11-reference-front-albedo-roughness.png"),
                benchmarkOutput_ / "captures" / (base + "11-reference-front-albedo-roughness.pfm")
            );
            saveAttachmentCapture(
                attributeReferenceFront_, 1, attributeReferenceFront_.color(1),
                benchmarkOutput_ / "captures" / (base + "12-reference-front-normal.png"),
                benchmarkOutput_ / "captures" / (base + "12-reference-front-normal.pfm")
            );
            saveAttachmentCapture(
                attributeReferenceFront_, 3, attributeReferenceFront_.color(3),
                benchmarkOutput_ / "captures" / (base + "13-reference-front-alpha.png"),
                benchmarkOutput_ / "captures" / (base + "13-reference-front-alpha.pfm")
            );
            saveAttachmentCapture(
                attributeReferenceBack_, 0, attributeReferenceBack_.color(0),
                benchmarkOutput_ / "captures" / (base + "14-reference-back-albedo-roughness.png"),
                benchmarkOutput_ / "captures" / (base + "14-reference-back-albedo-roughness.pfm")
            );
            saveAttachmentCapture(
                attributeReferenceBack_, 1, attributeReferenceBack_.color(1),
                benchmarkOutput_ / "captures" / (base + "15-reference-back-normal.png"),
                benchmarkOutput_ / "captures" / (base + "15-reference-back-normal.pfm")
            );
            saveAttachmentCapture(
                attributeReferenceBack_, 3, attributeReferenceBack_.color(3),
                benchmarkOutput_ / "captures" / (base + "16-reference-back-alpha.png"),
                benchmarkOutput_ / "captures" / (base + "16-reference-back-alpha.pfm")
            );
        }

        void writeCapture(const BenchmarkCase& test) {
            const std::string stem = test.pairId + "-" + METHOD_IDS[test.method];
            const std::filesystem::path png = benchmarkOutput_ / "captures" / (stem + ".png");
            const std::filesystem::path pfm = benchmarkOutput_ / "captures" / (stem + ".pfm");
            if (test.method == METHOD_DEPTH_PEEL_REFERENCE) referenceTarget_.bind();
            else resultTarget_.bind();
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            gfx::research::save_framebuffer_png(png, renderWidth_, renderHeight_);
            saveTexturePfm(pfm, selectedOutputTexture(), renderWidth_, renderHeight_);

            const std::string referenceStem = test.pairId + "-" + METHOD_IDS[METHOD_DEPTH_PEEL_REFERENCE];
            const std::filesystem::path referencePng = benchmarkOutput_ / "captures" / (referenceStem + ".png");
            const std::filesystem::path referencePfm = benchmarkOutput_ / "captures" / (referenceStem + ".pfm");
            if (!std::filesystem::exists(referencePng)) {
                referenceTarget_.bind();
                glReadBuffer(GL_COLOR_ATTACHMENT0);
                gfx::research::save_framebuffer_png(referencePng, renderWidth_, renderHeight_);
                saveTexturePfm(referencePfm, referenceTarget_.color(0), renderWidth_, renderHeight_);
            }

            const std::filesystem::path alphaPfm = benchmarkOutput_ / "captures" / (test.pairId + "-alpha-reference.pfm");
            if (!std::filesystem::exists(alphaPfm)) {
                saveTextureChannelPfm(alphaPfm, alphaReferenceTarget_.color(0), renderWidth_, renderHeight_, 0);
            }
            if (test.method != METHOD_DEPTH_PEEL_REFERENCE && test.method != METHOD_CLASSIC_FORWARD_ALPHA && test.method != METHOD_SORTED_FORWARD_ALPHA) {
                saveTextureChannelPfm(
                    benchmarkOutput_ / "captures" / (stem + "-coverage-estimate.pfm"),
                    resultTarget_.color(1), renderWidth_, renderHeight_, 0
                );
                saveTextureChannelPfm(
                    benchmarkOutput_ / "captures" / (stem + "-support-estimate.pfm"),
                    resultTarget_.color(1), renderWidth_, renderHeight_, 1
                );
                const int alphaChannel = (test.method == METHOD_CONTROL_GUIDED_ANISOTROPIC || test.method == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                    || test.method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                    || test.method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                    || test.method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE) ? 3 : 0;
                saveTextureChannelPfm(
                    benchmarkOutput_ / "captures" / (stem + "-alpha-estimate.pfm"),
                    resultTarget_.color(1), renderWidth_, renderHeight_, alphaChannel
                );
            }
            writeStageCaptures(test);
        }

        std::vector<BenchmarkCase> makeBaselineBenchmarkCases() const {
            std::vector<BenchmarkCase> cases = makeBenchmarkCases(false);
            const auto isKnownBaseline = [](const int method) {
                return method == METHOD_RAW_DITHER
                    || method == METHOD_OCCUPANCY_2X2
                    || method == METHOD_BOX
                    || method == METHOD_GAUSSIAN
                    || method == METHOD_POSITION_AWARE
                    || method == METHOD_WBOIT
                    || method == METHOD_LAYERED_WBOIT
                    || method == METHOD_CLASSIC_FORWARD_ALPHA
                    || method == METHOD_SORTED_FORWARD_ALPHA;
            };
            std::erase_if(cases, [&](const BenchmarkCase& test) { return !isKnownBaseline(test.method); });
            return cases;
        }

        std::vector<BenchmarkCase> makeBenchmarkCases(const bool full) const {
            std::vector<BenchmarkCase> cases;
            const std::vector<std::pair<int, int>> resolutions = full
                ? std::vector<std::pair<int, int>>{{1920, 1080}, {2560, 1440}, {3840, 2160}}
                : std::vector<std::pair<int, int>>{{1920, 1080}};
            const std::vector<float> timingAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            const std::vector<int> lightCounts = full ? std::vector<int>{16, 64} : std::vector<int>{32};
            const int repeats = full ? 5 : 1;
            const int warmup = full ? 45 : 5;
            const int samples = full ? 180 : 20;

            // Primary single-layer timing. The panel is single-sided, so full-rate alpha,
            // WBOIT and the dithered methods solve the same visibility problem.
            for (const auto [width, height] : resolutions) {
                for (const int lights : lightCounts) {
                    for (const float alpha : timingAlphas) {
                        for (int repeat = 0; repeat < repeats; ++repeat) {
                            const std::array<int, 13> normalOrder = {
                                METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_WBOIT, METHOD_LAYERED_WBOIT,
                                METHOD_RAW_DITHER, METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN,
                                METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL,
                                METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                            };
                            const std::array<int, 13> reverseOrder = {
                                METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION, METHOD_FULL_DEFERRED_LIGHT_CONTROL, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_TWO_COMPONENT_LINEAR, METHOD_POSITION_AWARE, METHOD_GAUSSIAN, METHOD_BOX,
                                METHOD_OCCUPANCY_2X2, METHOD_RAW_DITHER, METHOD_LAYERED_WBOIT,
                                METHOD_WBOIT, METHOD_SORTED_FORWARD_ALPHA, METHOD_CLASSIC_FORWARD_ALPHA
                            };
                            const auto& order = (repeat & 1) == 0 ? normalOrder : reverseOrder;
                            for (const int method : order) {
                                BenchmarkCase test;
                                test.pairId = "timing-single-" + std::to_string(width) + "x" + std::to_string(height)
                                    + "-l" + std::to_string(lights) + "-a" + scalarToken(alpha)
                                    + "-r" + std::to_string(repeat);
                                test.sweep = "timing-single";
                                test.method = method;
                                test.repeat = repeat;
                                test.targetWidth = width;
                                test.targetHeight = height;
                                test.lightCount = lights;
                                test.alpha = alpha;
                                test.alphaMode = 0;
                                test.ditherMode = 0;
                                test.kernelRadius = 2;
                                test.transparentGeometry = 0;
                                test.transparentLayers = 1;
                                test.warmupFrames = warmup;
                                test.sampleFrames = samples;
                                cases.push_back(test);
                            }
                        }
                    }
                }
            }

            // Main image-quality sweep against the depth-peeling per-pixel reference.
            const std::vector<float> qualityAlphas = full
                ? std::vector<float>{0.0625f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f}
                : std::vector<float>{0.5f};
            const int qualityWarmup = full ? 6 : 2;
            const int qualitySamples = full ? 12 : 4;
            const int alphaModes = full ? 4 : 1;
            for (int alphaMode = 0; alphaMode < alphaModes; ++alphaMode) {
                for (const float alpha : qualityAlphas) {
                    for (const int dither : {0, 3, 1}) {
                        for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_RAW_DITHER, METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL, METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION}) {
                            BenchmarkCase test;
                            test.pairId = std::string("quality-") + ALPHA_MODE_IDS[alphaMode] + "-" + DITHER_IDS[dither] + "-a" + scalarToken(alpha);
                            test.sweep = "quality";
                            test.method = method;
                            test.targetWidth = 1920;
                            test.targetHeight = 1080;
                            test.lightCount = 64;
                            test.alpha = alpha;
                            test.alphaMode = alphaMode;
                            test.ditherMode = dither;
                            test.kernelRadius = 2;
                            test.transparentGeometry = alphaMode == 2 ? 2 : 0;
                            test.transparentLayers = 1;
                            test.warmupFrames = qualityWarmup;
                            test.sampleFrames = qualitySamples;
                            test.quality = true;
                            test.capture = full && dither == 0 && (alpha == 0.25f || alpha == 0.5f || alpha == 0.75f);
                            cases.push_back(test);
                        }
                    }
                    // OIT baselines are independent of the dither pattern, so record them once.
                    for (const int method : {METHOD_WBOIT, METHOD_LAYERED_WBOIT}) {
                        BenchmarkCase test;
                        test.pairId = std::string("quality-") + ALPHA_MODE_IDS[alphaMode] + "-oit-a" + scalarToken(alpha);
                        test.sweep = "quality";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = alphaMode;
                        test.ditherMode = 0;
                        test.kernelRadius = 2;
                        test.transparentGeometry = alphaMode == 2 ? 2 : 0;
                        test.transparentLayers = 1;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = full && alpha == 0.5f;
                        test.recordOverdraw = method == METHOD_CONTROL_GUIDED_ANISOTROPIC;
                        cases.push_back(test);
                    }
                }
            }

            // Component ablation: isolate the contribution of each RODD stage.
            if (full) {
                const std::array<std::pair<int, const char*>, 8> ablations = {{
                    {RODD_ABLATE_NONE, "full-bayer"},
                    {RODD_ABLATE_ORDINAL, "no-ordinal"},
                    {RODD_ABLATE_CENSORED_NEGATIVES, "no-censored-negatives"},
                    {RODD_ABLATE_GEOMETRY, "no-geometry-gating"},
                    {RODD_ABLATE_RANK_CONDITIONING, "no-rank-conditioning"},
                    {RODD_ABLATE_MOMENTS, "no-moment-reproduction"},
                    {RODD_ABLATE_DIRECTIONAL_PAIRS, "no-directional-pairs"},
                    {RODD_ABLATE_CONFIDENCE_GUARD, "no-confidence-guard"}
                }};
                for (const float alpha : {0.25f, 0.5f, 0.75f}) {
                    for (const int geometry : {0, 3, 5}) {
                        for (const auto& [mask, label] : ablations) {
                            BenchmarkCase test;
                            test.pairId = std::string("rodd-ablation-") + label + "-" + TRANSPARENT_GEOMETRY_IDS[geometry] + "-a" + scalarToken(alpha);
                            test.sweep = "rodd-ablation";
                            test.method = METHOD_CONTROL_GUIDED_ANISOTROPIC;
                            test.roddAblationMask = mask;
                            test.targetWidth = 1920;
                            test.targetHeight = 1080;
                            test.lightCount = 64;
                            test.alpha = alpha;
                            test.alphaMode = 0;
                            test.ditherMode = 0;
                            test.kernelRadius = 2;
                            test.transparentGeometry = geometry;
                            test.transparentLayers = 1;
                            test.warmupFrames = qualityWarmup;
                            test.sampleFrames = qualitySamples;
                            test.quality = true;
                            test.capture = alpha == 0.5f && geometry == 3;
                            cases.push_back(test);
                        }

                        BenchmarkCase optimized;
                        optimized.pairId = std::string("rodd-ablation-full-optimized-pattern-") + TRANSPARENT_GEOMETRY_IDS[geometry] + "-a" + scalarToken(alpha);
                        optimized.sweep = "rodd-ablation";
                        optimized.method = METHOD_CONTROL_GUIDED_ANISOTROPIC;
                        optimized.roddAblationMask = RODD_ABLATE_NONE;
                        optimized.targetWidth = 1920;
                        optimized.targetHeight = 1080;
                        optimized.lightCount = 64;
                        optimized.alpha = alpha;
                        optimized.alphaMode = 0;
                        optimized.ditherMode = 2;
                        optimized.kernelRadius = 2;
                        optimized.transparentGeometry = geometry;
                        optimized.transparentLayers = 1;
                        optimized.warmupFrames = qualityWarmup;
                        optimized.sampleFrames = qualitySamples;
                        optimized.quality = true;
                        optimized.capture = alpha == 0.5f && geometry == 3;
                        cases.push_back(optimized);
                    }
                }
            }

            // Measurement-design sweep: larger ordered matrices and invertible alpha companding.
            // The encoded coverage is allowed to be biased because RODD reconstructs F/B
            // separately and decodes the original-alpha threshold before final composition.
            if (full) {
                for (const float alpha : {0.125f, 0.25f, 0.5f, 0.75f, 0.875f}) {
                    for (const int dither : {0, 3, 4}) {
                        for (const int encoding : {0, 1, 2, 3, 4}) {
                            const std::vector<float> biases = encoding == 0
                                ? std::vector<float>{0.0f}
                                : (encoding == 3
                                    ? std::vector<float>{0.35f, 0.65f, 1.15f}
                                    : (encoding == 4 ? std::vector<float>{0.15f, 0.35f, 0.55f, 0.75f} : std::vector<float>{0.65f, 1.15f}));
                            for (const float bias : biases) {
                                BenchmarkCase test;
                                test.pairId = std::string("measurement-design-") + DITHER_IDS[dither] + "-" + COVERAGE_ENCODING_IDS[encoding] + "-b" + scalarToken(bias) + "-a" + scalarToken(alpha);
                                test.sweep = "measurement-design";
                                test.method = METHOD_CONTROL_GUIDED_ANISOTROPIC;
                                test.targetWidth = 1920;
                                test.targetHeight = 1080;
                                test.lightCount = 64;
                                test.alpha = alpha;
                                test.alphaMode = 0;
                                test.ditherMode = dither;
                                test.coverageEncodingMode = encoding;
                                test.coverageBias = bias;
                                test.kernelRadius = 3;
                                test.transparentGeometry = 0;
                                test.transparentLayers = 1;
                                test.warmupFrames = qualityWarmup;
                                test.sampleFrames = qualitySamples;
                                test.quality = true;
                                test.recordOverdraw = true;
                                test.capture = alpha == 0.5f && (encoding == 0 || encoding == 3 || encoding == 4);
                                cases.push_back(test);
                            }
                        }
                    }
                }
            }

            // Mixed-scene quality sweep with multiple transparent objects inside Sponza.
            for (const float alpha : qualityAlphas) {
                for (const int dither : {0, 3, 1}) {
                    for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_RAW_DITHER, METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL}) {
                        BenchmarkCase test;
                        test.pairId = std::string("mixed-quality-") + DITHER_IDS[dither] + "-a" + scalarToken(alpha);
                        test.sweep = "mixed-quality";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 0;
                        test.ditherMode = dither;
                        test.kernelRadius = 2;
                        test.transparentGeometry = 4;
                        test.transparentLayers = 1;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = full && dither == 0 && alpha == 0.5f;
                        cases.push_back(test);
                    }
                }
                for (const int method : {METHOD_WBOIT, METHOD_LAYERED_WBOIT}) {
                    BenchmarkCase test;
                    test.pairId = std::string("mixed-quality-oit-a") + scalarToken(alpha);
                    test.sweep = "mixed-quality";
                    test.method = method;
                    test.targetWidth = 1920;
                    test.targetHeight = 1080;
                    test.lightCount = 64;
                    test.alpha = alpha;
                    test.alphaMode = 0;
                    test.ditherMode = 0;
                    test.kernelRadius = 2;
                    test.transparentGeometry = 4;
                    test.transparentLayers = 1;
                    test.warmupFrames = qualityWarmup;
                    test.sampleFrames = qualitySamples;
                    test.quality = true;
                    test.capture = full && alpha == 0.5f;
                    cases.push_back(test);
                }
            }

            // Explicit architecture-audit case: multiple transparent objects must all
            // populate the same shared layered G-buffer before any transparent lighting.
            {
                BenchmarkCase test;
                test.pairId = "shared-gbuffer-audit-mixed";
                test.sweep = "shared-gbuffer-audit";
                test.method = METHOD_CONTROL_GUIDED_ANISOTROPIC;
                test.targetWidth = 1920;
                test.targetHeight = 1080;
                test.lightCount = 64;
                test.alpha = 0.5f;
                test.alphaMode = 0;
                test.ditherMode = 0;
                test.kernelRadius = 2;
                test.transparentGeometry = 4;
                test.transparentLayers = 1;
                test.warmupFrames = qualityWarmup;
                test.sampleFrames = qualitySamples;
                test.quality = true;
                test.capture = true;
                test.captureStages = true;
                test.recordOverdraw = true;
                cases.push_back(test);
            }

            // Angled-plates sweep. One card is rotated 45 degrees and one 90 degrees
            // so the methods are tested away from the camera-parallel configuration.
            for (const float alpha : qualityAlphas) {
                for (const int dither : {0, 3, 1}) {
                    for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_RAW_DITHER, METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL}) {
                        BenchmarkCase test;
                        test.pairId = std::string("angled-quality-") + DITHER_IDS[dither] + "-a" + scalarToken(alpha);
                        test.sweep = "angled-quality";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 0;
                        test.ditherMode = dither;
                        test.kernelRadius = 2;
                        test.transparentGeometry = 5;
                        test.transparentLayers = 1;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = full && dither == 0 && alpha == 0.5f;
                        cases.push_back(test);
                    }
                }
                for (const int method : {METHOD_WBOIT, METHOD_LAYERED_WBOIT}) {
                    BenchmarkCase test;
                    test.pairId = std::string("angled-quality-oit-a") + scalarToken(alpha);
                    test.sweep = "angled-quality";
                    test.method = method;
                    test.targetWidth = 1920;
                    test.targetHeight = 1080;
                    test.lightCount = 64;
                    test.alpha = alpha;
                    test.alphaMode = 0;
                    test.ditherMode = 0;
                    test.kernelRadius = 2;
                    test.transparentGeometry = 5;
                    test.transparentLayers = 1;
                    test.warmupFrames = qualityWarmup;
                    test.sampleFrames = qualitySamples;
                    test.quality = true;
                    test.capture = full && alpha == 0.5f;
                    cases.push_back(test);
                }
            }

            // Additional timing sweeps on the mixed scene and angled plates use one
            // representative lighting/alpha configuration. Resolution, alpha and light-count
            // scaling are already covered by the primary single-layer timing sweep.
            const std::vector<std::pair<int, int>> sceneTimingResolutions = full
                ? std::vector<std::pair<int, int>>{{1920, 1080}, {2560, 1440}}
                : std::vector<std::pair<int, int>>{{1920, 1080}};
            const std::vector<int> sceneTimingLights = full ? std::vector<int>{64} : std::vector<int>{32};
            const std::vector<float> sceneTimingAlphas = {0.5f};
            for (const auto [width, height] : sceneTimingResolutions) {
                for (const int lights : sceneTimingLights) {
                    for (int geometry : {4, 5}) {
                        for (const float alpha : sceneTimingAlphas) {
                            for (int repeat = 0; repeat < repeats; ++repeat) {
                                const std::array<int, 11> normalOrder = {
                                    METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_WBOIT, METHOD_LAYERED_WBOIT,
                                    METHOD_RAW_DITHER, METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN,
                                    METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC
                                };
                                const std::array<int, 11> reverseOrder = {
                                    METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_TWO_COMPONENT_LINEAR, METHOD_POSITION_AWARE, METHOD_GAUSSIAN, METHOD_BOX,
                                    METHOD_OCCUPANCY_2X2, METHOD_RAW_DITHER, METHOD_LAYERED_WBOIT,
                                    METHOD_WBOIT, METHOD_SORTED_FORWARD_ALPHA, METHOD_CLASSIC_FORWARD_ALPHA
                                };
                                const auto& order = (repeat & 1) == 0 ? normalOrder : reverseOrder;
                                for (const int method : order) {
                                    BenchmarkCase test;
                                    test.pairId = std::string(geometry == 4 ? "timing-mixed-" : "timing-angled-")
                                        + std::to_string(width) + "x" + std::to_string(height)
                                        + "-l" + std::to_string(lights) + "-a" + scalarToken(alpha)
                                        + "-r" + std::to_string(repeat);
                                    test.sweep = geometry == 4 ? "timing-mixed" : "timing-angled";
                                    test.method = method;
                                    test.repeat = repeat;
                                    test.targetWidth = width;
                                    test.targetHeight = height;
                                    test.lightCount = lights;
                                    test.alpha = alpha;
                                    test.alphaMode = 0;
                                    test.ditherMode = 0;
                                    test.kernelRadius = 2;
                                    test.transparentGeometry = geometry;
                                    test.transparentLayers = 1;
                                    test.warmupFrames = warmup;
                                    test.sampleFrames = samples;
                                    cases.push_back(test);
                                }
                            }
                        }
                    }
                }
            }

            // Geometric discontinuity sweep for the spatial decoder.
            const std::vector<float> edgeAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            const std::vector<int> edgeDithers = full ? std::vector<int>{0, 3, 1} : std::vector<int>{0};
            for (const float alpha : edgeAlphas) {
                for (const int dither : edgeDithers) {
                    for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_BOX, METHOD_GAUSSIAN, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL}) {
                        BenchmarkCase test;
                        test.pairId = std::string("depth-edge-") + DITHER_IDS[dither] + "-a" + scalarToken(alpha);
                        test.sweep = "depth-edge";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 0;
                        test.ditherMode = dither;
                        test.kernelRadius = 2;
                        test.transparentGeometry = 3;
                        test.transparentLayers = 1;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = full && dither == 0 && alpha == 0.5f;
                        cases.push_back(test);
                    }
                }
            }

            // Kernel radius sweep. Forward rendering is recorded for every radius as
            // the zero-error full-rate reference so exported tables remain self-contained.
            const std::vector<float> kernelAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            for (const int radius : {1, 2, 3}) {
                for (const float alpha : kernelAlphas) {
                    for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC}) {
                        BenchmarkCase test;
                        test.pairId = "kernel-r" + std::to_string(radius) + "-a" + scalarToken(alpha);
                        test.sweep = "kernel";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 1;
                        test.ditherMode = 0;
                        test.kernelRadius = radius;
                        test.transparentGeometry = 0;
                        test.transparentLayers = 1;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        cases.push_back(test);
                    }
                }
            }

            // Multi-layer transparency context. Parallel cards give an exact object-sorted
            // back-to-front reference. Coverage reconstruction processes every transparent
            // object layer independently and composites the reconstructed layers back-to-front.
            const std::vector<int> stackLayers = full ? std::vector<int>{2, 4, 8, 16} : std::vector<int>{4};
            const std::vector<float> stackAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            for (const int layers : stackLayers) {
                for (const float alpha : stackAlphas) {
                    for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_RAW_DITHER, METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL, METHOD_WBOIT, METHOD_LAYERED_WBOIT}) {
                        BenchmarkCase test;
                        test.pairId = "oit-quality-n" + std::to_string(layers) + "-a" + scalarToken(alpha);
                        test.sweep = "oit-stack-quality";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 0;
                        test.ditherMode = 0;
                        test.kernelRadius = 2;
                        test.transparentGeometry = 6;
                        test.transparentLayers = layers;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = full && alpha == 0.5f;
                        test.recordOverdraw = method == METHOD_CONTROL_GUIDED_ANISOTROPIC;
                        cases.push_back(test);
                    }
                }
            }

            const std::vector<std::pair<int, int>> oitTimingResolutions = full
                ? std::vector<std::pair<int, int>>{{1920, 1080}, {2560, 1440}}
                : std::vector<std::pair<int, int>>{{1920, 1080}};
            for (const auto [width, height] : oitTimingResolutions) {
                for (const int layers : stackLayers) {
                    for (int repeat = 0; repeat < repeats; ++repeat) {
                        const std::array<int, 12> normalOrder = {
                            METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_WBOIT, METHOD_LAYERED_WBOIT, METHOD_RAW_DITHER,
                            METHOD_OCCUPANCY_2X2, METHOD_BOX, METHOD_GAUSSIAN, METHOD_POSITION_AWARE, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL
                        };
                        const std::array<int, 12> reverseOrder = {
                            METHOD_FULL_DEFERRED_LIGHT_CONTROL, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_TWO_COMPONENT_LINEAR, METHOD_POSITION_AWARE, METHOD_GAUSSIAN, METHOD_BOX,
                            METHOD_OCCUPANCY_2X2, METHOD_RAW_DITHER, METHOD_LAYERED_WBOIT, METHOD_WBOIT,
                            METHOD_SORTED_FORWARD_ALPHA, METHOD_CLASSIC_FORWARD_ALPHA
                        };
                        const auto& order = (repeat & 1) == 0 ? normalOrder : reverseOrder;
                        for (const int method : order) {
                            BenchmarkCase test;
                            test.pairId = "oit-timing-" + std::to_string(width) + "x" + std::to_string(height)
                                + "-n" + std::to_string(layers) + "-r" + std::to_string(repeat);
                            test.sweep = "oit-stack-timing";
                            test.method = method;
                            test.repeat = repeat;
                            test.targetWidth = width;
                            test.targetHeight = height;
                            test.lightCount = 64;
                            test.alpha = 0.5f;
                            test.alphaMode = 0;
                            test.ditherMode = 0;
                            test.kernelRadius = 2;
                            test.transparentGeometry = 6;
                            test.transparentLayers = layers;
                            test.warmupFrames = warmup;
                            test.sampleFrames = samples;
                            test.recordOverdraw = method == METHOD_CONTROL_GUIDED_ANISOTROPIC;
                            cases.push_back(test);
                        }
                    }
                }
            }

            // Transparent depth-complexity / overdraw sweep.
            const std::vector<int> overdrawLayers = full ? std::vector<int>{1, 2, 4, 8, 16} : std::vector<int>{1, 4, 8, 16};
            const std::vector<float> overdrawAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            const std::vector<std::pair<int, int>> overdrawResolutions = {{1920, 1080}};
            const int overdrawRepeats = full ? 3 : 1;
            const int overdrawWarmup = full ? 30 : 4;
            const int overdrawSamples = full ? 120 : 16;
            for (const auto [width, height] : overdrawResolutions) {
                for (const int layers : overdrawLayers) {
                    for (const float alpha : overdrawAlphas) {
                        for (int repeat = 0; repeat < overdrawRepeats; ++repeat) {
                            const std::array<int, 7> order = {
                                METHOD_CLASSIC_FORWARD_ALPHA,
                                METHOD_SORTED_FORWARD_ALPHA,
                                METHOD_RAW_DITHER,
                                METHOD_WBOIT,
                                METHOD_LAYERED_WBOIT,
                                METHOD_POSITION_AWARE,
                                METHOD_TWO_COMPONENT_LINEAR
                            };
                            for (const int method : order) {
                                BenchmarkCase test;
                                test.pairId = "overdraw-" + std::to_string(width) + "x" + std::to_string(height)
                                    + "-n" + std::to_string(layers) + "-a" + scalarToken(alpha)
                                    + "-r" + std::to_string(repeat);
                                test.sweep = "overdraw";
                                test.method = method;
                                test.repeat = repeat;
                                test.targetWidth = width;
                                test.targetHeight = height;
                                test.lightCount = 64;
                                test.alpha = alpha;
                                test.alphaMode = 0;
                                test.ditherMode = 0;
                                test.kernelRadius = 2;
                                test.transparentGeometry = 6;
                                test.transparentLayers = layers;
                                test.warmupFrames = overdrawWarmup;
                                test.sampleFrames = overdrawSamples;
                                test.recordOverdraw = method == METHOD_CONTROL_GUIDED_ANISOTROPIC;
                                cases.push_back(test);
                            }
                        }
                    }
                }
            }

            // Texture-edge preservation sweep. The panel uses the real curtain texture,
            // so gradient-relative-RMSE directly measures whether the decoder preserves
            // high-frequency texture boundaries rather than only geometry/depth edges.
            const std::vector<float> textureAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            for (const float alpha : textureAlphas) {
                for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_GAUSSIAN, METHOD_TWO_COMPONENT_LINEAR, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL}) {
                    BenchmarkCase test;
                    test.pairId = "texture-protection-a" + scalarToken(alpha);
                    test.sweep = "texture-protection";
                    test.method = method;
                    test.targetWidth = 1920;
                    test.targetHeight = 1080;
                    test.lightCount = 64;
                    test.alpha = alpha;
                    test.alphaMode = 0;
                    test.ditherMode = 0;
                    test.kernelRadius = 2;
                    test.transparentGeometry = 0;
                    test.transparentLayers = 1;
                    test.warmupFrames = qualityWarmup;
                    test.sampleFrames = qualitySamples;
                    test.quality = true;
                    test.capture = full && alpha == 0.5f;
                    cases.push_back(test);
                }
            }

            // Core hypothesis: source alpha is converted to occupancy in a full-rate
            // UNLIT G-buffer. Occupied pixels are compacted to an explicit list and only
            // that list executes the expensive deferred-light compute kernel. The
            // full-rate control uses the same G-buffer and the same lighting kernel, so
            // the comparison isolates post-G-buffer filtering plus sparse execution from full-rate work.
            const std::vector<float> orderAlphas = full ? std::vector<float>{0.25f, 0.5f, 0.75f} : std::vector<float>{0.5f};
            for (const float alpha : orderAlphas) {
                for (const int geometry : {0, 1}) {
                    for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL}) {
                        BenchmarkCase test;
                        test.pairId = std::string("reconstruction-order-") + TRANSPARENT_GEOMETRY_IDS[geometry]
                            + "-a" + scalarToken(alpha);
                        test.sweep = "reconstruction-order";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 0;
                        test.ditherMode = 0;
                        test.kernelRadius = 2;
                        test.transparentGeometry = geometry;
                        test.transparentLayers = 1;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = full && alpha == 0.5f;
                        cases.push_back(test);
                    }
                }
            }

            // Light-count scaling verifies that proposed and raw-control use the same
            // full-frame deferred-lighting workload. Their deferred_lighting_ms values
            // should therefore track closely; the method-specific cost is reconstruction
            // after lighting, not a hidden sparse or forward lighting path.
            const std::vector<int> scalingLights = full
                ? std::vector<int>{1, 4, 16, 64, 128, 256}
                : std::vector<int>{4, 64, 256};
            const int scalingRepeats = full ? 3 : 1;
            const int scalingWarmup = full ? 30 : 5;
            const int scalingSamples = full ? 120 : 20;
            for (const int lights : scalingLights) {
                for (int repeat = 0; repeat < scalingRepeats; ++repeat) {
                    const std::array<int, 6> normalOrder = {
                        METHOD_CLASSIC_FORWARD_ALPHA,
                        METHOD_SORTED_FORWARD_ALPHA,
                        METHOD_WBOIT,
                        METHOD_CONTROL_GUIDED_ANISOTROPIC,
                        METHOD_FULL_DEFERRED_LIGHT_CONTROL,
                        METHOD_LAYERED_WBOIT
                    };
                    const std::array<int, 6> reverseOrder = {
                        METHOD_LAYERED_WBOIT,
                        METHOD_FULL_DEFERRED_LIGHT_CONTROL,
                        METHOD_CONTROL_GUIDED_ANISOTROPIC,
                        METHOD_WBOIT,
                        METHOD_SORTED_FORWARD_ALPHA,
                        METHOD_CLASSIC_FORWARD_ALPHA
                    };
                    const auto& order = (repeat & 1) == 0 ? normalOrder : reverseOrder;
                    for (const int method : order) {
                        BenchmarkCase test;
                        test.pairId = "light-scaling-l" + std::to_string(lights) + "-r" + std::to_string(repeat);
                        test.sweep = "light-scaling";
                        test.method = method;
                        test.repeat = repeat;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = lights;
                        test.alpha = 0.5f;
                        test.alphaMode = 0;
                        test.ditherMode = 0;
                        test.kernelRadius = 2;
                        test.transparentGeometry = 1;
                        test.transparentLayers = 1;
                        test.warmupFrames = scalingWarmup;
                        test.sampleFrames = scalingSamples;
                        test.recordDeferredLighting = method == METHOD_CONTROL_GUIDED_ANISOTROPIC || method == METHOD_FULL_DEFERRED_LIGHT_CONTROL;
                        test.recordOverdraw = method == METHOD_CONTROL_GUIDED_ANISOTROPIC && lights == 64 && repeat == 0;
                        cases.push_back(test);
                    }
                }
            }

            // Matching quality at different lighting costs. This lets the paper verify
            // that error does not silently grow as small/high-frequency highlights become
            // more complex. These cases are intentionally short and separate from timing.
            const std::vector<int> scalingQualityLights = full
                ? std::vector<int>{1, 6, 16, 64, 128, 256}
                : std::vector<int>{6, 64, 256};
            for (const int lights : scalingQualityLights) {
                for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_CONTROL_GUIDED_ANISOTROPIC, METHOD_FULL_DEFERRED_LIGHT_CONTROL, METHOD_WBOIT, METHOD_LAYERED_WBOIT}) {
                    BenchmarkCase test;
                    test.pairId = "light-quality-l" + std::to_string(lights);
                    test.sweep = "light-scaling-quality";
                    test.method = method;
                    test.targetWidth = 1920;
                    test.targetHeight = 1080;
                    test.lightCount = lights;
                    test.alpha = 0.5f;
                    test.alphaMode = 0;
                    test.ditherMode = 0;
                    test.kernelRadius = 2;
                    test.transparentGeometry = 1;
                    test.transparentLayers = 1;
                    test.warmupFrames = qualityWarmup;
                    test.sampleFrames = qualitySamples;
                    test.quality = true;
                    test.capture = full && lights == 6;
                    test.captureStages = full && lights == 6 && method == METHOD_CONTROL_GUIDED_ANISOTROPIC;
                    cases.push_back(test);
                }
            }

            // Temporal stability sequences. Every moving pose has its own matched
            // full-rate reference, so offline analysis measures temporal variation of
            // the reconstruction residual rather than mistaking real scene motion for
            // flicker. No history is used by RODD itself.
            if (full) {
                constexpr int temporalFrames = 7;
                for (const int dither : {0, 2}) {
                    for (int frame = 0; frame < temporalFrames; ++frame) {
                        const float centered = static_cast<float>(frame - temporalFrames / 2);
                        for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_GAUSSIAN, METHOD_CONTROL_GUIDED_ANISOTROPIC}) {
                            BenchmarkCase test;
                            test.pairId = std::string("temporal-camera-") + DITHER_IDS[dither] + "-f" + (frame < 10 ? "0" : "") + std::to_string(frame);
                            test.sweep = "temporal-camera";
                            test.method = method;
                            test.targetWidth = 640;
                            test.targetHeight = 360;
                            test.lightCount = 64;
                            test.alpha = 0.5f;
                            test.alphaMode = 3;
                            test.ditherMode = dither;
                            test.kernelRadius = 2;
                            test.transparentGeometry = 0;
                            test.transparentLayers = 1;
                            test.cameraYawOffset = centered * 0.00055f;
                            test.warmupFrames = 2;
                            test.sampleFrames = 3;
                            test.quality = true;
                            test.capture = true;
                            cases.push_back(test);
                        }
                    }

                    for (int frame = 0; frame < temporalFrames; ++frame) {
                        const float centered = static_cast<float>(frame - temporalFrames / 2);
                        for (const int method : {METHOD_CLASSIC_FORWARD_ALPHA, METHOD_SORTED_FORWARD_ALPHA, METHOD_GAUSSIAN, METHOD_CONTROL_GUIDED_ANISOTROPIC}) {
                            BenchmarkCase test;
                            test.pairId = std::string("temporal-object-") + DITHER_IDS[dither] + "-f" + (frame < 10 ? "0" : "") + std::to_string(frame);
                            test.sweep = "temporal-object";
                            test.method = method;
                            test.targetWidth = 640;
                            test.targetHeight = 360;
                            test.lightCount = 64;
                            test.alpha = 0.5f;
                            test.alphaMode = 3;
                            test.ditherMode = dither;
                            test.kernelRadius = 2;
                            test.transparentGeometry = 0;
                            test.transparentLayers = 1;
                            test.objectOffsetX = centered * 0.004f;
                            test.warmupFrames = 2;
                            test.sampleFrames = 3;
                            test.quality = true;
                            test.capture = true;
                            cases.push_back(test);
                        }
                    }
                }
            }

            // WBOIT weight sensitivity. This is important because the literature notes
            // scene-dependent tuning of the depth weighting function.
            if (full) {
                for (const float exponent : {1.0f, 2.0f, 3.0f, 4.0f, 6.0f}) {
                    BenchmarkCase test;
                    test.pairId = "wboit-weight-" + scalarToken(exponent);
                    test.sweep = "wboit-weight";
                    test.method = METHOD_WBOIT;
                    test.targetWidth = 1920;
                    test.targetHeight = 1080;
                    test.lightCount = 64;
                    test.alpha = 0.5f;
                    test.alphaMode = 0;
                    test.transparentGeometry = 6;
                    test.transparentLayers = 4;
                    test.wboitDepthExponent = exponent;
                    test.warmupFrames = qualityWarmup;
                    test.sampleFrames = qualitySamples;
                    test.quality = true;
                    cases.push_back(test);
                }
            }
            return cases;
        }

        std::vector<BenchmarkCase> makeEdgeDecisionBenchmarkCases() const {
            std::vector<BenchmarkCase> cases;
            constexpr int qualityWarmup = 12;
            constexpr int qualitySamples = 36;
            constexpr int timingWarmup = 40;
            constexpr int timingSamples = 180;

            const auto addQuality = [&](const std::string& pairId, const int method, const int geometry, const int layers, const float alpha, const bool capture) {
                BenchmarkCase test;
                test.pairId = pairId;
                test.sweep = "v32-edge-decision-quality";
                test.method = method;
                test.targetWidth = 1920;
                test.targetHeight = 1080;
                test.lightCount = 64;
                test.alpha = alpha;
                test.alphaMode = 0;
                test.ditherMode = 1; // stable hash: no Bayer lattice in the decision test.
                const bool sparseTwoLayer = method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                    || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                const bool useHighAlphaReveal = sparseTwoLayer && geometry == 6 && layers >= 4 && alpha > 0.5f;
                test.coverageEncodingMode = useHighAlphaReveal ? 5 : 0;
                test.coverageBias = useHighAlphaReveal ? 1.5f : 0.0f;
                test.kernelRadius = 2;
                test.transparentGeometry = geometry;
                test.transparentLayers = layers;
                test.warmupFrames = qualityWarmup;
                test.sampleFrames = qualitySamples;
                test.quality = true;
                test.capture = capture;
                test.captureStages = capture && (method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                    || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
                test.recordOverdraw = method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                test.recordDeferredLighting = method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                    || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                cases.push_back(test);
            };

            // Primary GO/NO-GO: preserve the high-alpha color advantage while reducing
            // the gradient penalty. The isotropic K=2 path is the direct ablation.
            for (const float alpha : {0.75f, 0.875f}) {
                const std::string id = "v32-depth-step-a" + scalarToken(alpha);
                for (const int method : {
                    METHOD_WBOIT,
                    METHOD_LAYERED_WBOIT,
                    METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                    METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                    METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                }) {
                    addQuality(id, method, 3, 1, alpha, true);
                }
            }

            // Secondary GO/NO-GO from the plan: four layers at alpha=0.75.
            for (const int method : {
                METHOD_WBOIT,
                METHOD_LAYERED_WBOIT,
                METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
            }) {
                addQuality("v32-stack4-a0p750", method, 6, 4, 0.75f, true);
            }

            // Attribute diagnostics: simple single surface plus two overlap geometries.
            for (const int geometry : {1, 3, 5}) {
                for (const int method : {METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION, METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC, METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE}) {
                    addQuality(std::string("v32-attribute-") + TRANSPARENT_GEOMETRY_IDS[geometry] + "-a0p750",
                        method, geometry, geometry == 6 ? 4 : 1, 0.75f, false);
                }
            }

            // Small timing sample: enough to detect catastrophic regressions before the
            // expensive publication timing sweep with ten full repeats.
            for (int repeat = 0; repeat < 3; ++repeat) {
                for (const int method : {
                    METHOD_WBOIT,
                    METHOD_LAYERED_WBOIT,
                    METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                    METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                    METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                }) {
                    BenchmarkCase test;
                    test.pairId = "v32-perf-depth-step-r" + std::to_string(repeat);
                    test.sweep = "v32-edge-decision-performance";
                    test.method = method;
                    test.repeat = repeat;
                    test.targetWidth = 1920;
                    test.targetHeight = 1080;
                    test.lightCount = 64;
                    test.alpha = 0.75f;
                    test.alphaMode = 0;
                    test.ditherMode = 1;
                    test.coverageEncodingMode = 0;
                    test.coverageBias = 0.0f;
                    test.kernelRadius = 2;
                    test.transparentGeometry = 3;
                    test.transparentLayers = 1;
                    test.warmupFrames = timingWarmup;
                    test.sampleFrames = timingSamples;
                    test.recordDeferredLighting = method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                        || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                        || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                    cases.push_back(test);
                }
            }
            return cases;
        }

        std::vector<BenchmarkCase> makePaperBenchmarkCases() const {
            std::vector<BenchmarkCase> cases;
            cases.reserve(420);

            constexpr int qualityWarmup = 8;
            constexpr int qualitySamples = 24;
            constexpr int timingWarmup = 30;
            constexpr int timingSamples = 120;

            const auto addQuality = [&](
                const std::string& pairId,
                const std::string& sweep,
                const int method,
                const int geometry,
                const float alpha,
                const int dither,
                const int encoding,
                const float encodingParameter,
                const int lights,
                const int kernel,
                const bool capture,
                const bool captureStages,
                const bool recordOverdraw
            ) {
                BenchmarkCase test;
                test.pairId = pairId;
                test.sweep = sweep;
                test.method = method;
                test.targetWidth = 1920;
                test.targetHeight = 1080;
                test.lightCount = lights;
                test.alpha = alpha;
                test.alphaMode = geometry == 2 ? 2 : 0;
                test.ditherMode = dither;
                test.coverageEncodingMode = encoding;
                test.coverageBias = encodingParameter;
                test.kernelRadius = kernel;
                test.transparentGeometry = geometry;
                test.transparentLayers = geometry == 6 ? 4 : 1;
                test.warmupFrames = qualityWarmup;
                test.sampleFrames = qualitySamples;
                test.quality = true;
                test.capture = capture;
                test.captureStages = captureStages;
                test.recordOverdraw = recordOverdraw;
                test.recordDeferredLighting = method == METHOD_CONTROL_GUIDED_ANISOTROPIC
                    || method == METHOD_FULL_DEFERRED_LIGHT_CONTROL
                    || method == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                    || method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                    || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                cases.push_back(test);
            };

            // P1: Single-surface quality. This establishes whether reconstructing material
            // attributes before lighting actually improves over post-lighting reconstruction.
            for (const int geometry : {0, 1, 2}) {
                for (const float alpha : {0.125f, 0.25f, 0.5f, 0.75f, 0.875f}) {
                    for (const int dither : {0, 3, 1}) {
                        for (const int method : {
                            METHOD_RAW_DITHER,
                            METHOD_POSITION_AWARE,
                            METHOD_CONTROL_GUIDED_ANISOTROPIC,
                            METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION,
                            METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                            METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                            METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                        }) {
                            const std::string id = std::string("pub-single-") + TRANSPARENT_GEOMETRY_IDS[geometry]
                                + "-" + DITHER_IDS[dither] + "-a" + scalarToken(alpha);
                            const bool keyCapture = geometry == 1 && dither == 3
                                && (std::abs(alpha - 0.25f) < 1e-6f || std::abs(alpha - 0.75f) < 1e-6f)
                                && (method == METHOD_CONTROL_GUIDED_ANISOTROPIC
                                    || method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                                    || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
                            addQuality(id, "publication-single-quality", method, geometry, alpha, dither, 0, 0.0f, 64, 2,
                                keyCapture, keyCapture,
                                method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                                    || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
                        }
                    }

                    // OIT/forward baselines are independent of dither phase; record them once.
                    for (const int method : {
                        METHOD_CLASSIC_FORWARD_ALPHA,
                        METHOD_SORTED_FORWARD_ALPHA,
                        METHOD_WBOIT,
                        METHOD_LAYERED_WBOIT
                    }) {
                        const std::string id = std::string("pub-single-") + TRANSPARENT_GEOMETRY_IDS[geometry]
                            + "-baseline-a" + scalarToken(alpha);
                        addQuality(id, "publication-single-quality", method, geometry, alpha, 3, 0, 0.0f, 64, 2,
                            false, false, false);
                    }
                }
            }

            // P2: Lighting-complexity stress. Attribute-space reconstruction should not
            // interpolate already-created specular highlights, and its shading cost is
            // bounded by the number of reconstructed layers rather than input overdraw.
            for (const int lights : {1, 16, 64, 256}) {
                for (const int method : {
                    METHOD_CLASSIC_FORWARD_ALPHA,
                    METHOD_WBOIT,
                    METHOD_LAYERED_WBOIT,
                    METHOD_CONTROL_GUIDED_ANISOTROPIC,
                    METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION,
                    METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                    METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                    METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                }) {
                    const std::string id = "pub-light-l" + std::to_string(lights);
                    const bool capture = lights == 256 && (method == METHOD_CLASSIC_FORWARD_ALPHA
                        || method == METHOD_WBOIT
                        || method == METHOD_LAYERED_WBOIT
                        || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
                    addQuality(id, "publication-lighting-quality", method, 1, 0.5f, 1, 0, 0.0f, lights, 2,
                        capture, capture && method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE,
                        false);
                }
            }

            // P3: Two-surface overlap is the primary contribution test. The proposed
            // decoder infers two local transparent layers and inverts the visibility
            // censoring of the back layer before deferred lighting.
            for (const int geometry : {3, 5}) {
                for (const float alpha : {0.5f, 0.75f, 0.875f}) {
                    // Compare translated Bayer ranks against surface-decorrelated static
                    // hash thresholds. The latter are expected to expose a broader set of
                    // back-layer thresholds after front-layer censoring.
                    for (const int dither : {3, 1}) {
                        for (const int method : {
                            METHOD_CLASSIC_FORWARD_ALPHA,
                            METHOD_SORTED_FORWARD_ALPHA,
                            METHOD_WBOIT,
                            METHOD_LAYERED_WBOIT,
                            METHOD_CONTROL_GUIDED_ANISOTROPIC,
                            METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION,
                            METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                            METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                            METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                        }) {
                            const std::string id = std::string("pub-overlap-") + TRANSPARENT_GEOMETRY_IDS[geometry]
                                + "-" + DITHER_IDS[dither] + "-identity-a" + scalarToken(alpha);
                            const bool capture = dither == 1 && std::abs(alpha - 0.75f) < 1e-6f
                                && (method == METHOD_CLASSIC_FORWARD_ALPHA
                                    || method == METHOD_SORTED_FORWARD_ALPHA
                                    || method == METHOD_WBOIT
                                    || method == METHOD_LAYERED_WBOIT
                                    || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
                            addQuality(id, "publication-overlap-quality", method, geometry, alpha, dither, 0, 0.0f, 64, 2,
                                capture, capture && method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE,
                                method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE);
                        }
                    }

                    // The user's alpha^2 idea is tested as an invertible high-alpha-only
                    // measurement transform. Low alpha is left unchanged; p>1 only opens
                    // additional holes above alpha=0.5 to expose the second layer.
                    for (const float exponent : {1.25f, 1.5f, 2.0f, 2.5f}) {
                        const std::string id = std::string("pub-overlap-") + TRANSPARENT_GEOMETRY_IDS[geometry]
                            + "-hash-power-p" + scalarToken(exponent) + "-a" + scalarToken(alpha);
                        addQuality(id, "publication-power-reveal", METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE,
                            geometry, alpha, 1, 5, exponent, 64, 2,
                            std::abs(exponent - 1.5f) < 1e-6f && std::abs(alpha - 0.875f) < 1e-6f,
                            false, true);
                    }
                }
            }

            // P4: Kernel ablation. A publishable result cannot depend on one cherry-picked
            // neighborhood size.
            for (const int kernel : {1, 2, 3}) {
                for (const int geometry : {1, 3, 5}) {
                    for (const float alpha : {0.25f, 0.75f}) {
                        const std::string id = std::string("pub-kernel-k") + std::to_string(kernel)
                            + "-" + TRANSPARENT_GEOMETRY_IDS[geometry] + "-a" + scalarToken(alpha);
                        addQuality(id, "publication-kernel-ablation", METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE,
                            geometry, alpha, 1, 0, 0.0f, 64, kernel, false, false, false);
                    }
                }
            }

            // P5: Out-of-scope stress test. The formal method reconstructs at most two
            // transparent layers; 4/8-layer stacks must be reported as a limitation,
            // not hidden from the paper.
            for (const int layers : {2, 4, 8}) {
                for (const float alpha : {0.5f, 0.75f}) {
                    for (const int method : {
                        METHOD_CLASSIC_FORWARD_ALPHA,
                        METHOD_SORTED_FORWARD_ALPHA,
                        METHOD_WBOIT,
                        METHOD_LAYERED_WBOIT,
                        METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                        METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                        METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                    }) {
                        BenchmarkCase test;
                        test.pairId = "pub-stack-layers" + std::to_string(layers) + "-a" + scalarToken(alpha);
                        test.sweep = "publication-layer-stack-stress";
                        test.method = method;
                        test.targetWidth = 1920;
                        test.targetHeight = 1080;
                        test.lightCount = 64;
                        test.alpha = alpha;
                        test.alphaMode = 0;
                        test.ditherMode = 1;
                        const bool sparseTwoLayer = method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                            || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                            || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                        test.coverageEncodingMode = sparseTwoLayer && alpha > 0.5f ? 5 : 0;
                        test.coverageBias = test.coverageEncodingMode == 5 ? 1.5f : 0.0f;
                        test.kernelRadius = 2;
                        test.transparentGeometry = 6;
                        test.transparentLayers = layers;
                        test.warmupFrames = qualityWarmup;
                        test.sampleFrames = qualitySamples;
                        test.quality = true;
                        test.capture = layers == 4 && std::abs(alpha - 0.75f) < 1e-6f;
                        test.recordOverdraw = method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                        test.recordDeferredLighting = sparseTwoLayer;
                        cases.push_back(test);
                    }
                }
            }

            // P6: Performance crossover. These are timing-only and repeated. With many
            // lights and high overdraw, deferred sparse-layer shading has a plausible
            // advantage because it shades at most two reconstructed layers per pixel.
            for (const int geometry : {3, 6}) {
                for (const int lights : {16, 64, 256}) {
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        for (const int method : {
                            METHOD_WBOIT,
                            METHOD_LAYERED_WBOIT,
                            METHOD_CONTROL_GUIDED_ANISOTROPIC,
                            METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION,
                            METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION,
                            METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC,
                            METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE
                        }) {
                            BenchmarkCase test;
                            test.pairId = std::string("pub-perf-") + TRANSPARENT_GEOMETRY_IDS[geometry]
                                + "-l" + std::to_string(lights) + "-r" + std::to_string(repeat);
                            test.sweep = "publication-performance";
                            test.method = method;
                            test.repeat = repeat;
                            test.targetWidth = 1920;
                            test.targetHeight = 1080;
                            test.lightCount = lights;
                            test.alpha = 0.75f;
                            test.alphaMode = 0;
                            test.ditherMode = 1;
                            const bool sparseTwoLayer = method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                                || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                                || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                            test.coverageEncodingMode = sparseTwoLayer ? 5 : 0;
                            test.coverageBias = test.coverageEncodingMode == 5 ? 1.5f : 0.0f;
                            test.kernelRadius = 2;
                            test.transparentGeometry = geometry;
                            test.transparentLayers = geometry == 6 ? 4 : 1;
                            test.warmupFrames = timingWarmup;
                            test.sampleFrames = timingSamples;
                            test.quality = false;
                            test.recordOverdraw = method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE && repeat == 0;
                            test.recordDeferredLighting = method == METHOD_CONTROL_GUIDED_ANISOTROPIC
                                || method == METHOD_DEFERRED_ATTRIBUTE_RECONSTRUCTION
                                || method == METHOD_DEFERRED_TWO_LAYER_RECONSTRUCTION
                                || method == METHOD_DEFERRED_TWO_LAYER_DIRECT_ISOTROPIC
                                || method == METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
                            cases.push_back(test);
                        }
                    }
                }
            }

            return cases;
        }

        LaunchOptions launchOptions_;
        std::filesystem::path root_;
        gfx::research::Shader opaqueShader_;
        gfx::research::Shader mixedOpaqueGBufferShader_;
        gfx::research::Shader mixedTransparentGBufferShader_;
        gfx::research::Shader mixedReconstructShader_;
        gfx::research::Shader attributeReconstructShader_;
        gfx::research::Shader attributeEdgeReconstructShader_;
        gfx::research::Shader attributeReferenceShader_;
        gfx::research::Shader attributeCompositeShader_;
        gfx::research::Shader twoLayerCompositeShader_;
        gfx::research::Shader twoLayerDirectCompositeShader_;
        gfx::research::Shader rawDitherShader_;
        gfx::research::Shader layeredSupportShader_;
        gfx::research::Shader deferredLightingShader_;
        gfx::research::Shader forwardShader_;
        gfx::research::Shader depthPeelShader_;
        gfx::research::Shader depthPeelAccumulateShader_;
        gfx::research::Shader depthPeelResolveShader_;
        gfx::research::Shader wboitShader_;
        gfx::research::Shader wboitResolveShader_;
        gfx::research::Shader layeredWboitShader_;
        gfx::research::Shader globalRevealShader_;
        gfx::research::Shader layeredResolveShader_;
        gfx::research::Shader alphaReferenceShader_;
        gfx::research::Shader overdrawShader_;
        gfx::research::Shader displayShader_;
        gfx::research::Framebuffer opaqueGBuffer_;
        gfx::research::Framebuffer opaqueLit_;
        gfx::research::Framebuffer layeredSupportTarget_;
        gfx::research::Framebuffer mixedGBuffer_;
        gfx::research::Framebuffer mixedLit_;
        gfx::research::Framebuffer reconstructedTransparentGBuffer_;
        gfx::research::Framebuffer reconstructedTransparentLit_;
        gfx::research::Framebuffer reconstructedTransparentLayer1GBuffer_;
        gfx::research::Framebuffer reconstructedTransparentLayer1Lit_;
        gfx::research::Framebuffer attributeReferenceFront_;
        gfx::research::Framebuffer attributeReferenceBack_;
        int attributeReferenceWidth_ = 0;
        int attributeReferenceHeight_ = 0;
        gfx::research::Framebuffer resultTarget_;
        gfx::research::Framebuffer referenceTarget_;
        gfx::research::Framebuffer depthPeelA_;
        gfx::research::Framebuffer depthPeelB_;
        gfx::research::Framebuffer depthPeelAccum_;
        gfx::research::Framebuffer alphaReferenceTarget_;
        unsigned int overdrawInputTexture_ = 0;
        unsigned int overdrawSurvivorTexture_ = 0;
        gfx::alpha::WboitTarget wboitTarget_;
        gfx::alpha::LayeredWboitTarget layeredWboitTarget_;
        gfx::research::FullscreenTriangle fullscreen_;
        gfx::research::GpuTimer gpuTimer_;
        gfx::research::GpuTimer deferredLightingTimer_;
        gfx::research::Mesh plane_;
        gfx::research::Mesh card_;
        gfx::research::Mesh sphere_;
        gfx::alpha::TexturedObjScene sponza_;
        gfx::research::Model dragon_;
        gfx::research::Model suzanne_;
        gfx::research::Model teapot_;
        gfx::research::Texture2D curtainTexture_;
        gfx::research::Texture2D foliageTexture_;
        gfx::research::OrbitCamera camera_;
        double cameraLastMouseX_ = 0.0;
        double cameraLastMouseY_ = 0.0;
        bool cameraMouseCaptured_ = false;
        bool cameraResetPressed_ = false;
        glm::vec3 transparentMotionOffset_ = glm::vec3(0.0f);

        int method_ = METHOD_DEFERRED_TWO_LAYER_EDGE_AWARE;
        int ditherMode_ = 3;
        int coverageEncodingMode_ = 0;
        float coverageBias_ = 0.0f;
        int alphaMode_ = 0;
        int kernelRadius_ = 2;
        int roddAblationMask_ = RODD_ABLATE_NONE;
        int lightCount_ = 12;
        int transparentGeometry_ = 4;
        int transparentLayers_ = 4;
        float alpha_ = 0.55f;
        float depthSigma_ = 0.35f;
        float albedoSigma_ = 0.12f;
        float detailStrength_ = 0.0f;
        float wboitDepthExponent_ = 2.0f;
        float layeredDepthSpan_ = 12.0f;
        bool showSponza_ = true;
        bool sponzaLoaded_ = false;
        bool dragonLoaded_ = false;
        bool suzanneLoaded_ = false;
        bool teapotLoaded_ = false;
        bool curtainTextureLoaded_ = false;
        bool foliageTextureLoaded_ = false;
        int renderWidth_ = 1600;
        int renderHeight_ = 900;

        std::vector<BenchmarkCase> benchmarkCases_;
        std::vector<double> benchmarkSamples_;
        std::vector<double> benchmarkDeferredLightingSamples_;
        std::filesystem::path benchmarkOutput_;
        std::ofstream benchmarkRaw_;
        std::ofstream benchmarkSummary_;
        std::ofstream benchmarkQuality_;
        std::ofstream benchmarkAttributeQuality_;
        std::ofstream benchmarkOverdraw_;
        std::ofstream benchmarkManifest_;
        std::size_t benchmarkCaseIndex_ = 0;
        int benchmarkWarmupRemaining_ = 0;
        int benchmarkSampleIndex_ = 0;
        bool benchmarkRunning_ = false;
        bool benchmarkCaseApplied_ = false;
        bool overdrawMeasuredThisCase_ = false;
        OverdrawMetrics currentOverdraw_{};
    };

    LaunchOptions parseLaunchOptions(const int argc, char** argv) {
        LaunchOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--benchmark-suite") options.benchmarkSuite = true;
            else if (argument == "--benchmark-quick") options.benchmarkQuick = true;
            else if (argument == "--benchmark-paper" || argument == "--benchmark-publication") options.benchmarkPaper = true;
            else if (argument == "--benchmark-edge-decision") options.benchmarkEdgeDecision = true;
            else if (argument.starts_with("--benchmark-output=")) {
                options.benchmarkOutput = std::string(argument.substr(std::string_view("--benchmark-output=").size()));
            }
        }
        return options;
    }
}

int main(const int argc, char** argv) {
    return AlphaResearch{parseLaunchOptions(argc, argv)}.run();
}
