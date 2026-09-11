#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
    constexpr std::array<int, 16> BAYER4 = {
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5
    };

    constexpr std::array<int, 16> RODD4 = {
         3, 11,  0,  8,
        13,  9, 12,  4,
         2, 10,  1, 15,
        14,  5,  6,  7
    };

    float threshold(const std::array<int, 16>& pattern, int x, int y, const unsigned seed) {
        const int sx = static_cast<int>(seed & 3u);
        const int sy = static_cast<int>((seed >> 2u) & 3u);
        x = (x + sx) & 3;
        y = (y + sy) & 3;
        return (static_cast<float>(pattern[static_cast<std::size_t>(y * 4 + x)]) + 0.5f) / 16.0f;
    }

    float decodedCoverage(const std::array<int, 16>& pattern, const float alpha, const unsigned seed) {
        int covered = 0;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) covered += alpha > threshold(pattern, x, y, seed) ? 1 : 0;
        }
        return static_cast<float>(covered) / 16.0f;
    }

    std::pair<float, float> ordinalInterval(const std::array<int, 16>& pattern, const float alpha, const unsigned seed) {
        float lower = 0.0f;
        float upper = 1.0f;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                const float tau = threshold(pattern, x, y, seed);
                if (alpha > tau) lower = std::max(lower, tau);
                else upper = std::min(upper, tau);
            }
        }
        return {lower, upper};
    }

    float normalizedGeometrySupport(const float occupancyDensity, const float reconstructedAlpha) {
        return std::clamp(occupancyDensity / std::max(reconstructedAlpha, 1.0f / 32.0f), 0.0f, 1.0f);
    }

    float compose(const float background, const float foreground, const float alpha) {
        return background * (1.0f - alpha) + foreground * alpha;
    }

    float determinant3(const std::array<float, 9>& m) {
        return
            m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
    }

    std::array<float, 3> solve3(std::array<float, 9> a, std::array<float, 3> b) {
        for (int c = 0; c < 3; ++c) {
            int pivot = c;
            for (int r = c + 1; r < 3; ++r) {
                if (std::abs(a[r * 3 + c]) > std::abs(a[pivot * 3 + c])) pivot = r;
            }
            assert(std::abs(a[pivot * 3 + c]) > 1e-8f);
            if (pivot != c) {
                for (int k = 0; k < 3; ++k) std::swap(a[c * 3 + k], a[pivot * 3 + k]);
                std::swap(b[c], b[pivot]);
            }
            const float inv = 1.0f / a[c * 3 + c];
            for (int k = c; k < 3; ++k) a[c * 3 + k] *= inv;
            b[c] *= inv;
            for (int r = 0; r < 3; ++r) {
                if (r == c) continue;
                const float f = a[r * 3 + c];
                for (int k = c; k < 3; ++k) a[r * 3 + k] -= f * a[c * 3 + k];
                b[r] -= f * b[c];
            }
        }
        return b;
    }

    void testOrderedRankIdentifiability(const std::array<int, 16>& pattern) {
        for (unsigned seed = 0; seed < 16; ++seed) {
            for (int rank = 3; rank < 16; ++rank) {
                std::array<float, 9> normal{};
                int samples = 0;
                for (int y = 0; y < 4; ++y) {
                    for (int x = 0; x < 4; ++x) {
                        const float tau = threshold(pattern, x, y, seed);
                        if (tau >= static_cast<float>(rank) / 16.0f) continue;
                        const std::array<float, 3> phi = {1.0f, static_cast<float>(x) - 1.5f, static_cast<float>(y) - 1.5f};
                        for (int r = 0; r < 3; ++r) {
                            for (int c = 0; c < 3; ++c) normal[r * 3 + c] += phi[r] * phi[c];
                        }
                        ++samples;
                    }
                }
                assert(samples == rank);
                assert(std::abs(determinant3(normal)) > 1e-5f);
            }
        }
    }


    int bayerRank(const int x, const int y, const int size) {
        const int levels = size == 16 ? 4 : (size == 8 ? 3 : 2);
        int weight = size * size / 4;
        int rank = 0;
        for (int bit = 0; bit < levels; ++bit) {
            const int xb = (x >> bit) & 1;
            const int yb = (y >> bit) & 1;
            const int code = yb == 0 ? (xb == 0 ? 0 : 2) : (xb == 0 ? 3 : 1);
            rank += code * weight;
            weight /= 4;
        }
        return rank;
    }

    float generalBayerThreshold(int x, int y, const int size, const unsigned seed) {
        const int levels = size == 16 ? 4 : (size == 8 ? 3 : 2);
        const int mask = size - 1;
        const int sx = static_cast<int>(seed) & mask;
        const int sy = static_cast<int>(seed >> static_cast<unsigned>(levels)) & mask;
        x = (x + sx) & mask;
        y = (y + sy) & mask;
        return (static_cast<float>(bayerRank(x, y, size)) + 0.5f) / static_cast<float>(size * size);
    }

    std::pair<float, float> generalBayerInterval(const float alpha, const int size, const unsigned seed) {
        float lower = 0.0f;
        float upper = 1.0f;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float tau = generalBayerThreshold(x, y, size, seed);
                if (alpha > tau) lower = std::max(lower, tau);
                else upper = std::min(upper, tau);
            }
        }
        return {lower, upper};
    }

    void testPhaseAnchoredBayer8Interval() {
        for (unsigned seed = 1; seed <= 64; ++seed) {
            for (const float alpha : {0.03f, 0.07f, 0.125f, 0.19f, 0.25f, 0.37f, 0.5f, 0.73f, 0.875f, 0.96f}) {
                const auto [lower, upper] = generalBayerInterval(alpha, 8, seed);
                assert(alpha > lower - 1e-6f);
                assert(alpha <= upper + 1e-6f);
                assert(upper - lower <= 1.0f / 64.0f + 1e-6f);

                int survivors = 0;
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        survivors += alpha > generalBayerThreshold(x, y, 8, seed) ? 1 : 0;
                    }
                }
                const int expected = std::clamp(static_cast<int>(std::floor(alpha * 64.0f + 0.5f)), 0, 64);
                assert(std::abs(survivors - expected) <= 1);
            }
        }
    }

    float logistic(const float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float logit(const float a) {
        const float x = std::clamp(a, 1e-5f, 1.0f - 1e-5f);
        return std::log(x / (1.0f - x));
    }

    float encodeAlpha(const float alpha, const float bias) {
        return logistic(logit(alpha) + bias);
    }

    float decodeThreshold(const float thresholdValue, const float bias) {
        return logistic(logit(thresholdValue) - bias);
    }

    float balancedGamma(const float compression) {
        return 1.0f / (1.0f + std::max(compression, 0.0f));
    }

    float encodeBalancedAlpha(const float alpha, const float compression) {
        return logistic(balancedGamma(compression) * logit(alpha));
    }

    float decodeBalancedThreshold(const float thresholdValue, const float compression) {
        return logistic(logit(thresholdValue) / balancedGamma(compression));
    }

    void testBalancedCompanding() {
        for (const float compression : {0.0f, 0.15f, 0.35f, 0.55f, 0.75f, 1.0f}) {
            const float gamma = balancedGamma(compression);
            assert(gamma > 0.0f && gamma <= 1.0f);
            assert(std::abs(encodeBalancedAlpha(0.5f, compression) - 0.5f) < 1e-6f);
            if (compression > 0.0f) {
                assert(encodeBalancedAlpha(0.125f, compression) > 0.125f);
                assert(encodeBalancedAlpha(0.875f, compression) < 0.875f);
            }

            for (const float alpha : {0.03f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.97f}) {
                for (const float thresholdValue : {0.01f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.99f}) {
                    const float encoded = encodeBalancedAlpha(alpha, compression);
                    const float decodedThreshold = decodeBalancedThreshold(thresholdValue, compression);
                    if (std::abs(encoded - thresholdValue) < 2e-6f || std::abs(alpha - decodedThreshold) < 2e-6f) continue;
                    assert((encoded > thresholdValue) == (alpha > decodedThreshold));
                }
            }
        }
    }

    void testGeneralBayerAndCompanding() {
        for (const int size : {4, 8, 16}) {
            std::vector<int> seen(static_cast<std::size_t>(size * size), 0);
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const int rank = bayerRank(x, y, size);
                    assert(rank >= 0 && rank < size * size);
                    ++seen[static_cast<std::size_t>(rank)];
                }
            }
            for (const int count : seen) assert(count == 1);
        }

        for (const float alpha : {0.03f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.97f}) {
            for (const float bias : {-1.65f, -1.15f, -0.65f, 0.0f, 0.65f, 1.15f, 1.65f}) {
                for (const float thresholdValue : {0.01f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.99f}) {
                    const float encoded = encodeAlpha(alpha, bias);
                    const float decodedThreshold = decodeThreshold(thresholdValue, bias);
                    if (std::abs(encoded - thresholdValue) < 2e-6f || std::abs(alpha - decodedThreshold) < 2e-6f) continue;
                    const bool encodedDecision = encoded > thresholdValue;
                    const bool decodedDecision = alpha > decodedThreshold;
                    assert(encodedDecision == decodedDecision);
                }
            }
        }
    }


    int complementaryBayerRank(const int x, const int y, const int size) {
        const int base = bayerRank(x, y, size);
        const int count = size * size;
        const int parity = (x + y) & 1;
        if (parity == 0) {
            assert(base < count / 2);
            return 2 * base;
        }
        assert(base >= count / 2);
        return 2 * (base - count / 2) + 1;
    }

    void testComplementaryBayerCompanding() {
        constexpr float bias = 1.15f;
        for (const int size : {4, 8, 16}) {
            std::vector<int> seen(static_cast<std::size_t>(size * size), 0);
            std::vector<float> effectiveThresholds;
            effectiveThresholds.reserve(static_cast<std::size_t>(size * size));

            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    const int rank = complementaryBayerRank(x, y, size);
                    assert(rank >= 0 && rank < size * size);
                    assert((rank & 1) == ((x + y) & 1));
                    ++seen[static_cast<std::size_t>(rank)];

                    const float thresholdValue = (static_cast<float>(rank) + 0.5f) / static_cast<float>(size * size);
                    const float signedBias = ((rank & 1) == 0) ? bias : -bias;
                    effectiveThresholds.push_back(decodeThreshold(thresholdValue, signedBias));
                }
            }
            for (const int count : seen) assert(count == 1);

            std::sort(effectiveThresholds.begin(), effectiveThresholds.end());
            float maxGap = effectiveThresholds.front();
            for (std::size_t i = 1; i < effectiveThresholds.size(); ++i) {
                maxGap = std::max(maxGap, effectiveThresholds[i] - effectiveThresholds[i - 1]);
            }
            maxGap = std::max(maxGap, 1.0f - effectiveThresholds.back());

            // The co-designed even/odd rank permutation avoids the >0.5 alpha hole
            // produced by applying checkerboard +/- bias directly to standard Bayer.
            const float limit = size == 4 ? 0.13f : (size == 8 ? 0.05f : 0.015f);
            assert(maxGap < limit);
        }
    }

    float directionalPairInterpolate(const float negativeValue, const int negativeDistance, const float positiveValue, const int positiveDistance) {
        const float totalDistance = static_cast<float>(negativeDistance + positiveDistance);
        return (
            negativeValue * static_cast<float>(positiveDistance)
            + positiveValue * static_cast<float>(negativeDistance)
        ) / totalDistance;
    }

    void testDirectionalAffineReproduction() {
        for (const float intercept : {0.1f, 0.7f, 2.0f}) {
            for (const float slope : {-0.23f, 0.0f, 0.41f}) {
                for (const int negativeDistance : {1, 2, 3}) {
                    for (const int positiveDistance : {1, 2, 3}) {
                        const float negativeValue = intercept - slope * static_cast<float>(negativeDistance);
                        const float positiveValue = intercept + slope * static_cast<float>(positiveDistance);
                        const float reconstructed = directionalPairInterpolate(
                            negativeValue, negativeDistance, positiveValue, positiveDistance
                        );
                        assert(std::abs(reconstructed - intercept) < 1e-6f);
                    }
                }
            }
        }
    }

    void testAffineReproduction() {
        constexpr std::array<std::array<float, 2>, 8> points = {{
            {{-2.0f, -1.0f}}, {{-1.0f, 1.0f}}, {{0.0f, -2.0f}}, {{0.0f, 0.0f}},
            {{1.0f, 2.0f}}, {{2.0f, -1.0f}}, {{-2.0f, 2.0f}}, {{2.0f, 1.0f}}
        }};
        std::array<float, 9> m{};
        std::array<float, 3> rhs{};
        for (const auto& p : points) {
            const float x = p[0];
            const float y = p[1];
            const float value = 0.30f + 0.11f * x - 0.07f * y;
            const std::array<float, 3> phi = {1.0f, x, y};
            const float weight = 0.4f + 1.0f / (1.0f + x * x + y * y);
            for (int r = 0; r < 3; ++r) {
                rhs[r] += weight * phi[r] * value;
                for (int c = 0; c < 3; ++c) m[r * 3 + c] += weight * phi[r] * phi[c];
            }
        }
        assert(std::abs(determinant3(m)) > 1e-5f);
        const auto beta = solve3(m, rhs);
        assert(std::abs(beta[0] - 0.30f) < 1e-5f);
        assert(std::abs(beta[1] - 0.11f) < 1e-5f);
        assert(std::abs(beta[2] + 0.07f) < 1e-5f);
    }

    float encodeHighAlphaPower(const float alpha, const float exponent) {
        constexpr float pivot = 0.5f;
        const float p = std::max(exponent, 1.0f);
        const float a = std::clamp(alpha, 0.0f, 1.0f);
        if (a <= pivot || p <= 1.0001f) return a;
        const float t = (a - pivot) / (1.0f - pivot);
        return pivot + (1.0f - pivot) * std::pow(t, p);
    }

    float decodeHighAlphaPower(const float beta, const float exponent) {
        constexpr float pivot = 0.5f;
        const float p = std::max(exponent, 1.0f);
        const float b = std::clamp(beta, 0.0f, 1.0f);
        if (b <= pivot || p <= 1.0001f) return b;
        const float t = (b - pivot) / (1.0f - pivot);
        return pivot + (1.0f - pivot) * std::pow(t, 1.0f / p);
    }

    float stableHashThreshold(const int x, const int y, const unsigned seed) {
        unsigned qx = static_cast<unsigned>(x) ^ seed;
        unsigned qy = static_cast<unsigned>(y) ^ (seed * 0x9E3779B9u);
        unsigned h = qx * 0x9E3779B9u ^ qy * 0x85EBCA6Bu;
        h ^= h >> 16u;
        h *= 0x7FEB352Du;
        h ^= h >> 15u;
        h *= 0x846CA68Bu;
        h ^= h >> 16u;
        return static_cast<float>(h & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    }

    struct TwoLayerEstimate {
        float frontAlpha = 0.0f;
        float backAlpha = 0.0f;
        int frontSamples = 0;
        int backSamples = 0;
        int backgroundSamples = 0;
    };

    template<typename ThresholdFn>
    TwoLayerEstimate estimateTwoLayers(
        const float frontAlpha,
        const float backAlpha,
        const float exponent,
        const unsigned frontSeed,
        const unsigned backSeed,
        ThresholdFn&& thresholdFn
    ) {
        const float frontBeta = encodeHighAlphaPower(frontAlpha, exponent);
        const float backBeta = encodeHighAlphaPower(backAlpha, exponent);
        float frontLower = 0.0f;
        float frontUpper = 1.0f;
        float backLower = 0.0f;
        float backUpper = 1.0f;
        TwoLayerEstimate result;

        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                const float frontThreshold = thresholdFn(x, y, frontSeed);
                const float backThreshold = thresholdFn(x, y, backSeed);
                if (frontBeta > frontThreshold) {
                    ++result.frontSamples;
                    frontLower = std::max(frontLower, frontThreshold);
                } else if (backBeta > backThreshold) {
                    ++result.backSamples;
                    frontUpper = std::min(frontUpper, frontThreshold);
                    backLower = std::max(backLower, backThreshold);
                } else {
                    ++result.backgroundSamples;
                    frontUpper = std::min(frontUpper, frontThreshold);
                    backUpper = std::min(backUpper, backThreshold);
                }
            }
        }

        const float frontEstimate = 0.5f * (frontLower + frontUpper);
        const float backEstimate = 0.5f * (backLower + backUpper);
        result.frontAlpha = decodeHighAlphaPower(frontEstimate, exponent);
        result.backAlpha = decodeHighAlphaPower(backEstimate, exponent);
        return result;
    }

    void testHighAlphaPowerCompanding() {
        for (const float exponent : {1.0f, 1.25f, 1.5f, 2.0f, 2.5f}) {
            for (const float alpha : {0.05f, 0.25f, 0.5f, 0.625f, 0.75f, 0.875f, 0.97f}) {
                const float beta = encodeHighAlphaPower(alpha, exponent);
                const float decoded = decodeHighAlphaPower(beta, exponent);
                assert(std::abs(decoded - alpha) < 2e-6f);
                if (alpha <= 0.5f) assert(std::abs(beta - alpha) < 1e-6f);
                if (alpha > 0.5f && exponent > 1.0f) assert(beta <= alpha + 1e-6f);
            }
        }
    }

    void testCensoredTwoLayerOrdinalInversion() {
        auto hashThreshold = [](const int x, const int y, const unsigned seed) {
            return stableHashThreshold(x, y, seed);
        };
        auto bayerThreshold8 = [](const int x, const int y, const unsigned seed) {
            return generalBayerThreshold(x, y, 8, seed);
        };

        double hashBackError = 0.0;
        double bayerBackError = 0.0;
        double revealedBackSamplesIdentity = 0.0;
        double revealedBackSamplesPower = 0.0;
        int samples = 0;

        for (const float frontAlpha : {0.5f, 0.625f, 0.75f, 0.875f}) {
            for (const float backAlpha : {0.25f, 0.5f, 0.75f, 0.875f}) {
                for (unsigned frontSeed = 1; frontSeed <= 8; ++frontSeed) {
                    for (unsigned backSeed = 17; backSeed <= 24; ++backSeed) {
                        const TwoLayerEstimate hashEstimate = estimateTwoLayers(
                            frontAlpha, backAlpha, 1.5f, frontSeed, backSeed, hashThreshold
                        );
                        const TwoLayerEstimate bayerEstimate = estimateTwoLayers(
                            frontAlpha, backAlpha, 1.5f, frontSeed, backSeed, bayerThreshold8
                        );
                        hashBackError += std::abs(hashEstimate.backAlpha - backAlpha);
                        bayerBackError += std::abs(bayerEstimate.backAlpha - backAlpha);

                        if (frontAlpha >= 0.75f) {
                            const TwoLayerEstimate identity = estimateTwoLayers(
                                frontAlpha, backAlpha, 1.0f, frontSeed, backSeed, hashThreshold
                            );
                            const TwoLayerEstimate power = estimateTwoLayers(
                                frontAlpha, backAlpha, 1.5f, frontSeed, backSeed, hashThreshold
                            );
                            revealedBackSamplesIdentity += identity.backSamples;
                            revealedBackSamplesPower += power.backSamples;
                        }
                        ++samples;
                    }
                }
            }
        }

        hashBackError /= static_cast<double>(samples);
        bayerBackError /= static_cast<double>(samples);
        // The core publication hypothesis: decorrelated thresholds provide materially
        // better censored back-layer inversion than translated Bayer ranks.
        assert(hashBackError < 0.06);
        assert(hashBackError < bayerBackError * 0.60);
        // High-alpha-only power companding must expose more back-layer observations on
        // average without changing alpha <= 0.5.
        assert(revealedBackSamplesPower > revealedBackSamplesIdentity * 1.10);
    }

    void testFrontToBackPeelAccumulation() {
        struct Rgba { float r, g, b, a; };
        auto frontToBack = [](Rgba dst, Rgba src) {
            const float transmit = 1.0f - dst.a;
            dst.r += transmit * src.r * src.a;
            dst.g += transmit * src.g * src.a;
            dst.b += transmit * src.b * src.a;
            dst.a += transmit * src.a;
            return dst;
        };
        auto over = [](Rgba front, Rgba back) {
            return Rgba{
                front.r * front.a + back.r * back.a * (1.0f - front.a),
                front.g * front.a + back.g * back.a * (1.0f - front.a),
                front.b * front.a + back.b * back.a * (1.0f - front.a),
                front.a + back.a * (1.0f - front.a)
            };
        };

        const Rgba front{1.0f, 0.2f, 0.1f, 0.35f};
        const Rgba back{0.1f, 0.5f, 1.0f, 0.60f};
        Rgba accum{0.0f, 0.0f, 0.0f, 0.0f};
        accum = frontToBack(accum, front);
        accum = frontToBack(accum, back);

        const Rgba exact = over(front, back);
        assert(std::abs(accum.r - exact.r) < 1e-6f);
        assert(std::abs(accum.g - exact.g) < 1e-6f);
        assert(std::abs(accum.b - exact.b) < 1e-6f);
        assert(std::abs(accum.a - exact.a) < 1e-6f);
    }

}

int main() {
    for (const auto& pattern : {BAYER4, RODD4}) {
        for (unsigned seed = 0; seed < 16; ++seed) {
            assert(decodedCoverage(pattern, 0.0f, seed) == 0.0f);
            assert(decodedCoverage(pattern, 1.0f, seed) == 1.0f);
            assert(std::abs(decodedCoverage(pattern, 0.25f, seed) - 0.25f) < 1e-6f);
            assert(std::abs(decodedCoverage(pattern, 0.50f, seed) - 0.50f) < 1e-6f);
            assert(std::abs(decodedCoverage(pattern, 0.75f, seed) - 0.75f) < 1e-6f);

            for (const float alpha : {0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f}) {
                const auto [lower, upper] = ordinalInterval(pattern, alpha, seed);
                assert(alpha > lower - 1e-6f);
                assert(alpha <= upper + 1e-6f);
                const float midpoint = 0.5f * (lower + upper);
                assert(std::abs(midpoint - alpha) <= 1.0f / 32.0f + 1e-6f);
            }
        }
    }

    testOrderedRankIdentifiability(BAYER4);
    testOrderedRankIdentifiability(RODD4);
    testGeneralBayerAndCompanding();
    testComplementaryBayerCompanding();
    testBalancedCompanding();
    testPhaseAnchoredBayer8Interval();

    assert(std::abs(compose(0.2f, 1.0f, 0.5f) - 0.6f) < 1e-6f);
    assert(std::abs(normalizedGeometrySupport(0.25f, 0.25f) - 1.0f) < 1e-6f);
    assert(std::abs(normalizedGeometrySupport(0.25f, 0.50f) - 0.50f) < 1e-6f);
    testAffineReproduction();
    testDirectionalAffineReproduction();
    testHighAlphaPowerCompanding();
    testCensoredTwoLayerOrdinalInversion();
    testFrontToBackPeelAccumulation();

    std::cout << "Reference tests passed: ordered coverage, attribute reconstruction math, high-alpha power companding, censored two-layer ordinal inversion, and depth-peel composition\n";
    return 0;
}
