#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAlphaMeta;

layout(binding = 0) uniform sampler2D uLit;
layout(binding = 1) uniform sampler2D uAlbedo;
layout(binding = 2) uniform sampler2D uNormal;
layout(binding = 3) uniform sampler2D uPosition;
layout(binding = 4) uniform sampler2D uControl;

uniform int uMethod;
uniform int uKernelRadius;
uniform int uDitherMode;
uniform int uCoverageEncodingMode;
uniform float uCoverageBias;
uniform int uRoddAblationMask;
uniform float uDepthSigma;
uniform float uAlbedoSigma;
uniform float uDetailStrength;

const float EPS = 1e-6;
const int RODD_ABLATE_ORDINAL = 1 << 0;
const int RODD_ABLATE_CENSORED_NEGATIVES = 1 << 1;
const int RODD_ABLATE_GEOMETRY = 1 << 2;
const int RODD_ABLATE_RANK_CONDITIONING = 1 << 3;
const int RODD_ABLATE_MOMENTS = 1 << 4;
const int RODD_ABLATE_DIRECTIONAL_PAIRS = 1 << 5;
const int RODD_ABLATE_CONFIDENCE_GUARD = 1 << 6;

bool roddAblated(int bit) {
    return (uRoddAblationMask & bit) != 0;
}

float edgeSideGate(vec2 d, vec2 edgeNormal, float edgeStrength, bool foreground);
float anisotropicSpatial(vec2 d, float sigma, vec2 edgeNormal, float edgeStrength, bool anisotropic);
float guideWeight(ivec2 p, vec3 refAlbedo, vec3 refNormal, vec3 refPosition);

ivec2 clampPixel(ivec2 p, ivec2 size) {
    return clamp(p, ivec2(0), size - 1);
}

vec4 controlAt(ivec2 p) {
    return texelFetch(uControl, p, 0);
}

int surfaceKey(ivec2 p) {
    return int(floor(controlAt(p).w + 0.5));
}

vec3 litAt(ivec2 p) {
    return texelFetch(uLit, p, 0).rgb;
}

vec3 albedoAt(ivec2 p) {
    return texelFetch(uAlbedo, p, 0).rgb;
}

vec3 normalAt(ivec2 p) {
    return normalize(texelFetch(uNormal, p, 0).rgb * 2.0 - 1.0);
}

vec3 positionAt(ivec2 p) {
    return texelFetch(uPosition, p, 0).xyz;
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float gaussianSpatial(vec2 d, float sigma) {
    return exp(-0.5 * dot(d, d) / max(sigma * sigma, 1e-4));
}

int patternSizeForMode(int mode) {
    if (mode == 3) return 8;
    if (mode == 4) return 16;
    return 4;
}

int patternLevelsForSize(int size) {
    if (size == 16) return 4;
    if (size == 8) return 3;
    return 2;
}

int bayerRank(ivec2 phase, int size) {
    int levels = patternLevelsForSize(size);
    int weight = (size * size) / 4;
    int rank = 0;
    for (int bit = 0; bit < 4; ++bit) {
        if (bit >= levels) break;
        int xb = (phase.x >> bit) & 1;
        int yb = (phase.y >> bit) & 1;
        int code = yb == 0 ? (xb == 0 ? 0 : 2) : (xb == 0 ? 3 : 1);
        rank += code * weight;
        weight /= 4;
    }
    return rank;
}

int complementaryBayerRank(ivec2 phase, int size, int baseRank) {
    // Standard Bayer places one checkerboard parity in the lower half of the
    // ranks and the other parity in the upper half. Applying +/- companding
    // directly to that layout leaves a very large hole in effective alpha
    // thresholds. Reindex both halves as even/odd ranks instead. The spatial
    // +/- bias remains checkerboard-local while physical-alpha thresholds are
    // interleaved across the complete range.
    int count = size * size;
    int parity = (phase.x + phase.y) & 1;
    if (parity == 0) return 2 * baseRank;
    return 2 * (baseRank - count / 2) + 1;
}

float orderedThreshold(ivec2 p, uint seed, int mode) {
    const float rodd[16] = float[16](
         3.0, 11.0,  0.0,  8.0,
        13.0,  9.0, 12.0,  4.0,
         2.0, 10.0,  1.0, 15.0,
        14.0,  5.0,  6.0,  7.0
    );
    int size = patternSizeForMode(mode);
    int levels = patternLevelsForSize(size);
    int mask = size - 1;
    int sx = int(seed) & mask;
    int sy = int(seed >> uint(levels)) & mask;
    ivec2 phase = (p + ivec2(sx, sy)) & ivec2(mask);
    int baseRank = mode == 2 ? int(rodd[phase.y * 4 + phase.x]) : bayerRank(phase, size);
    if (uCoverageEncodingMode == 3 && mode != 2) {
        baseRank = complementaryBayerRank(phase, size, baseRank);
    }
    return (float(baseRank) + 0.5) / float(size * size);
}

float stableHashThreshold(ivec2 p, uint seed) {
    uvec2 q = uvec2(p) ^ uvec2(seed, seed * 0x9E3779B9u);
    uint h = q.x * 0x9E3779B9u ^ q.y * 0x85EBCA6Bu;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return float(h & 0x00FFFFFFu) / float(0x01000000u);
}

float safeLogit(float a) {
    float x = clamp(a, 1e-5, 1.0 - 1e-5);
    return log(x / (1.0 - x));
}

float logistic(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float coverageBiasForPixel(ivec2 p, int key) {
    if (uCoverageEncodingMode == 0 || uCoverageEncodingMode == 4 || uCoverageEncodingMode == 5 || uCoverageBias <= 0.0) return 0.0;
    if (uCoverageEncodingMode == 1) return uCoverageBias;
    if (uCoverageEncodingMode == 2) return -uCoverageBias;

    uint seed = uint(max(key, 1));
    if (uDitherMode != 1) {
        int size = patternSizeForMode(uDitherMode);
        int rank = clamp(int(floor(orderedThreshold(p, seed, uDitherMode) * float(size * size))), 0, size * size - 1);
        return (rank & 1) == 0 ? uCoverageBias : -uCoverageBias;
    }
    uint h = uint(p.x) * 0x9E3779B9u ^ uint(p.y) * 0x85EBCA6Bu ^ (seed * 0xC2B2AE35u);
    h ^= h >> 16;
    return (h & 1u) == 0u ? uCoverageBias : -uCoverageBias;
}

float balancedLogitScale() {
    return 1.0 / (1.0 + max(uCoverageBias, 0.0));
}

float decodeHighAlphaPowerThreshold(float threshold) {
    const float pivot = 0.5;
    float p = max(uCoverageBias, 1.0);
    float t = clamp(threshold, 0.0, 1.0);
    if (t <= pivot || p <= 1.0001) return t;
    float u = clamp((t - pivot) / (1.0 - pivot), 0.0, 1.0);
    return pivot + (1.0 - pivot) * pow(u, 1.0 / p);
}

float decodeCoverageThreshold(float threshold, float bias) {
    if (uCoverageEncodingMode == 4) {
        // Encoder: beta = sigmoid(gamma * logit(alpha)). Therefore
        // beta > tau is exactly alpha > sigmoid(logit(tau) / gamma).
        float gamma = balancedLogitScale();
        return logistic(safeLogit(threshold) / max(gamma, 1e-4));
    }
    if (uCoverageEncodingMode == 5) return decodeHighAlphaPowerThreshold(threshold);
    if (abs(bias) <= 1e-6) return threshold;
    return logistic(safeLogit(threshold) - bias);
}

float rawThresholdForSurface(ivec2 p, int key) {
    uint seed = uint(max(key, 1));
    return uDitherMode != 1 ? orderedThreshold(p, seed, uDitherMode) : stableHashThreshold(p, seed);
}

float thresholdForSurface(ivec2 p, int key) {
    float rawThreshold = rawThresholdForSurface(p, key);
    return decodeCoverageThreshold(rawThreshold, coverageBiasForPixel(p, key));
}

float orderedQuantStep() {
    if (uDitherMode == 1) return 1.0 / 64.0;
    int size = patternSizeForMode(uDitherMode);
    return 1.0 / float(size * size);
}

ivec2 phaseFootprintOrigin(ivec2 center, ivec2 size) {
    // Keep an actual contiguous 4x4 footprint even at framebuffer borders.
    // Per-sample clamping would duplicate phases and break the complete-rank property.
    ivec2 maxOrigin = max(size - ivec2(4), ivec2(0));
    return clamp(center - ivec2(1), ivec2(0), maxOrigin);
}

int phaseCompleteRank(ivec2 center, ivec2 size, int key) {
    ivec2 origin = phaseFootprintOrigin(center, size);
    int rank = 0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            rank += surfaceKey(origin + ivec2(x, y)) == key ? 1 : 0;
        }
    }
    return rank;
}

int nearestTransparentKey(ivec2 center, ivec2 size, int radius) {
    int centerKey = surfaceKey(center);
    if (centerKey > 0) return centerKey;

    float best = 1e30;
    int key = 0;
    for (int y = -5; y <= 5; ++y) {
        for (int x = -5; x <= 5; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            int k = surfaceKey(p);
            if (k <= 0) continue;
            float score = float(x * x + y * y);
            if (score < best) {
                best = score;
                key = k;
            }
        }
    }
    return key;
}

bool findReference(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    bool foreground,
    out vec3 refAlbedo,
    out vec3 refNormal,
    out vec3 refPosition
) {
    float best = 1e30;
    bool found = false;

    for (int y = -5; y <= 5; ++y) {
        for (int x = -5; x <= 5; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            int sampleKey = surfaceKey(p);
            if (foreground && sampleKey != key) continue;
            // For the proposed single-dominant-surface model, exposed background
            // is the untouched opaque G-buffer (surface key 0). Mixing samples
            // from a different transparent surface creates the broad dark/color
            // shadows seen at overlaps and violates the F/B decomposition.
            if (!foreground && sampleKey != 0) continue;

            float score = float(x * x + y * y);
            if (score >= best) continue;
            best = score;
            refAlbedo = albedoAt(p);
            refNormal = normalAt(p);
            refPosition = positionAt(p);
            found = true;
        }
    }
    return found;
}

bool nearestSurfaceColor(ivec2 center, ivec2 size, int key, int radius, out vec3 color) {
    float best = 1e30;
    bool found = false;
    color = litAt(center);
    for (int y = -5; y <= 5; ++y) {
        for (int x = -5; x <= 5; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            if (surfaceKey(p) != key) continue;
            float score = float(x * x + y * y);
            if (score >= best) continue;
            best = score;
            color = litAt(p);
            found = true;
        }
    }
    return found;
}

// A contiguous 4x4 footprint contains every Bayer phase exactly once.
// Using this footprint prevents the 4x4 pattern itself from becoming a
// low-frequency coverage ripple after reconstruction.
float phaseCompleteMembership(ivec2 center, ivec2 size, int key) {
    ivec2 origin = phaseFootprintOrigin(center, size);
    float sum = 0.0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            sum += surfaceKey(origin + ivec2(x, y)) == key ? 1.0 : 0.0;
        }
    }
    return sum * (1.0 / 16.0);
}

float smoothedMembership(ivec2 center, ivec2 size, int key) {
    // A 4x4 footprint is phase-complete only for 4x4 ordered patterns. Using it
    // for Bayer 8x8/16x16 creates false coverage gradients that look like edges.
    // Larger patterns therefore use the same local 7x7 smoothing footprint as
    // the hash baseline for geometry/edge inference, while ordinal thresholds
    // remain exact and pattern-aware.
    if (uDitherMode != 1 && patternSizeForMode(uDitherMode) == 4) {
        return phaseCompleteMembership(center, size, key);
    }

    float sum = 0.0;
    float weight = 0.0;
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            float w = gaussianSpatial(vec2(x, y), 1.8);
            sum += (surfaceKey(p) == key ? 1.0 : 0.0) * w;
            weight += w;
        }
    }
    return weight > EPS ? sum / weight : 0.0;
}

float localCoverageMean(ivec2 center, ivec2 size, int key, vec2 edgeNormal, float edgeStrength, int radius) {
    float sum = 0.0;
    float weight = 0.0;
    int r = clamp(radius, 1, 3);
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            if (abs(x) > r || abs(y) > r) continue;
            vec2 d = vec2(x, y);
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            float w = gaussianSpatial(d, 1.35);
            w *= edgeSideGate(d, edgeNormal, edgeStrength, true);
            sum += (surfaceKey(p) == key ? 1.0 : 0.0) * w;
            weight += w;
        }
    }
    return weight > EPS ? sum / weight : smoothedMembership(center, size, key);
}

vec3 localSeparatedMean(
    ivec2 center,
    ivec2 size,
    int key,
    bool foreground,
    vec3 refAlbedo,
    vec3 refNormal,
    vec3 refPosition,
    vec2 edgeNormal,
    float edgeStrength,
    int radius,
    vec3 fallbackValue
) {
    vec3 sum = vec3(0.0);
    float weight = 0.0;
    int r = clamp(radius, 1, 3);
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            if (abs(x) > r || abs(y) > r) continue;
            vec2 d = vec2(x, y);
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            bool sameSurface = surfaceKey(p) == key;
            if (foreground != sameSurface) continue;
            float w = guideWeight(p, refAlbedo, refNormal, refPosition);
            w *= anisotropicSpatial(d, 1.25, edgeNormal, edgeStrength, true);
            w *= edgeSideGate(d, edgeNormal, edgeStrength, foreground);
            sum += litAt(p) * w;
            weight += w;
        }
    }
    return weight > EPS ? sum / weight : fallbackValue;
}

// Return the screen-space origin of the ordered-pattern cell containing center.
// Unlike a center-relative kernel this cell is phase anchored, so a constant-alpha
// surface produces the same survivor count for every target pixel in the cell.
bool orderedCellOrigin(ivec2 center, ivec2 framebufferSize, int key, out ivec2 origin, out int patternSize) {
    if (uDitherMode == 1) return false;
    patternSize = patternSizeForMode(uDitherMode);
    int levels = patternLevelsForSize(patternSize);
    int mask = patternSize - 1;
    uint seed = uint(max(key, 1));
    ivec2 shift = ivec2(int(seed) & mask, int(seed >> uint(levels)) & mask);
    ivec2 phase = (center + shift) & ivec2(mask);
    origin = center - phase;
    return all(greaterThanEqual(origin, ivec2(0)))
        && all(lessThanEqual(origin + ivec2(patternSize), framebufferSize));
}

// Stable normalized coverage rank used only to choose reconstruction support.
// For Bayer 8x8 the full phase-anchored cell is cheap enough (64 taps) and removes
// the target-phase modulation that previously leaked the ordered pattern into the
// reconstruction weights. 16x16 remains experimental and uses a local estimate.
int stableCoverageRank16(ivec2 center, ivec2 framebufferSize, int key) {
    if (uDitherMode != 1 && patternSizeForMode(uDitherMode) == 4) {
        return phaseCompleteRank(center, framebufferSize, key);
    }

    if (uDitherMode != 1 && patternSizeForMode(uDitherMode) == 8) {
        ivec2 origin;
        int patternSize;
        if (orderedCellOrigin(center, framebufferSize, key, origin, patternSize)) {
            int survivors = 0;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    survivors += surfaceKey(origin + ivec2(x, y)) == key ? 1 : 0;
                }
            }
            return clamp(int(round(float(survivors) * 16.0 / 64.0)), 0, 16);
        }
    }

    return clamp(int(round(smoothedMembership(center, framebufferSize, key) * 16.0)), 0, 16);
}

float edgeSideGate(vec2 d, vec2 edgeNormal, float edgeStrength, bool foreground) {
    float hardEdge = smoothstep(0.10, 0.40, edgeStrength);
    if (hardEdge <= 1e-4) return 1.0;
    float side = dot(d, edgeNormal) * (foreground ? 1.0 : -1.0);
    float sameSide = smoothstep(-0.20, 0.08, side);
    return mix(1.0, sameSide, hardEdge);
}

void coverageGradient(
    ivec2 center,
    ivec2 size,
    int key,
    out vec2 edgeNormal,
    out float edgeStrength
) {
    float left = smoothedMembership(clampPixel(center + ivec2(-2, 0), size), size, key);
    float right = smoothedMembership(clampPixel(center + ivec2(2, 0), size), size, key);
    float down = smoothedMembership(clampPixel(center + ivec2(0, -2), size), size, key);
    float up = smoothedMembership(clampPixel(center + ivec2(0, 2), size), size, key);

    vec2 g = vec2(right - left, up - down);
    float magnitude = length(g);
    edgeStrength = clamp(magnitude * 1.8, 0.0, 1.0);
    edgeNormal = magnitude > 1e-5 ? g / magnitude : vec2(1.0, 0.0);
}

float anisotropicSpatial(
    vec2 d,
    float sigma,
    vec2 edgeNormal,
    float edgeStrength,
    bool anisotropic
) {
    if (!anisotropic || edgeStrength < 1e-4) return gaussianSpatial(d, sigma);

    vec2 tangent = vec2(-edgeNormal.y, edgeNormal.x);
    float across = dot(d, edgeNormal);
    float along = dot(d, tangent);

    float sigmaAcross = mix(sigma, max(0.40, sigma * 0.22), edgeStrength);
    float sigmaAlong = mix(sigma, sigma * 1.28, edgeStrength);
    float q =
        (across * across) / max(sigmaAcross * sigmaAcross, 1e-4)
        + (along * along) / max(sigmaAlong * sigmaAlong, 1e-4);
    return exp(-0.5 * q);
}

float guideWeight(
    ivec2 p,
    vec3 refAlbedo,
    vec3 refNormal,
    vec3 refPosition
) {
    vec3 samplePosition = positionAt(p);
    vec3 sampleNormal = normalAt(p);
    vec3 sampleAlbedo = albedoAt(p);

    vec3 delta = samplePosition - refPosition;
    float planeDistance = abs(dot(delta, refNormal));
    float depthW = exp(-planeDistance / max(uDepthSigma, 1e-4));

    float nd = max(dot(refNormal, sampleNormal), 0.0);
    float normalW = max(pow(nd, 12.0), 0.001);

    vec3 da = sampleAlbedo - refAlbedo;
    float sigma2 = max(uAlbedoSigma * uAlbedoSigma, 1e-5);
    float albedoW = 0.10 + 0.90 * exp(-dot(da, da) / (2.0 * sigma2));

    // Radiance reconstruction is intentionally phase-neutral in v1.8. Thresholds
    // belong to the alpha inverse problem; using them again as color weights leaked
    // the Bayer phase into F_hat and produced the visible dot pattern in v1.7.
    return depthW * normalW * albedoW;
}

void addRegressionSample(
    vec2 d,
    vec3 value,
    float weight,
    inout mat3 normalMatrix,
    inout vec3 rhsR,
    inout vec3 rhsG,
    inout vec3 rhsB
) {
    vec3 phi = vec3(1.0, d.x, d.y);
    normalMatrix += weight * outerProduct(phi, phi);
    rhsR += weight * phi * value.r;
    rhsG += weight * phi * value.g;
    rhsB += weight * phi * value.b;
}

bool solveRegression(
    mat3 normalMatrix,
    vec3 rhsR,
    vec3 rhsG,
    vec3 rhsB,
    out vec3 betaR,
    out vec3 betaG,
    out vec3 betaB
) {
    // Full-rank weighted affine least squares exactly reproduces affine fields.
    // Do not bias that case with unconditional ridge regularization.
    float detM = determinant(normalMatrix);
    if (abs(detM) >= 1e-7) {
        mat3 invM = inverse(normalMatrix);
        betaR = invM * rhsR;
        betaG = invM * rhsG;
        betaB = invM * rhsB;
        return true;
    }

    normalMatrix[0][0] += 2e-4;
    normalMatrix[1][1] += 2e-4;
    normalMatrix[2][2] += 2e-4;
    detM = determinant(normalMatrix);
    if (abs(detM) < 1e-10) return false;
    mat3 invM = inverse(normalMatrix);
    betaR = invM * rhsR;
    betaG = invM * rhsG;
    betaB = invM * rhsB;
    return true;
}

vec3 evaluateRegression(vec3 betaR, vec3 betaG, vec3 betaB, vec2 d) {
    vec3 phi = vec3(1.0, d.x, d.y);
    return vec3(dot(betaR, phi), dot(betaG, phi), dot(betaB, phi));
}

bool fitColorRegression(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    bool foreground,
    vec3 refAlbedo,
    vec3 refNormal,
    vec3 refPosition,
    vec2 edgeNormal,
    float edgeStrength,
    bool anisotropic,
    bool robustPass,
    vec3 priorR,
    vec3 priorG,
    vec3 priorB,
    out vec3 betaR,
    out vec3 betaG,
    out vec3 betaB,
    out vec3 weightedMean,
    out vec3 localMin,
    out vec3 localMax,
    out float support
) {
    mat3 M = mat3(0.0);
    vec3 rhsR = vec3(0.0);
    vec3 rhsG = vec3(0.0);
    vec3 rhsB = vec3(0.0);
    vec3 sum = vec3(0.0);
    float sumW = 0.0;
    support = 0.0;
    localMin = vec3(1e30);
    localMax = vec3(-1e30);

    float sigma = max(float(radius) * 0.82, 0.85);

    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            int sampleKey = surfaceKey(p);
            bool belongs = foreground ? sampleKey == key : sampleKey == 0;
            if (!belongs) continue;

            vec2 d = vec2(x, y);
            float w = anisotropicSpatial(d, sigma, edgeNormal, edgeStrength, anisotropic);
            float guide = guideWeight(p, refAlbedo, refNormal, refPosition);
            if (guide < (foreground ? 0.015 : 0.045)) continue;
            w *= guide;
            w *= edgeSideGate(d, edgeNormal, edgeStrength, foreground);
            if (w <= 1e-6) continue;

            vec3 c = litAt(p);
            if (robustPass) {
                vec3 predicted = evaluateRegression(priorR, priorG, priorB, d);
                float residual = length(c - predicted);
                float scale = 0.035 + 0.16 * max(luminance(abs(predicted)), 0.05);
                float t = residual / max(scale, 1e-4);
                w *= 1.0 / (1.0 + t * t);
            }

            if (w <= 1e-7) continue;
            addRegressionSample(d, c, w, M, rhsR, rhsG, rhsB);
            sum += c * w;
            sumW += w;
            support += 1.0;
            localMin = min(localMin, c);
            localMax = max(localMax, c);
        }
    }

    if (sumW <= EPS) return false;
    weightedMean = sum / sumW;

    // With very sparse alpha we still return a stable mean rather than leaking
    // the raw screen-door sample back into the final image.
    if (support < 3.0) {
        betaR = vec3(weightedMean.r, 0.0, 0.0);
        betaG = vec3(weightedMean.g, 0.0, 0.0);
        betaB = vec3(weightedMean.b, 0.0, 0.0);
        return true;
    }
    if (!solveRegression(M, rhsR, rhsG, rhsB, betaR, betaG, betaB)) {
        betaR = vec3(weightedMean.r, 0.0, 0.0);
        betaG = vec3(weightedMean.g, 0.0, 0.0);
        betaB = vec3(weightedMean.b, 0.0, 0.0);
    }
    return true;
}


bool directionalPairEstimate(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    bool foreground,
    vec3 refAlbedo,
    vec3 refNormal,
    vec3 refPosition,
    vec2 edgeNormal,
    float edgeStrength,
    out vec3 estimate,
    out float confidence
) {
    const ivec2 directions[4] = ivec2[4](
        ivec2(1, 0), ivec2(0, 1), ivec2(1, 1), ivec2(1, -1)
    );

    vec3 sum = vec3(0.0);
    float sumW = 0.0;
    int pairCount = 0;

    for (int directionIndex = 0; directionIndex < 4; ++directionIndex) {
        ivec2 direction = directions[directionIndex];
        bool haveNegative = false;
        bool havePositive = false;
        ivec2 negativePixel = center;
        ivec2 positivePixel = center;
        int negativeDistance = 0;
        int positiveDistance = 0;

        for (int step = 1; step <= 3; ++step) {
            if (step > radius) break;
            if (!haveNegative) {
                ivec2 p = clampPixel(center - direction * step, size);
                int sampleKey = surfaceKey(p);
                bool component = foreground ? sampleKey == key : sampleKey == 0;
                if (component) {
                    float guide = guideWeight(p, refAlbedo, refNormal, refPosition);
                    guide *= edgeSideGate(vec2(p - center), edgeNormal, edgeStrength, foreground);
                    if (guide >= (foreground ? 0.025 : 0.060)) {
                        haveNegative = true;
                        negativePixel = p;
                        negativeDistance = step;
                    }
                }
            }
            if (!havePositive) {
                ivec2 p = clampPixel(center + direction * step, size);
                int sampleKey = surfaceKey(p);
                bool component = foreground ? sampleKey == key : sampleKey == 0;
                if (component) {
                    float guide = guideWeight(p, refAlbedo, refNormal, refPosition);
                    guide *= edgeSideGate(vec2(p - center), edgeNormal, edgeStrength, foreground);
                    if (guide >= (foreground ? 0.025 : 0.060)) {
                        havePositive = true;
                        positivePixel = p;
                        positiveDistance = step;
                    }
                }
            }
        }

        if (!haveNegative || !havePositive) continue;

        vec3 negativeColor = litAt(negativePixel);
        vec3 positiveColor = litAt(positivePixel);
        float totalDistance = float(negativeDistance + positiveDistance);
        vec3 candidate = (
            negativeColor * float(positiveDistance)
            + positiveColor * float(negativeDistance)
        ) / max(totalDistance, 1.0);

        float gradient = length(positiveColor - negativeColor) / max(totalDistance, 1.0);
        float guideNegative = guideWeight(
            negativePixel, refAlbedo, refNormal, refPosition
        ) * edgeSideGate(vec2(negativePixel - center), edgeNormal, edgeStrength, foreground);
        float guidePositive = guideWeight(
            positivePixel, refAlbedo, refNormal, refPosition
        ) * edgeSideGate(vec2(positivePixel - center), edgeNormal, edgeStrength, foreground);
        float guide = sqrt(max(guideNegative * guidePositive, 0.0));
        float distanceWeight = 1.0 / (0.65 + totalDistance);
        float gradientWeight = 1.0 / (0.035 + gradient);
        // The pair estimator is phase-neutral. The ordered pattern controls which
        // samples exist, but target-phase-specific color weights are deliberately
        // avoided to prevent periodic residuals in reconstructed radiance.
        float w = guide * distanceWeight * gradientWeight;
        if (w <= 1e-5) continue;
        sum += candidate * w;
        sumW += w;
        pairCount += 1;
    }

    if (sumW <= EPS || pairCount == 0) {
        estimate = vec3(0.0);
        confidence = 0.0;
        return false;
    }

    estimate = sum / sumW;
    confidence = clamp(float(pairCount) * 0.22 + sumW / (sumW + 8.0), 0.0, 1.0);
    return true;
}

bool solveScalarSystem(mat3 M, vec3 rhs, out vec3 beta) {
    float detM = determinant(M);
    if (abs(detM) < 1e-9) return false;
    beta = inverse(M) * rhs;
    return true;
}

bool fitSelectedDepthPlane(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    out vec3 beta,
    out float support
) {
    mat3 M = mat3(0.0);
    vec3 rhs = vec3(0.0);
    support = 0.0;
    float sigma = max(float(radius) * 0.95, 1.0);

    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            if (surfaceKey(p) != key) continue;

            vec2 d = vec2(p - center);
            vec3 phi = vec3(1.0, d.x, d.y);
            float w = gaussianSpatial(d, sigma);
            M += w * outerProduct(phi, phi);
            rhs += w * phi * controlAt(p).x;
            support += 1.0;
        }
    }

    if (support < 3.0) return false;
    if (solveScalarSystem(M, rhs, beta)) return true;

    // Near-collinear survivor layouts still provide a useful local depth mean.
    float sum = 0.0;
    float weight = 0.0;
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            if (surfaceKey(p) != key) continue;
            float w = gaussianSpatial(vec2(p - center), sigma);
            sum += controlAt(p).x * w;
            weight += w;
        }
    }
    if (weight <= EPS) return false;
    beta = vec3(sum / weight, 0.0, 0.0);
    return true;
}

float censoredNegativeConfidence(
    ivec2 p,
    vec2 d,
    vec2 edgeNormal,
    float edgeStrength,
    vec3 depthBeta,
    bool depthModelOk
) {
    if (roddAblated(RODD_ABLATE_GEOMETRY)) return 1.0;

    // Screen-space occupancy indicates the likely silhouette side. A local depth
    // plane adds an independent visibility test: if the extrapolated transparent
    // surface would be behind the currently selected lower scene, absence at this
    // pixel is not reliable evidence that alpha <= tau.
    float side = smoothstep(-0.40, 1.10, dot(d, edgeNormal));
    float silhouetteGate = mix(1.0, side, edgeStrength);

    float visibilityGate = 1.0;
    if (depthModelOk) {
        float predictedDepth = dot(depthBeta, vec3(1.0, d.x, d.y));
        float selectedDepth = controlAt(p).x;
        float frontMargin = selectedDepth - predictedDepth;
        visibilityGate = smoothstep(-2e-4, 8e-4, frontMargin);
    }

    return clamp(silhouetteGate * visibilityGate, 0.0, 1.0);
}

// For Bayer 8x8, estimate alpha from one phase-anchored complete measurement cell.
// This is used only when the target is not near a silhouette. It removes the
// target-phase-dependent alpha ripple of a sliding 7x7 neighborhood while remaining
// an ordinal inversion: no source-alpha or rejected-fragment attributes are read.
bool phaseAnchoredOrdinalInterval8(
    ivec2 center,
    ivec2 framebufferSize,
    int key,
    vec2 edgeNormal,
    float edgeStrength,
    vec3 depthBeta,
    bool depthModelOk,
    out float midpoint,
    out float confidence
) {
    midpoint = 0.5;
    confidence = 0.0;
    if (uDitherMode == 1 || patternSizeForMode(uDitherMode) != 8 || edgeStrength > 0.16) return false;

    ivec2 origin;
    int patternSize;
    if (!orderedCellOrigin(center, framebufferSize, key, origin, patternSize)) return false;

    float lower = 0.0;
    float upper = 1.0;
    float positives = 0.0;
    float reliableNegatives = 0.0;
    float negativeConfidenceSum = 0.0;
    float quantStep = 1.0 / 64.0;

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            ivec2 p = origin + ivec2(x, y);
            vec2 d = vec2(p - center);
            float tau = thresholdForSurface(p, key);
            bool positive = surfaceKey(p) == key;
            if (positive) {
                lower = max(lower, min(tau + 0.02 * quantStep, 1.0));
                positives += 1.0;
            } else {
                float negativeConfidence = censoredNegativeConfidence(
                    p, d, edgeNormal, edgeStrength, depthBeta, depthModelOk
                );
                if (roddAblated(RODD_ABLATE_CENSORED_NEGATIVES)) negativeConfidence = 0.0;
                negativeConfidenceSum += negativeConfidence;
                if (negativeConfidence >= 0.72) {
                    upper = min(upper, tau);
                    reliableNegatives += 1.0;
                }
            }
        }
    }

    // A constant/slowly varying alpha field must admit one common interval. If the
    // complete cell contradicts that model, fall back to the local affine ordinal fit.
    if (positives < 1.0 || reliableNegatives < 1.0 || lower > upper + 1e-5) return false;

    float width = max(upper - lower, quantStep);
    float tightness = 1.0 / (1.0 + width / (3.0 * quantStep));
    float evidence = clamp((positives + reliableNegatives) / 64.0, 0.0, 1.0);
    float negativeEvidence = clamp(negativeConfidenceSum / 32.0, 0.0, 1.0);
    midpoint = clamp(0.5 * (lower + upper), 0.5 * quantStep, 1.0 - 0.5 * quantStep);
    confidence = clamp((0.55 + 0.45 * evidence) * (0.65 + 0.35 * negativeEvidence) * tightness, 0.0, 1.0);
    return true;
}

void ordinalPriorAndGeometrySupport(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    vec2 edgeNormal,
    float edgeStrength,
    vec3 depthBeta,
    bool depthModelOk,
    out float alphaPrior,
    out float geometrySupport
) {
    float occupied = 0.0;
    float eligible = 0.0;
    float footprint = 0.0;

    if (uDitherMode != 1 && patternSizeForMode(uDitherMode) == 4) {
        ivec2 origin = phaseFootprintOrigin(center, size);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                ivec2 p = origin + ivec2(x, y);
                vec2 d = vec2(p - center);
                bool positive = surfaceKey(p) == key;
                float supportConfidence = positive ? 1.0 : censoredNegativeConfidence(
                    p, d, edgeNormal, edgeStrength, depthBeta, depthModelOk
                );
                occupied += positive ? 1.0 : 0.0;
                eligible += supportConfidence;
                footprint += 1.0;
            }
        }
    } else {
        float sigma = max(float(radius) * 0.95, 1.15);
        for (int y = -3; y <= 3; ++y) {
            for (int x = -3; x <= 3; ++x) {
                if (abs(x) > radius || abs(y) > radius) continue;
                ivec2 p = clampPixel(center + ivec2(x, y), size);
                vec2 d = vec2(p - center);
                float spatial = gaussianSpatial(d, sigma);
                bool positive = surfaceKey(p) == key;
                float supportConfidence = positive ? 1.0 : censoredNegativeConfidence(
                    p, d, edgeNormal, edgeStrength, depthBeta, depthModelOk
                );
                occupied += (positive ? 1.0 : 0.0) * spatial;
                eligible += supportConfidence * spatial;
                footprint += spatial;
            }
        }
    }

    geometrySupport = clamp(eligible / max(footprint, EPS), 0.0, 1.0);
    alphaPrior = clamp(occupied / max(eligible, EPS), 0.0, 1.0);
}


bool ordinalIntervalPrior(
    ivec2 center,
    ivec2 size,
    int key,
    vec2 edgeNormal,
    float edgeStrength,
    vec3 depthBeta,
    bool depthModelOk,
    out float midpoint,
    out float intervalConfidence
) {
    if (uDitherMode == 1) {
        midpoint = 0.5;
        intervalConfidence = 0.0;
        return false;
    }

    float lower = 0.0;
    float upper = 1.0;
    float positiveCount = 0.0;
    float negativeConfidenceSum = 0.0;
    float evidenceCount = 0.0;
    float quantStep = orderedQuantStep();
    int patternSize = patternSizeForMode(uDitherMode);

    if (patternSize <= 4) {
        ivec2 origin = phaseFootprintOrigin(center, size);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                ivec2 p = origin + ivec2(x, y);
                vec2 d = vec2(p - center);
                float tau = thresholdForSurface(p, key);
                bool positive = surfaceKey(p) == key;
                if (positive) {
                    lower = max(lower, min(tau + 0.02 * quantStep, 1.0));
                    positiveCount += 1.0;
                    evidenceCount += 1.0;
                } else {
                    float negativeConfidence = censoredNegativeConfidence(
                        p, d, edgeNormal, edgeStrength, depthBeta, depthModelOk
                    );
                    if (roddAblated(RODD_ABLATE_CENSORED_NEGATIVES)) negativeConfidence = 0.0;
                    negativeConfidenceSum += negativeConfidence;
                    evidenceCount += negativeConfidence;
                    if (negativeConfidence >= 0.62) upper = min(upper, tau);
                }
            }
        }
    } else {
        // 8x8/16x16 improve threshold resolution, but reconstruction remains local.
        // A 7x7 measurement neighborhood samples many fine thresholds without turning
        // the radiance reconstruction into a 16x16 blur kernel.
        for (int y = -3; y <= 3; ++y) {
            for (int x = -3; x <= 3; ++x) {
                ivec2 p = clampPixel(center + ivec2(x, y), size);
                vec2 d = vec2(p - center);
                float tau = thresholdForSurface(p, key);
                bool positive = surfaceKey(p) == key;
                if (positive) {
                    lower = max(lower, min(tau + 0.02 * quantStep, 1.0));
                    positiveCount += 1.0;
                    evidenceCount += 1.0;
                } else {
                    float negativeConfidence = censoredNegativeConfidence(
                        p, d, edgeNormal, edgeStrength, depthBeta, depthModelOk
                    );
                    if (roddAblated(RODD_ABLATE_CENSORED_NEGATIVES)) negativeConfidence = 0.0;
                    negativeConfidenceSum += negativeConfidence;
                    evidenceCount += negativeConfidence;
                    if (negativeConfidence >= 0.62) upper = min(upper, tau);
                }
            }
        }
    }

    if (positiveCount <= 0.0 || lower > upper + 1e-5) {
        midpoint = 0.5;
        intervalConfidence = 0.0;
        return false;
    }

    midpoint = clamp(0.5 * (lower + upper), 0.5 * quantStep, 1.0 - 0.5 * quantStep);
    float nominalSamples = patternSize <= 4 ? 16.0 : 49.0;
    float evidence = clamp(evidenceCount / nominalSamples, 0.0, 1.0);
    float hasUpperBound = upper < 1.0 - 0.25 * quantStep ? 1.0 : 0.35;
    float intervalWidth = max(upper - lower, quantStep);
    float tightness = 1.0 / (1.0 + intervalWidth / max(4.0 * quantStep, 0.035));
    float negativeEvidence = clamp(negativeConfidenceSum / max(nominalSamples * 0.5, 1.0), 0.0, 1.0);
    intervalConfidence = clamp(
        (0.30 + 0.70 * evidence) * (0.45 + 0.55 * negativeEvidence) * hasUpperBound * (0.55 + 0.45 * tightness),
        0.0,
        1.0
    );
    return true;
}

bool fitOrdinalAlpha(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    vec2 edgeNormal,
    float edgeStrength,
    out vec3 beta,
    out float constraintSupport,
    out float conditionProxy,
    out float geometrySupport
) {
    vec3 depthBeta = vec3(0.0);
    float depthSupport = 0.0;
    bool depthModelOk = fitSelectedDepthPlane(center, size, radius, key, depthBeta, depthSupport);

    float prior = 0.0;
    ordinalPriorAndGeometrySupport(
        center, size, radius, key,
        edgeNormal, edgeStrength,
        depthBeta, depthModelOk,
        prior, geometrySupport
    );

    float quantStep = orderedQuantStep();
    prior = clamp(prior, 0.5 * quantStep, 1.0 - 0.5 * quantStep);

    // Ordered coverage gives an actual quantized interval, not merely an occupancy
    // ratio. In well-supported regions use its midpoint as the intercept prior.
    // Near silhouettes the geometry-conditioned occupancy prior remains safer.
    float intervalMidpoint = prior;
    float intervalConfidence = 0.0;
    if (ordinalIntervalPrior(
        center, size, key, edgeNormal, edgeStrength, depthBeta, depthModelOk,
        intervalMidpoint, intervalConfidence
    )) {
        // Non-identity coverage encodings deliberately bias the survivor ratio, so
        // occupancy is no longer an alpha estimate. Larger Bayer matrices likewise
        // use only a local subset of their phases. In both cases the inverse-mapped
        // ordinal interval is the physically meaningful alpha prior.
        bool measurementPreconditioned = uCoverageEncodingMode != 0 || patternSizeForMode(uDitherMode) > 4;
        float intervalBlend = measurementPreconditioned
            ? clamp(0.58 + 0.42 * intervalConfidence, 0.0, 1.0)
            : intervalConfidence * geometrySupport * (1.0 - 0.65 * edgeStrength);
        prior = mix(prior, intervalMidpoint, clamp(intervalBlend, 0.0, 1.0));
    }

    // On a locally constant/slow alpha field, one complete Bayer 8x8 cell provides
    // a phase-invariant quantization interval. Use it as a strong prior only when
    // all reliable inequalities admit the same interval; otherwise the local affine
    // ordinal model remains in control (gradients, cutouts, silhouettes).
    float anchoredMidpoint = prior;
    float anchoredConfidence = 0.0;
    if (geometrySupport > 0.72 && phaseAnchoredOrdinalInterval8(
        center, size, key, edgeNormal, edgeStrength, depthBeta, depthModelOk,
        anchoredMidpoint, anchoredConfidence
    )) {
        float anchoredBlend = clamp(anchoredConfidence * geometrySupport, 0.0, 0.96);
        prior = mix(prior, anchoredMidpoint, anchoredBlend);
    }

    beta = vec3(prior, 0.0, 0.0);
    constraintSupport = 0.0;
    conditionProxy = 1e6;

    // Occupancy-only ablation: preserve the same geometry support and pattern,
    // but remove all threshold inequalities from the inverse problem.
    if (roddAblated(RODD_ABLATE_ORDINAL)) {
        conditionProxy = 1.0;
        return geometrySupport > 1e-3 && stableCoverageRank16(center, size, key) > 0;
    }

    // Fixed-count active-set IRLS for one-sided ordinal inequalities:
    // survivor => alpha(p) > tau(p), censored rejected => alpha(p) <= tau(p).
    // The geometry-derived opacity prior selects a stable member of a feasible
    // interval without reading the hidden source-alpha channel.
    for (int iteration = 0; iteration < 5; ++iteration) {
        mat3 M = mat3(0.0);
        vec3 rhs = vec3(0.0);
        float activeConstraintWeight = 0.0;

        float interceptPrior = mix(0.055, 1.25, anchoredConfidence);
        const float slopePrior = 0.018;
        M[0][0] += interceptPrior;
        M[1][1] += slopePrior;
        M[2][2] += slopePrior;
        rhs.x += interceptPrior * prior;

        float sigma = max(float(radius) * 0.95, 1.15);
        for (int y = -3; y <= 3; ++y) {
            for (int x = -3; x <= 3; ++x) {
                if (abs(x) > radius || abs(y) > radius) continue;
                ivec2 p = clampPixel(center + ivec2(x, y), size);
                vec2 d = vec2(p - center);
                vec3 phi = vec3(1.0, d.x, d.y);
                float predicted = clamp(dot(beta, phi), 0.0, 1.0);
                float tau = thresholdForSurface(p, key);
                bool positive = surfaceKey(p) == key;
                float spatial = anisotropicSpatial(d, sigma, edgeNormal, edgeStrength, true);
                float w = spatial;
                float target = predicted;
                bool violated = false;

                if (positive) {
                    float lower = min(tau + 0.10 * quantStep, 1.0);
                    violated = predicted < lower;
                    target = lower;
                } else {
                    float negativeConfidence = censoredNegativeConfidence(
                        p, d, edgeNormal, edgeStrength, depthBeta, depthModelOk
                    );
                    if (roddAblated(RODD_ABLATE_CENSORED_NEGATIVES)) negativeConfidence = 0.0;
                    violated = predicted > tau;
                    target = tau;
                    w *= negativeConfidence;
                }

                if (!violated || w <= 1e-5) continue;
                M += w * outerProduct(phi, phi);
                rhs += w * phi * target;
                activeConstraintWeight += w;
            }
        }

        vec3 nextBeta;
        if (!solveScalarSystem(M, rhs, nextBeta)) break;
        beta = mix(beta, nextBeta, 0.78);
        beta.x = clamp(beta.x, 0.0, 1.0);
        beta.yz = clamp(beta.yz, vec2(-0.28), vec2(0.28));
        constraintSupport = max(constraintSupport, activeConstraintWeight);

        float diagProduct = max(M[0][0] * M[1][1] * M[2][2], 1e-12);
        conditionProxy = clamp(diagProduct / max(abs(determinant(M)), 1e-12), 1.0, 1e6);
    }

    // A feasible interval can require no active hinge constraints. Do not mark such
    // a solution as catastrophically ill-conditioned merely because the active set
    // is empty; use interval evidence as the confidence proxy in that case.
    if (conditionProxy >= 9e5) {
        if (anchoredConfidence > 0.0) conditionProxy = mix(24.0, 1.0, anchoredConfidence);
        else conditionProxy = mix(128.0, 8.0, intervalConfidence);
    }

    // A feasible prior is a valid interval solution even when no hinge constraint
    // had to activate. We only fail when there is no meaningful visible support.
    return geometrySupport > 1e-3 && stableCoverageRank16(center, size, key) > 0;
}

float measurementNormalizedGeometrySupport(
    ivec2 center,
    ivec2 size,
    int key,
    float alphaHat,
    float priorSupport
) {
    // Compare observed survivors with the survivors predicted by the recovered
    // physical alpha under the known measurement thresholds. This separates
    // missing samples caused by low opacity from missing samples caused by the
    // geometric silhouette, without needing rejected-fragment attributes.
    float observedMass = 0.0;
    float expectedMass = 0.0;
    float weightMass = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            vec2 d = vec2(x, y);
            float w = gaussianSpatial(d, 1.15);
            float expected = alphaHat > thresholdForSurface(p, key) ? 1.0 : 0.0;
            float observed = surfaceKey(p) == key ? 1.0 : 0.0;
            expectedMass += expected * w;
            observedMass += observed * w;
            weightMass += w;
        }
    }

    // If this particular phase neighborhood contains almost no expected positive
    // measurements, it cannot identify geometry support by itself.
    if (expectedMass < 0.12 || weightMass <= EPS) return priorSupport;

    float ratio = clamp(observedMass / expectedMass, 0.0, 1.0);
    float crispSupport = smoothstep(0.18, 0.82, ratio);
    return clamp(mix(priorSupport, crispSupport, 0.82), 0.0, 1.0);
}

float estimateCoverageRODD(
    ivec2 center,
    ivec2 size,
    int radius,
    int key,
    vec2 edgeNormal,
    float edgeStrength,
    out float alphaHat,
    out float geometrySupport,
    out float ordinalSupport,
    out float conditionProxy
) {
    vec3 alphaBeta = vec3(0.0);
    bool alphaOk = fitOrdinalAlpha(
        center, size, radius, key,
        edgeNormal, edgeStrength,
        alphaBeta, ordinalSupport, conditionProxy, geometrySupport
    );

    if (!alphaOk) {
        alphaHat = smoothedMembership(center, size, key);
        geometrySupport = alphaHat > EPS ? 1.0 : 0.0;
        return alphaHat;
    }

    alphaHat = clamp(alphaBeta.x, 0.0, 1.0);
    geometrySupport = measurementNormalizedGeometrySupport(center, size, key, alphaHat, geometrySupport);
    if (geometrySupport > 0.97 && edgeStrength < 0.12) geometrySupport = 1.0;

    // geometrySupport is a visibility/support estimate, not a second opacity term.
    // It affects physical coverage only when either the geometric edge detector or
    // the measurement-normalized support indicates a silhouette. The nonlinear
    // support mapping intentionally avoids the broad, soft shadow of v1.6/v1.7.
    float edgeCue = smoothstep(0.13, 0.46, edgeStrength);
    float supportCue = smoothstep(0.10, 0.55, 1.0 - geometrySupport);
    float silhouetteMix = max(edgeCue, supportCue);
    float silhouetteVisibility = mix(1.0, geometrySupport, silhouetteMix);
    return clamp(alphaHat * silhouetteVisibility, 0.0, 1.0);
}

float occupancy2x2(ivec2 center, ivec2 size, int key) {
    float sum = 0.0;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            sum += surfaceKey(p) == key ? 1.0 : 0.0;
        }
    }
    return sum * 0.25;
}

float legacyCoverage(
    ivec2 center,
    ivec2 size,
    int radius,
    int key
) {
    float occupied = 0.0;
    float total = 0.0;
    for (int y = -3; y <= 3; ++y) {
        for (int x = -3; x <= 3; ++x) {
            if (abs(x) > radius || abs(y) > radius) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            occupied += surfaceKey(p) == key ? 1.0 : 0.0;
            total += 1.0;
        }
    }
    return total > EPS ? occupied / total : 0.0;
}

void main() {
    ivec2 size = textureSize(uLit, 0);
    ivec2 center = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
    vec3 raw = litAt(center);

    if (uMethod == 0) {
        outColor = vec4(raw, 1.0);
        outAlphaMeta = vec4(surfaceKey(center) > 0 ? 1.0 : 0.0, 0.0, 0.0, 1.0);
        return;
    }

    int radius = uDitherMode != 1 ? max(uKernelRadius, 2) : clamp(uKernelRadius, 1, 3);
    int keySearchRadius = uMethod >= 6 ? max(radius, 3) : radius;
    int key = nearestTransparentKey(center, size, keySearchRadius);
    if (key <= 0) {
        outColor = vec4(raw, 1.0);
        outAlphaMeta = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 fgRefA, fgRefN, fgRefP;
    vec3 bgRefA, bgRefN, bgRefP;
    int referenceRadius = uMethod >= 6 ? max(radius, 5) : radius;
    bool haveFg = findReference(center, size, referenceRadius, key, true, fgRefA, fgRefN, fgRefP);
    bool haveBg = findReference(center, size, referenceRadius, key, false, bgRefA, bgRefN, bgRefP);

    if (!haveFg) {
        outColor = vec4(raw, 1.0);
        outAlphaMeta = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    if (!haveBg) {
        bgRefA = fgRefA;
        bgRefN = fgRefN;
        bgRefP = fgRefP;
    }

    // Historical baselines 1-4 remain unchanged in spirit.
    if (uMethod <= 4) {
        vec3 fgSum = vec3(0.0);
        vec3 bgSum = vec3(0.0);
        float fgW = 0.0;
        float bgW = 0.0;
        float sigma = max(float(radius) * 0.9, 0.8);

        for (int y = -3; y <= 3; ++y) {
            for (int x = -3; x <= 3; ++x) {
                if (abs(x) > radius || abs(y) > radius) continue;
                ivec2 p = clampPixel(center + ivec2(x, y), size);
                bool fg = surfaceKey(p) == key;
                float w = 1.0;

                if (uMethod >= 3) w *= gaussianSpatial(vec2(x, y), sigma);
                if (uMethod >= 4) {
                    if (fg) w *= guideWeight(p, fgRefA, fgRefN, fgRefP);
                    else if (haveBg) w *= guideWeight(p, bgRefA, bgRefN, bgRefP);
                }

                if (fg) {
                    fgSum += litAt(p) * w;
                    fgW += w;
                } else {
                    bgSum += litAt(p) * w;
                    bgW += w;
                }
            }
        }

        vec3 fg = fgW > EPS ? fgSum / fgW : raw;
        vec3 bg = bgW > EPS ? bgSum / bgW : fg;
        float coverage = uMethod == 1
            ? occupancy2x2(center, size, key)
            : legacyCoverage(center, size, radius, key);
        outColor = vec4(mix(bg, fg, clamp(coverage, 0.0, 1.0)), 1.0);
        outAlphaMeta = vec4(coverage, 1.0, 0.0, surfaceKey(center) == key ? 1.0 : 0.0);
        return;
    }

    vec2 edgeNormal;
    float edgeStrength;
    coverageGradient(center, size, key, edgeNormal, edgeStrength);

    bool proposed = uMethod >= 6;
    if (proposed && uDitherMode != 1 && !roddAblated(RODD_ABLATE_RANK_CONDITIONING)) {
        // Rank is phase-stable for Bayer 8x8. Sparse coverage needs enough samples,
        // while silhouettes must stay compact to avoid the blurred outline seen in v1.7.
        int rank = stableCoverageRank16(center, size, key);
        if (rank <= 4) radius = 3;
        else radius = min(radius, 2);
        if (edgeStrength > 0.32) radius = min(radius, 2);
        if (edgeStrength > 0.68) radius = 1;
    }
    vec3 zero = vec3(0.0);

    vec3 fgR, fgG, fgB, fgMean, fgMin, fgMax;
    vec3 bgR, bgG, bgB, bgMean, bgMin, bgMax;
    float fgSupport = 0.0;
    float bgSupport = 0.0;

    bool fgOk = fitColorRegression(
        center, size, radius, key, true,
        fgRefA, fgRefN, fgRefP,
        edgeNormal, edgeStrength,
        proposed, false,
        zero, zero, zero,
        fgR, fgG, fgB, fgMean, fgMin, fgMax, fgSupport
    );

    bool bgOk = haveBg && fitColorRegression(
        center, size, radius, key, false,
        bgRefA, bgRefN, bgRefP,
        edgeNormal, edgeStrength,
        proposed, false,
        zero, zero, zero,
        bgR, bgG, bgB, bgMean, bgMin, bgMax, bgSupport
    );

    // If a transparent key is present, never fall back to the raw dithered center.
    // A sparse mean/model is preferred to reintroducing the screen-door sample.
    if (!fgOk) {
        vec3 nearestFg = litAt(center);
        nearestSurfaceColor(center, size, key, 5, nearestFg);
        fgR = vec3(nearestFg.r, 0.0, 0.0);
        fgG = vec3(nearestFg.g, 0.0, 0.0);
        fgB = vec3(nearestFg.b, 0.0, 0.0);
        fgMean = nearestFg;
        fgMin = nearestFg;
        fgMax = nearestFg;
        fgSupport = 1.0;
    }

    if (proposed) {
        vec3 rR, rG, rB, rMean, rMin, rMax;
        float rSupport = 0.0;
        if (fitColorRegression(
            center, size, radius, key, true,
            fgRefA, fgRefN, fgRefP,
            edgeNormal, edgeStrength,
            true, true,
            fgR, fgG, fgB,
            rR, rG, rB, rMean, rMin, rMax, rSupport
        )) {
            fgR = rR;
            fgG = rG;
            fgB = rB;
            fgMean = rMean;
            fgMin = rMin;
            fgMax = rMax;
            fgSupport = rSupport;
        }

        if (bgOk) {
            vec3 rR2, rG2, rB2, rMean2, rMin2, rMax2;
            float rSupport2 = 0.0;
            if (fitColorRegression(
                center, size, radius, key, false,
                bgRefA, bgRefN, bgRefP,
                edgeNormal, edgeStrength,
                true, true,
                bgR, bgG, bgB,
                rR2, rG2, rB2, rMean2, rMin2, rMax2, rSupport2
            )) {
                bgR = rR2;
                bgG = rG2;
                bgB = rB2;
                bgMean = rMean2;
                bgMin = rMin2;
                bgMax = rMax2;
                bgSupport = rSupport2;
            }
        }
    }


    vec3 fgDirectional = vec3(0.0);
    vec3 bgDirectional = vec3(0.0);
    float fgDirectionalConfidence = 0.0;
    float bgDirectionalConfidence = 0.0;
    bool fgDirectionalOk = proposed && !roddAblated(RODD_ABLATE_DIRECTIONAL_PAIRS) && directionalPairEstimate(
        center, size, radius, key, true,
        fgRefA, fgRefN, fgRefP,
        edgeNormal, edgeStrength,
        fgDirectional, fgDirectionalConfidence
    );
    bool bgDirectionalOk = proposed && haveBg && !roddAblated(RODD_ABLATE_DIRECTIONAL_PAIRS) && directionalPairEstimate(
        center, size, radius, key, false,
        bgRefA, bgRefN, bgRefP,
        edgeNormal, edgeStrength,
        bgDirectional, bgDirectionalConfidence
    );

    bool useMomentReproduction = proposed && !roddAblated(RODD_ABLATE_MOMENTS);
    vec3 foregroundModel = useMomentReproduction
        ? max(evaluateRegression(fgR, fgG, fgB, vec2(0.0)), vec3(0.0))
        : max(fgMean, vec3(0.0));
    vec3 backgroundModel = bgOk
        ? (useMomentReproduction
            ? max(evaluateRegression(bgR, bgG, bgB, vec2(0.0)), vec3(0.0))
            : max(bgMean, vec3(0.0)))
        : foregroundModel;

    // Edge-directed interpolation is the detail-preserving component. Pattern phase
    // affects sample availability and alpha constraints, but not radiance weights.
    // This separation is deliberate: v1.7's phase-weighted color fit reproduced the
    // Bayer pattern in the final image.
    int reconstructionRank = stableCoverageRank16(center, size, key);
    float sparseFactor = smoothstep(2.0, 8.0, float(reconstructionRank));
    float edgeFactor = 1.0 - 0.45 * edgeStrength;
    float fgDirectionalBlend = fgDirectionalOk
        ? clamp(fgDirectionalConfidence * (0.10 + 0.18 * sparseFactor) * edgeFactor, 0.0, 0.32)
        : 0.0;
    float bgDirectionalBlend = bgDirectionalOk
        ? clamp(bgDirectionalConfidence * 0.18 * edgeFactor, 0.0, 0.24)
        : 0.0;

    vec3 foreground = mix(foregroundModel, max(fgDirectional, vec3(0.0)), fgDirectionalBlend);
    vec3 background = mix(backgroundModel, max(bgDirectional, vec3(0.0)), bgDirectionalBlend);
    foreground = clamp(foreground, fgMin, fgMax);
    if (bgOk) background = clamp(background, bgMin, bgMax);

    vec3 safeForegroundMean = localSeparatedMean(
        center, size, key, true,
        fgRefA, fgRefN, fgRefP,
        edgeNormal, edgeStrength,
        max(radius, 2), max(fgMean, vec3(0.0))
    );
    vec3 safeBackgroundMean = bgOk
        ? localSeparatedMean(center, size, key, false, bgRefA, bgRefN, bgRefP, edgeNormal, edgeStrength, max(radius, 2), max(bgMean, vec3(0.0)))
        : safeForegroundMean;

    float modelConfidence = 1.0;
    float alphaHat = 0.0;
    float geometrySupport = 1.0;
    float ordinalSupport = 0.0;
    float conditionProxy = 1.0;
    float coverage;

    if (proposed) {
        coverage = estimateCoverageRODD(
            center, size, radius, key,
            edgeNormal, edgeStrength,
            alphaHat, geometrySupport, ordinalSupport, conditionProxy
        );
    } else {
        coverage = legacyCoverage(center, size, radius, key);
        alphaHat = coverage;
    }

    // Confidence-aware fusion is the safety mechanism for the information-poor
    // low-alpha regime. When the local inverse problem is weak, prefer a phase-neutral
    // normalized estimate over an unstable high-order prediction. This intentionally
    // trades some detail for removal of structured screen-door artifacts only where
    // the data do not support reliable detail reconstruction.
    if (proposed && !roddAblated(RODD_ABLATE_CONFIDENCE_GUARD)) {
        float rankConfidence = smoothstep(1.5, 7.0, float(reconstructionRank));
        float supportConfidence = clamp((fgSupport - 1.0) / 5.0, 0.0, 1.0);
        float conditionConfidence = 1.0 / (1.0 + 0.22 * log2(max(conditionProxy, 1.0)));
        float geometryConfidence = mix(0.68, 1.0, geometrySupport);
        modelConfidence = clamp(
            (0.34 * rankConfidence + 0.34 * supportConfidence + 0.32 * conditionConfidence)
            * geometryConfidence * (1.0 - 0.42 * edgeStrength),
            0.0,
            1.0
        );

        vec3 safeForeground = safeForegroundMean;
        if (fgDirectionalOk) {
            float safeDirectional = clamp(fgDirectionalConfidence * 0.12 * rankConfidence, 0.0, 0.12);
            safeForeground = mix(safeForeground, max(fgDirectional, vec3(0.0)), safeDirectional);
        }
        foreground = mix(safeForeground, foreground, modelConfidence);
        foreground = clamp(foreground, fgMin, fgMax);

        if (bgOk) {
            float backgroundConfidence = clamp(0.30 + 0.70 * modelConfidence, 0.0, 1.0);
            background = mix(safeBackgroundMean, background, backgroundConfidence);
            background = clamp(background, bgMin, bgMax);
        }
    }

    if (proposed) {
        float localCoverage = localCoverageMean(center, size, key, edgeNormal, edgeStrength, max(radius, 2));
        float lowAlpha = 1.0 - smoothstep(0.16, 0.34, alphaHat);
        float silhouette = smoothstep(0.10, 0.42, edgeStrength);
        float instability = clamp((1.0 - modelConfidence) * (0.65 + 0.35 * lowAlpha), 0.0, 1.0);
        float alphaStabilization = clamp(0.55 * lowAlpha + 0.30 * instability + 0.18 * silhouette, 0.0, 0.85);
        alphaHat = mix(alphaHat, localCoverage, alphaStabilization);
        coverage = mix(coverage, localCoverage, clamp(0.75 * alphaStabilization, 0.0, 0.70));

        float colorStabilization = clamp(0.55 * instability + 0.20 * lowAlpha, 0.0, 0.80);
        foreground = mix(safeForegroundMean, foreground, 1.0 - colorStabilization);
        if (bgOk) background = mix(safeBackgroundMean, background, 1.0 - 0.55 * colorStabilization);
        foreground = clamp(foreground, fgMin, fgMax);
        if (bgOk) background = clamp(background, bgMin, bgMax);

        // Remove the zero-mean periodic residual that survives the local inversion.
        // The correction is intentionally small and phase-neutral: it suppresses the
        // known screen-door component without trying to invent new texture.
        vec3 residual = raw - mix(background, foreground, coverage);
        float residualGate = clamp((1.0 - modelConfidence) * (0.65 + 0.35 * lowAlpha) * (1.0 - 0.45 * silhouette), 0.0, 0.45);
        foreground = clamp(foreground + residual * residualGate, fgMin, fgMax);
    }

    if (proposed && uDetailStrength > 0.0) {
        // Optional foreground-only detail restoration. It is disabled by default
        // by default. The term is attenuated at silhouettes and whenever geometric
        // support is uncertain, so it cannot simply copy the raw dither pattern.
        float detailGate = geometrySupport * (1.0 - 0.85 * edgeStrength);
        vec3 detail = foreground - fgMean;
        vec3 sharpened = foreground + detail * (uDetailStrength * detailGate);
        foreground = clamp(sharpened, fgMin, fgMax);
    }

    vec3 reconstructed = mix(background, foreground, coverage);
    outColor = vec4(reconstructed, 1.0);

    // R = final reconstructed coverage
    // G = estimated geometric support after alpha normalization
    // B = normalized conditioning proxy (0 good, 1 poor)
    // A = reconstructed continuous alpha from ordinal thresholds only
    float conditioningMeta = proposed ? clamp(log2(max(conditionProxy, 1.0)) / 12.0, 0.0, 1.0) : edgeStrength;
    outAlphaMeta = vec4(
        coverage,
        geometrySupport,
        conditioningMeta,
        alphaHat
    );
}
