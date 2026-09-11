#version 460 core

in VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} v;

layout(binding = 0) uniform sampler2D uOpaqueDepth;
layout(binding = 3) uniform sampler2D uBaseTexture;
layout(binding = 0, r32ui) uniform uimage2D uInputOverdraw;
layout(binding = 1, r32ui) uniform uimage2D uSurvivingOverdraw;

uniform float uAlpha;
uniform int uAlphaMode;
uniform int uUseBaseTexture;
uniform int uDitherMode;
uniform int uCoverageEncodingMode;
uniform float uCoverageBias;
uniform uint uDitherSeed;
uniform uint uSurfaceId;

float materialAlpha(vec2 uv) {
    float a = uAlpha;
    if (uAlphaMode == 1) a *= mix(0.08, 1.0, uv.x);
    else if (uAlphaMode == 2) {
        vec2 p = uv * 2.0 - 1.0;
        float body = 1.0 - smoothstep(0.62, 0.82, length(vec2(p.x * 0.78, p.y)));
        float veins = 0.76 + 0.24 * sin((p.x + p.y) * 48.0);
        a *= body * veins;
    } else if (uAlphaMode == 3) {
        float cells = 0.55 + 0.45 * sin(uv.x * 18.0) * sin(uv.y * 22.0);
        a *= cells;
    }
    if (uUseBaseTexture != 0) a *= texture(uBaseTexture, uv).a;
    return clamp(a, 0.0, 1.0);
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
    int count = size * size;
    int parity = (phase.x + phase.y) & 1;
    if (parity == 0) return 2 * baseRank;
    return 2 * (baseRank - count / 2) + 1;
}

float patternThreshold(ivec2 p, uint seed, int mode) {
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
    if (uCoverageEncodingMode == 3 && mode != 2) baseRank = complementaryBayerRank(phase, size, baseRank);
    return (float(baseRank) + 0.5) / float(size * size);
}

float safeLogit(float a) {
    float x = clamp(a, 1e-5, 1.0 - 1e-5);
    return log(x / (1.0 - x));
}

float logistic(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float coverageBiasForPixel(ivec2 p, uint seed) {
    if (uCoverageEncodingMode == 0 || uCoverageEncodingMode == 4 || uCoverageEncodingMode == 5 || uCoverageBias <= 0.0) return 0.0;
    if (uCoverageEncodingMode == 1) return uCoverageBias;
    if (uCoverageEncodingMode == 2) return -uCoverageBias;
    if (uDitherMode != 1) {
        int size = patternSizeForMode(uDitherMode);
        int rank = clamp(int(floor(patternThreshold(p, seed, uDitherMode) * float(size * size))), 0, size * size - 1);
        return (rank & 1) == 0 ? uCoverageBias : -uCoverageBias;
    }
    uint h = uint(p.x) * 0x9E3779B9u ^ uint(p.y) * 0x85EBCA6Bu ^ (seed * 0xC2B2AE35u);
    h ^= h >> 16;
    return (h & 1u) == 0u ? uCoverageBias : -uCoverageBias;
}

float balancedLogitScale() {
    // uCoverageBias is a non-negative compression strength. Mapping it through
    // 1/(1+c) keeps the transform invertible and prevents the degenerate gamma=0
    // case while giving a simple identity point at c=0.
    return 1.0 / (1.0 + max(uCoverageBias, 0.0));
}

float encodeHighAlphaPower(float alpha) {
    const float pivot = 0.5;
    float p = max(uCoverageBias, 1.0);
    float a = clamp(alpha, 0.0, 1.0);
    if (a <= pivot || p <= 1.0001) return a;
    float t = clamp((a - pivot) / (1.0 - pivot), 0.0, 1.0);
    return pivot + (1.0 - pivot) * pow(t, p);
}

float encodeCoverageAlpha(float alpha, float bias) {
    if (uCoverageEncodingMode == 4) {
        float gamma = balancedLogitScale();
        return logistic(gamma * safeLogit(alpha));
    }
    if (uCoverageEncodingMode == 5) return encodeHighAlphaPower(alpha);
    if (abs(bias) <= 1e-6) return alpha;
    return logistic(safeLogit(alpha) + bias);
}

float seededBayer(ivec2 p) {
    uint seed = max(uSurfaceId, 1u);
    return patternThreshold(p, seed, uDitherMode);
}

float hash12(ivec2 p) {
    uint seed = max(uSurfaceId, 1u);
    uvec2 q = uvec2(p) ^ uvec2(seed, seed * 0x9E3779B9u);
    uint h = q.x * 0x9E3779B9u ^ q.y * 0x85EBCA6Bu;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return float(h & 0x00FFFFFFu) / float(0x01000000u);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (gl_FragCoord.z >= texelFetch(uOpaqueDepth, pixel, 0).r - 1e-7) discard;
    float alpha = materialAlpha(v.uv);
    if (alpha <= 0.001) discard;
    imageAtomicAdd(uInputOverdraw, pixel, 1u);
    uint sampleSeed = max(uSurfaceId, 1u);
    float threshold = uDitherMode != 1 ? seededBayer(pixel) : hash12(pixel);
    float encodedAlpha = encodeCoverageAlpha(alpha, coverageBiasForPixel(pixel, sampleSeed));
    if (encodedAlpha > threshold) imageAtomicAdd(uSurvivingOverdraw, pixel, 1u);
    discard;
}
