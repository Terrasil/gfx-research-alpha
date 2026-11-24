#version 460 core

in vec2 vUv;

layout(location = 0) out vec4 outAlbedoRoughness;
layout(location = 1) out vec4 outNormalValid;
layout(location = 2) out vec4 outPositionValid;
layout(location = 3) out vec4 outControl;

layout(binding = 0) uniform sampler2D uMixedAlbedo;
layout(binding = 1) uniform sampler2D uMixedNormal;
layout(binding = 2) uniform sampler2D uMixedPosition;
layout(binding = 3) uniform sampler2D uMixedControl;

uniform int uKernelRadius;
uniform int uDitherMode;
uniform int uCoverageEncodingMode;
uniform float uCoverageBias;
uniform float uDepthSigma;
uniform float uAlbedoSigma;
uniform int uLayerIndex;
uniform int uDecoupleOpacityConfidence; // 0: archived attenuation; 1: separate alpha from confidence
uniform int uOpacityEstimator; // 0: threshold/censor inference, 1: raw occupancy, 2: conditional occupancy, 3: exposure-aware ordinal blend

const float EPS = 1e-6;
const int MAX_CANDIDATES = 8;
const int MAX_RADIUS = 3;

bool withinKernel(int x, int y, int radius) {
    if (abs(x) > radius || abs(y) > radius) return false;
    return x * x + y * y <= radius * radius + radius;
}

ivec2 clampPixel(ivec2 p, ivec2 size) {
    return clamp(p, ivec2(0), size - 1);
}

vec4 albedoAt(ivec2 p) {
    return texelFetch(uMixedAlbedo, p, 0);
}

vec3 normalAt(ivec2 p) {
    vec3 n = texelFetch(uMixedNormal, p, 0).rgb * 2.0 - 1.0;
    float len2 = dot(n, n);
    if (len2 <= EPS) return vec3(0.0, 0.0, 1.0);
    return n * inversesqrt(len2);
}

vec4 positionAt(ivec2 p) {
    return texelFetch(uMixedPosition, p, 0);
}

vec4 controlAt(ivec2 p) {
    return texelFetch(uMixedControl, p, 0);
}

int surfaceKeyAt(ivec2 p) {
    return int(floor(controlAt(p).w + 0.5));
}

float gaussianSpatial(vec2 d, float sigma) {
    float s = max(sigma, 0.75);
    return exp(-0.5 * dot(d, d) / (s * s));
}

float surfaceMask(ivec2 p, ivec2 size, int key) {
    p = clampPixel(p, size);
    return surfaceKeyAt(p) == key ? 1.0 : 0.0;
}

void estimateEdgeFrame(ivec2 center, ivec2 size, int key, out vec2 edgeNormal, out float edgeStrength) {
    // Average perpendicular triplets before differentiating. A raw 1-pixel mask
    // gradient would mostly estimate the dither pattern itself rather than a real
    // silhouette/depth-separated support boundary.
    float left = 0.0;
    float right = 0.0;
    float down = 0.0;
    float up = 0.0;
    for (int i = -1; i <= 1; ++i) {
        left += surfaceMask(center + ivec2(-1, i), size, key);
        right += surfaceMask(center + ivec2(1, i), size, key);
        down += surfaceMask(center + ivec2(i, -1), size, key);
        up += surfaceMask(center + ivec2(i, 1), size, key);
    }
    left /= 3.0;
    right /= 3.0;
    down /= 3.0;
    up /= 3.0;
    vec2 g = 0.5 * vec2(right - left, up - down);
    float lenG = length(g);
    edgeStrength = smoothstep(0.10, 0.48, lenG);
    edgeNormal = lenG > 1e-5 ? g / lenG : vec2(1.0, 0.0);
}

float anisotropicSpatial(vec2 d, float sigma, vec2 edgeNormal, float edgeStrength) {
    vec2 tangent = vec2(-edgeNormal.y, edgeNormal.x);
    float along = dot(d, tangent);
    float across = dot(d, edgeNormal);
    float sigmaAlong = mix(sigma, sigma * 1.40, edgeStrength);
    float sigmaAcross = mix(sigma, max(0.45, sigma * 0.32), edgeStrength);
    float q = along * along / max(sigmaAlong * sigmaAlong, 1e-4)
        + across * across / max(sigmaAcross * sigmaAcross, 1e-4);
    return exp(-0.5 * q);
}

float robustScalarWeight(float delta, float sigma) {
    float s = max(sigma, 1e-4);
    float x = abs(delta) / s;
    return 1.0 / (1.0 + x * x);
}

float safeLogit(float a) {
    float x = clamp(a, 1e-5, 1.0 - 1e-5);
    return log(x / (1.0 - x));
}

float logistic(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float balancedLogitScale() {
    return 1.0 / (1.0 + max(uCoverageBias, 0.0));
}

float decodeHighAlphaPower(float beta) {
    const float pivot = 0.5;
    float p = max(uCoverageBias, 1.0);
    float b = clamp(beta, 0.0, 1.0);
    if (b <= pivot || p <= 1.0001) return b;
    float t = clamp((b - pivot) / (1.0 - pivot), 0.0, 1.0);
    return pivot + (1.0 - pivot) * pow(t, 1.0 / p);
}

float decodeCoverageAlpha(float encodedCoverage) {
    float c = clamp(encodedCoverage, 0.0, 1.0);
    if (uCoverageEncodingMode == 1) return logistic(safeLogit(c) - uCoverageBias);
    if (uCoverageEncodingMode == 2) return logistic(safeLogit(c) + uCoverageBias);
    if (uCoverageEncodingMode == 4) return logistic(safeLogit(c) / balancedLogitScale());
    if (uCoverageEncodingMode == 5) return decodeHighAlphaPower(c);
    // Complementary +/- coding is intentionally not used by the publication candidate.
    // Leaving it in encoded-coverage space is safer than pretending a single local inverse.
    return c;
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
    if (uCoverageEncodingMode == 3 && mode != 2) baseRank = complementaryBayerRank(phase, size, baseRank);
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

float measurementThreshold(ivec2 p, int key) {
    uint seed = uint(max(key, 1));
    return uDitherMode == 1 ? stableHashThreshold(p, seed) : orderedThreshold(p, seed, uDitherMode);
}

void clearOutput() {
    outAlbedoRoughness = vec4(0.0, 0.0, 0.0, 0.5);
    outNormalValid = vec4(0.5, 0.5, 1.0, 0.0);
    outPositionValid = vec4(0.0);
    outControl = vec4(0.0);
}

int candidateIndex(int key, int count, int keys[MAX_CANDIDATES]) {
    for (int i = 0; i < MAX_CANDIDATES; ++i) {
        if (i >= count) break;
        if (keys[i] == key) return i;
    }
    return -1;
}

void gatherCandidates(
    ivec2 center,
    ivec2 size,
    int radius,
    out int count,
    out int keys[MAX_CANDIDATES],
    out float scores[MAX_CANDIDATES],
    out float depthSums[MAX_CANDIDATES],
    out ivec2 anchors[MAX_CANDIDATES],
    out float totalSpatialWeight
) {
    count = 0;
    totalSpatialWeight = 0.0;
    for (int i = 0; i < MAX_CANDIDATES; ++i) {
        keys[i] = 0;
        scores[i] = 0.0;
        depthSums[i] = 0.0;
        anchors[i] = center;
    }

    float anchorDistance[MAX_CANDIDATES];
    for (int i = 0; i < MAX_CANDIDATES; ++i) anchorDistance[i] = 1e30;

    float sigma = max(float(radius) * 0.95, 1.15);
    for (int y = -MAX_RADIUS; y <= MAX_RADIUS; ++y) {
        for (int x = -MAX_RADIUS; x <= MAX_RADIUS; ++x) {
            if (!withinKernel(x, y, radius)) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            float spatial = gaussianSpatial(vec2(x, y), sigma);
            totalSpatialWeight += spatial;

            int key = surfaceKeyAt(p);
            if (key <= 0) continue;
            int index = candidateIndex(key, count, keys);
            if (index < 0 && count < MAX_CANDIDATES) {
                index = count;
                keys[index] = key;
                count += 1;
            }
            if (index < 0) continue;

            scores[index] += spatial;
            depthSums[index] += controlAt(p).x * spatial;
            float dist2 = float(x * x + y * y);
            if (dist2 < anchorDistance[index]) {
                anchorDistance[index] = dist2;
                anchors[index] = p;
            }
        }
    }
}

void selectFrontTwo(
    int count,
    int keys[MAX_CANDIDATES],
    float scores[MAX_CANDIDATES],
    float depthSums[MAX_CANDIDATES],
    float totalSpatialWeight,
    out int frontIndex,
    out int backIndex
) {
    frontIndex = -1;
    backIndex = -1;
    float frontDepth = 1e30;
    float backDepth = 1e30;
    float minScore = max(totalSpatialWeight * 0.025, 0.02);

    for (int i = 0; i < MAX_CANDIDATES; ++i) {
        if (i >= count) break;
        if (keys[i] <= 0 || scores[i] < minScore) continue;
        float meanDepth = depthSums[i] / max(scores[i], EPS);
        if (meanDepth < frontDepth) {
            backDepth = frontDepth;
            backIndex = frontIndex;
            frontDepth = meanDepth;
            frontIndex = i;
        } else if (meanDepth < backDepth) {
            backDepth = meanDepth;
            backIndex = i;
        }
    }
}

void main() {
    ivec2 size = textureSize(uMixedPosition, 0);
    ivec2 center = clampPixel(ivec2(gl_FragCoord.xy), size);
    int radius = clamp(uKernelRadius, 1, MAX_RADIUS);

    int count;
    int keys[MAX_CANDIDATES];
    float scores[MAX_CANDIDATES];
    float depthSums[MAX_CANDIDATES];
    ivec2 anchors[MAX_CANDIDATES];
    float totalSpatialWeight;
    gatherCandidates(center, size, radius, count, keys, scores, depthSums, anchors, totalSpatialWeight);

    int frontIndex;
    int backIndex;
    selectFrontTwo(count, keys, scores, depthSums, totalSpatialWeight, frontIndex, backIndex);

    int selectedIndex = uLayerIndex == 0 ? frontIndex : backIndex;
    if (selectedIndex < 0) {
        clearOutput();
        return;
    }

    int key = keys[selectedIndex];
    ivec2 anchor = anchors[selectedIndex];
    vec4 refAlbedoRoughness = albedoAt(anchor);
    vec3 refNormal = normalAt(anchor);
    vec4 refPosition = positionAt(anchor);
    float refDepth = controlAt(anchor).x;

    vec2 edgeNormal;
    float edgeStrength;
    estimateEdgeFrame(center, size, key, edgeNormal, edgeStrength);

    vec3 sumAlbedo = vec3(0.0);
    float sumRoughness = 0.0;
    vec3 sumNormal = vec3(0.0);
    vec3 sumPosition = vec3(0.0);
    float sumWeights = 0.0;
    float sampleCount = 0.0;
    float sigma = max(float(radius) * 0.95, 1.0);
    float depthSigma = max(uDepthSigma, 0.02);
    float albedoSigma = max(uAlbedoSigma, 0.04);

    mat3 moment = mat3(0.0);
    vec3 rhsPosX = vec3(0.0);
    vec3 rhsPosY = vec3(0.0);
    vec3 rhsPosZ = vec3(0.0);
    vec3 rhsAlbR = vec3(0.0);
    vec3 rhsAlbG = vec3(0.0);
    vec3 rhsAlbB = vec3(0.0);
    vec3 rhsRoughness = vec3(0.0);
    vec3 minAlbedo = vec3(1e30);
    vec3 maxAlbedo = vec3(-1e30);
    float minRoughness = 1e30;
    float maxRoughness = -1e30;

    for (int y = -MAX_RADIUS; y <= MAX_RADIUS; ++y) {
        for (int x = -MAX_RADIUS; x <= MAX_RADIUS; ++x) {
            if (!withinKernel(x, y, radius)) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            if (surfaceKeyAt(p) != key) continue;

            vec4 localPosition = positionAt(p);
            vec4 localAlbedoRoughness = albedoAt(p);
            vec3 localNormal = normalAt(p);
            float depthDelta = abs(controlAt(p).x - refDepth);
            float depthWeight = robustScalarWeight(depthDelta, depthSigma);
            vec3 albedoDelta = localAlbedoRoughness.rgb - refAlbedoRoughness.rgb;
            float albedoDistance = length(albedoDelta);
            float albedoWeight = robustScalarWeight(albedoDistance, albedoSigma * 1.75);
            float normalAgreement = max(dot(localNormal, refNormal), 0.0);
            float normalWeight = smoothstep(0.45, 0.98, normalAgreement);
            float spatial = anisotropicSpatial(vec2(x, y), sigma, edgeNormal, edgeStrength);
            float w = spatial
                * mix(0.12, 1.0, depthWeight)
                * mix(0.35, 1.0, albedoWeight)
                * mix(0.18, 1.0, normalWeight);

            sumAlbedo += localAlbedoRoughness.rgb * w;
            sumRoughness += localAlbedoRoughness.a * w;
            sumNormal += localNormal * w;
            sumPosition += localPosition.xyz * w;
            sumWeights += w;
            sampleCount += 1.0;

            vec3 phi = vec3(1.0, float(x), float(y));
            moment += w * outerProduct(phi, phi);
            rhsPosX += phi * (localPosition.x * w);
            rhsPosY += phi * (localPosition.y * w);
            rhsPosZ += phi * (localPosition.z * w);
            rhsAlbR += phi * (localAlbedoRoughness.r * w);
            rhsAlbG += phi * (localAlbedoRoughness.g * w);
            rhsAlbB += phi * (localAlbedoRoughness.b * w);
            rhsRoughness += phi * (localAlbedoRoughness.a * w);
            minAlbedo = min(minAlbedo, localAlbedoRoughness.rgb);
            maxAlbedo = max(maxAlbedo, localAlbedoRoughness.rgb);
            minRoughness = min(minRoughness, localAlbedoRoughness.a);
            maxRoughness = max(maxRoughness, localAlbedoRoughness.a);
        }
    }

    if (sumWeights <= EPS) {
        clearOutput();
        return;
    }

    float observedFront = frontIndex >= 0 ? clamp(scores[frontIndex] / max(totalSpatialWeight, EPS), 0.0, 1.0) : 0.0;
    float observedSelected = clamp(scores[selectedIndex] / max(totalSpatialWeight, EPS), 0.0, 1.0);

    // Censored ordinal inversion. The selected sample label encodes inequalities on
    // each layer's *encoded* alpha. Front-visible pixels censor the back layer; pixels
    // where the front fails expose a direct ordered/hash threshold observation of the
    // back layer. Surface-decorrelated hash thresholds make that uncensored subset span
    // the back-layer threshold domain much more uniformly than translated Bayer ranks.
    float frontLower = 0.0;
    float frontUpper = 1.0;
    float backLower = 0.0;
    float backUpper = 1.0;
    float frontEvidence = 0.0;
    float backEvidence = 0.0;
    int frontKey = frontIndex >= 0 ? keys[frontIndex] : 0;
    int backKey = backIndex >= 0 ? keys[backIndex] : 0;

    for (int y = -MAX_RADIUS; y <= MAX_RADIUS; ++y) {
        for (int x = -MAX_RADIUS; x <= MAX_RADIUS; ++x) {
            if (!withinKernel(x, y, radius)) continue;
            ivec2 p = clampPixel(center + ivec2(x, y), size);
            int observedKey = surfaceKeyAt(p);
            float spatial = gaussianSpatial(vec2(x, y), sigma);

            if (frontKey > 0) {
                float tauFront = measurementThreshold(p, frontKey);
                if (observedKey == frontKey) {
                    frontLower = max(frontLower, tauFront);
                    frontEvidence += spatial;
                    continue; // back visibility is censored by the front survivor.
                }
                if (observedKey == backKey || observedKey == 0) {
                    frontUpper = min(frontUpper, tauFront);
                    frontEvidence += spatial;
                }
            }

            if (backKey > 0 && observedKey != frontKey) {
                float tauBack = measurementThreshold(p, backKey);
                if (observedKey == backKey) {
                    backLower = max(backLower, tauBack);
                    backEvidence += spatial;
                } else if (observedKey == 0) {
                    backUpper = min(backUpper, tauBack);
                    backEvidence += spatial;
                }
            }
        }
    }

    float frontEncoded = clamp(0.5 * (frontLower + frontUpper), 0.0, 1.0);
    float backEncoded = clamp(0.5 * (backLower + backUpper), 0.0, 1.0);
    // At silhouettes the ordinal model is intentionally conservative. Blend toward
    // the local occupancy estimate when the inequality set is weak or contradictory.
    if (frontLower > frontUpper || frontEvidence < 0.35) frontEncoded = observedFront;
    if (backKey > 0 && (backLower > backUpper || backEvidence < 0.35)) {
        float uncensored = max(totalSpatialWeight - scores[frontIndex], EPS);
        backEncoded = clamp(scores[backIndex] / uncensored, 0.0, 1.0);
    }

    float encodedCoverage = uLayerIndex == 0 ? frontEncoded : backEncoded;
    float ordinalEvidence = uLayerIndex == 0 ? frontEvidence : backEvidence;
    float support = clamp(sampleCount / max(float((2 * radius + 1) * (2 * radius + 1)) * 0.28, 1.0), 0.0, 1.0);
    support *= clamp(0.45 + 0.55 * ordinalEvidence / max(totalSpatialWeight, EPS), 0.0, 1.0);

    // The interval inversion carries the useful ordering information, but on ordered
    // patterns it can also retain phase-locked variation. Blend it toward the local
    // phase-neutral occupancy estimate according to confidence.
    float inversionTrust = clamp(ordinalEvidence / max(totalSpatialWeight, EPS), 0.0, 1.0);
    inversionTrust *= clamp(sampleCount / 4.0, 0.0, 1.0);
    if (uDitherMode != 1) inversionTrust *= 0.85;
    // Experimental exposure-aware ordinal blend. Identical to Decoupled except
    // the back-layer prior is conditioned on the observed non-front support.
    // Do not change the historical blend, confidence or layer existence rules.
    float occupancyPrior = observedSelected;
    if (uOpacityEstimator == 3 && uLayerIndex == 1 && frontIndex >= 0) {
        occupancyPrior = clamp(scores[selectedIndex] / max(totalSpatialWeight - scores[frontIndex], EPS), 0.0, 1.0);
    }
    encodedCoverage = mix(occupancyPrior, encodedCoverage, clamp(0.35 + 0.50 * inversionTrust, 0.0, 1.0));

    // Isolated ablations of known-threshold inference. These controls use precisely
    // the same candidates, material reconstruction and compositing as the decoupled
    // decoder. Only the opacity estimator differs.
    // RAW is deliberately exposure-biased when a front surface censors the back.
    // CONDITIONAL corrects that exposure using only observed occupancy, without
    // using any per-pixel threshold inequalities (not an oracle).
    if (uOpacityEstimator == 1) {
        encodedCoverage = observedSelected;
    } else if (uOpacityEstimator == 2) {
        encodedCoverage = uLayerIndex == 0 ? observedFront
            : clamp(scores[backIndex] / max(totalSpatialWeight - scores[frontIndex], EPS), 0.0, 1.0);
    }

    float alphaHat = clamp(decodeCoverageAlpha(encodedCoverage), 0.0, 1.0);
    // Reduce very weak layer hallucinations before composition. This is deliberately
    // conservative for low-support second-layer estimates.
    if (uDecoupleOpacityConfidence == 0) {
        if (uLayerIndex == 1) alphaHat *= smoothstep(0.08, 0.30, support);
        else alphaHat *= mix(0.90, 1.0, smoothstep(0.05, 0.25, support));
    }

    vec3 reconstructedNormal = refNormal;
    float normalLen2 = dot(sumNormal, sumNormal);
    if (normalLen2 > EPS) reconstructedNormal = sumNormal * inversesqrt(normalLen2);

    vec3 reconstructedAlbedo = sumAlbedo / sumWeights;
    float reconstructedRoughness = sumRoughness / sumWeights;
    vec3 reconstructedPosition = sumPosition / sumWeights;

    // Local affine reproduction preserves first-order material/position gradients instead
    // of flattening them with a normalized convolution. The fit is only trusted when the
    // weighted 3x3 moment system is well-conditioned enough for this small support.
    float detMoment = determinant(moment);
    float fitConfidence = smoothstep(1e-4, 2e-2, abs(detMoment)) * smoothstep(2.5, 6.0, sampleCount);
    if (fitConfidence > 0.0) {
        mat3 invMoment = inverse(moment);
        vec3 posFit = vec3(
            (invMoment * rhsPosX).x,
            (invMoment * rhsPosY).x,
            (invMoment * rhsPosZ).x
        );
        vec3 albedoFit = vec3(
            (invMoment * rhsAlbR).x,
            (invMoment * rhsAlbG).x,
            (invMoment * rhsAlbB).x
        );
        float roughnessFit = (invMoment * rhsRoughness).x;

        albedoFit = clamp(albedoFit, minAlbedo, maxAlbedo);
        roughnessFit = clamp(roughnessFit, minRoughness, maxRoughness);
        // Position is the attribute that benefits most directly from a plane/affine fit.
        reconstructedPosition = mix(reconstructedPosition, posFit, 0.85 * fitConfidence);
        reconstructedAlbedo = mix(reconstructedAlbedo, albedoFit, 0.62 * fitConfidence);
        reconstructedRoughness = mix(reconstructedRoughness, roughnessFit, 0.45 * fitConfidence);
    }

    // Geometry-consistency test at the target pixel. If the inferred encoded alpha says
    // the selected surface should have survived its known threshold but it is absent at
    // the center, the missing sample is more likely a silhouette/depth boundary than an
    // ordinary dither hole. Back-layer evidence is skipped when the front layer censored it.
    int centerObservedKey = surfaceKeyAt(center);
    bool centerCensored = uLayerIndex == 1 && frontKey > 0 && centerObservedKey == frontKey;
    float geometryConfidence = 1.0;
    if (centerObservedKey != key && !centerCensored) {
        float centerThreshold = measurementThreshold(center, key);
        float contradiction = smoothstep(centerThreshold + 0.025, centerThreshold + 0.12, encodedCoverage);
        geometryConfidence *= 1.0 - 0.92 * contradiction;
    }
    geometryConfidence *= mix(0.72, 1.0, support);
    if (uDecoupleOpacityConfidence == 0) alphaHat *= geometryConfidence;
    // In the decoupled ablation, confidence remains an independent diagnostic
    // and a hypothesis-existence guard below. Never interpret uncertain alpha
    // as systematically smaller physical opacity.

    // Suppress extremely weak second-layer hallucinations. High-alpha reveal companding
    // is designed specifically to raise this support when the front layer is dense.
    if ((uLayerIndex == 1 && (observedSelected < 0.02 || support < 0.10))
        || (support < 0.04 && alphaHat < 0.10) || alphaHat < 0.008) {
        clearOutput();
        return;
    }

    outAlbedoRoughness = vec4(reconstructedAlbedo, reconstructedRoughness);
    outNormalValid = vec4(reconstructedNormal * 0.5 + 0.5, 1.0);
    outPositionValid = vec4(reconstructedPosition, 1.0);
    // x = observed selected-layer coverage before censoring inversion
    // y = reconstructed physical alpha
    // z = attribute support/confidence proxy including geometry consistency
    // w = inferred transparent surface id
    outControl = vec4(observedSelected, alphaHat, support * geometryConfidence, float(key));
}
