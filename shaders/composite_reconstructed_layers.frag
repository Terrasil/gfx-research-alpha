#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAlphaMeta;

layout(binding = 0) uniform sampler2D uOpaqueLit;
layout(binding = 1) uniform sampler2D uFrontLit;
layout(binding = 2) uniform sampler2D uFrontPosition;
layout(binding = 3) uniform sampler2D uFrontControl;
layout(binding = 4) uniform sampler2D uBackLit;
layout(binding = 5) uniform sampler2D uBackPosition;
layout(binding = 6) uniform sampler2D uBackControl;

const float EPS = 1e-6;
const float DEPTH_SIGMA = 0.020;

struct LayerSample {
    vec3 color;
    float alpha;
    float coverage;
    float support;
    float depth;
    float key;
    bool valid;
};

vec3 over(vec3 background, vec3 foreground, float alpha) {
    return mix(background, foreground, clamp(alpha, 0.0, 1.0));
}

LayerSample fetchLayer(sampler2D litTex, sampler2D positionTex, sampler2D controlTex, ivec2 pixel) {
    ivec2 size = textureSize(positionTex, 0);
    ivec2 p = clamp(pixel, ivec2(0), size - 1);
    vec4 positionData = texelFetch(positionTex, p, 0);
    vec4 control = texelFetch(controlTex, p, 0);

    LayerSample s;
    s.color = texelFetch(litTex, p, 0).rgb;
    s.alpha = clamp(control.y, 0.0, 1.0);
    s.coverage = clamp(control.x, 0.0, 1.0);
    s.support = clamp(control.z, 0.0, 1.0);
    s.key = control.w;
    s.depth = positionData.a > 0.5 ? positionData.z : 0.0;
    s.valid = positionData.a > 0.5 && control.w > 0.5 && s.alpha > 1e-4;
    return s;
}

float spatialWeight(int dx, int dy) {
    return exp(-0.55 * float(dx * dx + dy * dy));
}

LayerSample invalidLayer() {
    LayerSample s;
    s.color = vec3(0.0);
    s.alpha = 0.0;
    s.coverage = 0.0;
    s.support = 0.0;
    s.depth = 0.0;
    s.key = 0.0;
    s.valid = false;
    return s;
}

LayerSample resolveLayer(sampler2D litTex, sampler2D positionTex, sampler2D controlTex, ivec2 pixel) {
    LayerSample center = fetchLayer(litTex, positionTex, controlTex, pixel);

    float targetKey = center.valid ? center.key : 0.0;
    float targetDepth = center.depth;

    if (!center.valid) {
        float candidateKeys[9];
        float candidateScores[9];
        float candidateDepths[9];
        int candidateCount = 0;
        for (int i = 0; i < 9; ++i) {
            candidateKeys[i] = 0.0;
            candidateScores[i] = 0.0;
            candidateDepths[i] = 0.0;
        }

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                LayerSample n = fetchLayer(litTex, positionTex, controlTex, pixel + ivec2(dx, dy));
                if (!n.valid) continue;
                float w = spatialWeight(dx, dy) * mix(0.35, 1.0, n.support);
                int index = -1;
                for (int i = 0; i < candidateCount; ++i) {
                    if (abs(candidateKeys[i] - n.key) < 0.5) {
                        index = i;
                        break;
                    }
                }
                if (index < 0 && candidateCount < 9) {
                    index = candidateCount;
                    candidateKeys[index] = n.key;
                    candidateCount += 1;
                }
                if (index >= 0) {
                    candidateScores[index] += w;
                    candidateDepths[index] += n.depth * w;
                }
            }
        }

        float bestScore = 0.0;
        for (int i = 0; i < candidateCount; ++i) {
            if (candidateScores[i] > bestScore) {
                bestScore = candidateScores[i];
                targetKey = candidateKeys[i];
                targetDepth = candidateDepths[i] / max(candidateScores[i], EPS);
            }
        }
        if (targetKey <= 0.5) return invalidLayer();
    }

    vec3 sumColor = vec3(0.0);
    float sumAlpha = 0.0;
    float sumCoverage = 0.0;
    float sumSupport = 0.0;
    float sumDepth = 0.0;
    float sumW = 0.0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            LayerSample n = fetchLayer(litTex, positionTex, controlTex, pixel + ivec2(dx, dy));
            if (!n.valid) continue;
            if (abs(n.key - targetKey) > 0.5) continue;
            float depthDelta = abs(n.depth - targetDepth);
            float depthW = exp(-depthDelta / DEPTH_SIGMA);
            float w = spatialWeight(dx, dy) * depthW * mix(0.35, 1.0, n.support);
            sumColor += n.color * w;
            sumAlpha += n.alpha * w;
            sumCoverage += n.coverage * w;
            sumSupport += n.support * w;
            sumDepth += n.depth * w;
            sumW += w;
        }
    }

    if (sumW <= EPS) return center.valid ? center : invalidLayer();

    LayerSample avg;
    avg.color = sumColor / sumW;
    avg.alpha = sumAlpha / sumW;
    avg.coverage = sumCoverage / sumW;
    avg.support = clamp(sumSupport / sumW, 0.0, 1.0);
    avg.depth = sumDepth / sumW;
    avg.key = targetKey;
    avg.valid = true;

    LayerSample result = avg;
    if (center.valid && abs(center.key - targetKey) < 0.5) {
        float blend = clamp(0.18 + 0.55 * (1.0 - center.support), 0.18, 0.78);
        result.color = mix(center.color, avg.color, blend);
        result.alpha = mix(center.alpha, avg.alpha, blend);
        result.coverage = mix(center.coverage, avg.coverage, blend);
        result.support = max(center.support, avg.support);
        result.depth = mix(center.depth, avg.depth, 0.5 * blend);
    }

    result.alpha *= mix(0.82, 1.0, smoothstep(0.08, 0.35, result.support));
    if (result.alpha < 0.02 || result.support < 0.04) return invalidLayer();
    return result;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 opaqueColor = texelFetch(uOpaqueLit, pixel, 0).rgb;

    LayerSample front = resolveLayer(uFrontLit, uFrontPosition, uFrontControl, pixel);
    LayerSample back = resolveLayer(uBackLit, uBackPosition, uBackControl, pixel);

    if (front.valid && back.valid && abs(front.key - back.key) < 0.5) {
        if (front.depth <= back.depth) {
            back.valid = false;
        } else {
            front.valid = false;
        }
    }

    vec3 color = opaqueColor;
    float alphaFront = front.valid ? clamp(front.alpha, 0.0, 1.0) : 0.0;
    float alphaBack = back.valid ? clamp(back.alpha, 0.0, 1.0) : 0.0;

    if (back.valid) color = over(color, back.color, alphaBack);
    if (front.valid) color = over(color, front.color, alphaFront);

    float totalAlpha = 1.0 - (1.0 - alphaFront) * (1.0 - alphaBack);
    float support = max(front.valid ? front.support : 0.0, back.valid ? back.support : 0.0);
    float coverage = max(front.valid ? front.coverage : 0.0, back.valid ? back.coverage : 0.0);

    outColor = vec4(color, 1.0);
    outAlphaMeta = vec4(coverage, support, alphaBack, totalAlpha);
}
