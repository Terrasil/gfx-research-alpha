#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAlphaMeta;

layout(binding = 0) uniform sampler2D uOpaqueLit;
layout(binding = 1) uniform sampler2D uTransparentLit;
layout(binding = 2) uniform sampler2D uTransparentPosition;
layout(binding = 3) uniform sampler2D uTransparentControl;

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
    s.valid = positionData.a > 0.5 && s.key > 0.5 && s.alpha > 1e-4;
    return s;
}

LayerSample resolveLayer(ivec2 pixel) {
    LayerSample center = fetchLayer(uTransparentLit, uTransparentPosition, uTransparentControl, pixel);
    float targetKey = center.valid ? center.key : 0.0;
    float targetDepth = center.depth;

    if (!center.valid) {
        float bestScore = 0.0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                LayerSample n = fetchLayer(uTransparentLit, uTransparentPosition, uTransparentControl, pixel + ivec2(dx, dy));
                if (!n.valid) continue;
                float w = spatialWeight(dx, dy) * mix(0.35, 1.0, n.support);
                if (w > bestScore) {
                    bestScore = w;
                    targetKey = n.key;
                    targetDepth = n.depth;
                }
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
            LayerSample n = fetchLayer(uTransparentLit, uTransparentPosition, uTransparentControl, pixel + ivec2(dx, dy));
            if (!n.valid) continue;
            if (abs(n.key - targetKey) > 0.5) continue;
            float depthW = exp(-abs(n.depth - targetDepth) / DEPTH_SIGMA);
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

    if (center.valid && abs(center.key - targetKey) < 0.5) {
        float blend = clamp(0.18 + 0.55 * (1.0 - center.support), 0.18, 0.78);
        avg.color = mix(center.color, avg.color, blend);
        avg.alpha = mix(center.alpha, avg.alpha, blend);
        avg.coverage = mix(center.coverage, avg.coverage, blend);
        avg.support = max(center.support, avg.support);
        avg.depth = mix(center.depth, avg.depth, 0.5 * blend);
    }

    avg.alpha *= mix(0.82, 1.0, smoothstep(0.08, 0.35, avg.support));
    if (avg.alpha < 0.02 || avg.support < 0.04) return invalidLayer();
    return avg;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 opaqueColor = texelFetch(uOpaqueLit, pixel, 0).rgb;
    LayerSample transparent = resolveLayer(pixel);

    vec3 color = transparent.valid ? mix(opaqueColor, transparent.color, transparent.alpha) : opaqueColor;
    outColor = vec4(color, 1.0);
    outAlphaMeta = vec4(transparent.valid ? transparent.coverage : 0.0, transparent.valid ? transparent.support : 0.0, 0.0, transparent.valid ? transparent.alpha : 0.0);
}
