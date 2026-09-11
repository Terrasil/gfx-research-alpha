#version 460 core

in GS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
    flat int layerIndex;
} g;

layout(location = 0) out vec4 outAccum;
layout(location = 1) out float outReveal;

layout(binding = 0) uniform sampler2D uOpaqueDepth;
layout(binding = 1) uniform sampler2D uFrontDepth;
layout(binding = 3) uniform sampler2D uBaseTexture;

uniform vec3 uEye;
uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uAlpha;
uniform int uAlphaMode;
uniform int uLightCount;
uniform int uUseBaseTexture;
uniform float uDepthExponent;
uniform float uNearPlane;
uniform float uFarPlane;
uniform float uLayerDepthSpan;
uniform int uBinCount;

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

vec3 lightPosition(int i) {
    float fi = float(i);
    float a = fi * 2.39996323;
    float ring = 2.8 + float(i % 7) * 0.72;
    return vec3(cos(a) * ring, 1.2 + float(i % 5) * 0.64, sin(a) * ring - 1.2);
}

vec3 lightColor(int i) {
    float fi = float(i);
    return 0.65 + 0.35 * vec3(
        0.5 + 0.5 * sin(fi * 1.71 + 0.2),
        0.5 + 0.5 * sin(fi * 1.13 + 2.1),
        0.5 + 0.5 * sin(fi * 0.91 + 4.0)
    );
}

vec3 shade(vec3 albedo, vec3 normal, vec3 worldPosition, float roughness) {
    vec3 N = normalize(normal);
    vec3 V = normalize(uEye - worldPosition);
    vec3 result = albedo * 0.028;
    for (int i = 0; i < uLightCount; ++i) {
        vec3 toLight = lightPosition(i) - worldPosition;
        float distanceSquared = max(dot(toLight, toLight), 0.05);
        vec3 L = toLight * inversesqrt(distanceSquared);
        float ndotl = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float exponent = mix(96.0, 8.0, roughness);
        float specular = pow(max(dot(N, H), 0.0), exponent) * mix(0.55, 0.08, roughness);
        float attenuation = (4.5 / sqrt(float(max(uLightCount, 1)))) / (1.0 + 0.28 * distanceSquared);
        vec3 radiance = lightColor(i) * attenuation;
        result += albedo * radiance * ndotl + radiance * specular;
    }
    return result;
}

float wboitWeight(float alpha, float depth) {
    float alphaWeight = clamp(alpha * 8.0 + 0.01, 0.01, 8.0);
    float depthWeight = pow(clamp(1.0 - depth, 0.02, 1.0), uDepthExponent);
    return clamp(alphaWeight * mix(0.35, 1.0, depthWeight), 0.01, 8.0);
}

void main() {
    if (g.layerIndex >= uBinCount) discard;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float opaqueDepth = texelFetch(uOpaqueDepth, pixel, 0).r;
    if (gl_FragCoord.z >= opaqueDepth - 1e-7) discard;

    float alpha = materialAlpha(g.uv);
    if (alpha <= 0.001) discard;

    float frontDistance = texelFetch(uFrontDepth, pixel, 0).r;
    float distanceToEye = length(g.worldPosition - uEye);
    if (frontDistance <= 0.0) frontDistance = uNearPlane;
    float span = max(min(uLayerDepthSpan, uFarPlane - frontDistance), 1e-3);
    float binWidth = span / float(max(uBinCount, 1));
    float center = frontDistance + (float(g.layerIndex) + 0.5) * binWidth;
    float normalizedDistance = abs(distanceToEye - center) / max(0.5 * binWidth, 1e-4);
    float binWeight = exp(-pow(0.832555 * normalizedDistance, 2.0));
    if (binWeight < 1e-4) discard;

    vec3 base = uBaseColor * (uUseBaseTexture != 0 ? texture(uBaseTexture, g.uv).rgb : vec3(1.0));
    vec3 color = shade(base, normalize(g.worldNormal), g.worldPosition, uRoughness);
    float weightedAlpha = clamp(alpha * binWeight, 0.0, 1.0);
    float weight = wboitWeight(alpha, gl_FragCoord.z) * binWeight;
    outAccum = vec4(color * alpha, alpha) * weight;
    outReveal = weightedAlpha;
}
