#version 460 core

in VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} v;

layout(location = 0) out vec4 outAlbedoRoughness;
layout(location = 1) out vec4 outNormalValid;
layout(location = 2) out vec4 outPositionValid;

layout(binding = 0) uniform sampler2D uBaseTexture;

uniform vec3 uBaseColor;
uniform float uRoughness;
uniform int uPattern;
uniform int uUseBaseTexture;

vec3 proceduralAlbedo(vec3 baseColor, vec3 p, vec2 uv) {
    if (uPattern == 1) {
        float checker = mod(floor(p.x * 1.25) + floor(p.z * 1.25), 2.0);
        return baseColor * mix(0.72, 1.18, checker);
    }
    if (uPattern == 2) {
        float bands = 0.82 + 0.18 * sin(uv.x * 55.0) * sin(uv.y * 41.0);
        return baseColor * bands;
    }
    return baseColor;
}

void main() {
    vec3 textureColor = uUseBaseTexture != 0 ? texture(uBaseTexture, v.uv).rgb : vec3(1.0);
    vec3 albedo = proceduralAlbedo(textureColor * uBaseColor, v.worldPosition, v.uv);
    outAlbedoRoughness = vec4(albedo, clamp(uRoughness, 0.02, 1.0));
    outNormalValid = vec4(normalize(v.worldNormal) * 0.5 + 0.5, 1.0);
    outPositionValid = vec4(v.worldPosition, 1.0);
}
