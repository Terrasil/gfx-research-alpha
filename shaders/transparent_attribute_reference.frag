#version 460 core

in VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} v;

layout(location = 0) out vec4 outAlbedoRoughness;
layout(location = 1) out vec4 outNormalValid;
layout(location = 2) out vec4 outPositionValid;
layout(location = 3) out vec4 outControl;

layout(binding = 1) uniform sampler2D uOpaqueDepth;
layout(binding = 2) uniform sampler2D uPrevDepth;
layout(binding = 3) uniform sampler2D uBaseTexture;

uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uAlpha;
uniform int uAlphaMode;
uniform int uUseBaseTexture;
uniform uint uSurfaceId;
uniform int uFirstLayer;

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

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float opaqueDepth = texelFetch(uOpaqueDepth, pixel, 0).r;
    if (gl_FragCoord.z >= opaqueDepth - 1e-7) discard;

    if (uFirstLayer == 0) {
        float prevDepth = texelFetch(uPrevDepth, pixel, 0).r;
        if (prevDepth >= 1.0 - 1e-7 || gl_FragCoord.z <= prevDepth + 2e-7) discard;
    }

    float alpha = materialAlpha(v.uv);
    if (alpha <= 0.001) discard;

    vec3 textureColor = uUseBaseTexture != 0 ? texture(uBaseTexture, v.uv).rgb : vec3(1.0);
    outAlbedoRoughness = vec4(textureColor * uBaseColor, clamp(uRoughness, 0.02, 1.0));
    outNormalValid = vec4(normalize(v.worldNormal) * 0.5 + 0.5, 1.0);
    outPositionValid = vec4(v.worldPosition, 1.0);
    // x = exact peel depth, y = physical alpha, z = valid/reference confidence, w = surface id.
    outControl = vec4(gl_FragCoord.z, alpha, 1.0, float(uSurfaceId));
}
