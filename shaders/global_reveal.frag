#version 460 core

in VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} v;

layout(location = 0) out float outReveal;
layout(binding = 0) uniform sampler2D uOpaqueDepth;
layout(binding = 3) uniform sampler2D uBaseTexture;

uniform float uAlpha;
uniform int uAlphaMode;
uniform int uUseBaseTexture;

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
    if (gl_FragCoord.z >= texelFetch(uOpaqueDepth, pixel, 0).r - 1e-7) discard;
    float alpha = materialAlpha(v.uv);
    if (alpha <= 0.001) discard;
    outReveal = alpha;
}
