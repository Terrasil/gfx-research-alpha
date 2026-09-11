#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAlphaMeta;

layout(binding = 0) uniform sampler2D uOpaque;
layout(binding = 1) uniform sampler2D uAccum;
layout(binding = 2) uniform sampler2D uReveal;

void main() {
    vec3 background = texture(uOpaque, vUv).rgb;
    vec4 accum = texture(uAccum, vUv);
    float reveal = clamp(texture(uReveal, vUv).r, 0.0, 1.0);
    if (accum.a <= 1e-7) {
        outColor = vec4(background, 1.0);
        outAlphaMeta = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3 average = accum.rgb / max(accum.a, 1e-7);
    float alpha = 1.0 - reveal;
    outColor = vec4(mix(background, average, alpha), 1.0);
    outAlphaMeta = vec4(alpha, 1.0, alpha, 1.0);
}
