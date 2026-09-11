#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D uAccum;
layout(binding = 1) uniform sampler2D uOpaque;

void main() {
    vec4 accum = texture(uAccum, vUv);
    vec3 opaque = texture(uOpaque, vUv).rgb;
    vec3 color = accum.rgb + (1.0 - clamp(accum.a, 0.0, 1.0)) * opaque;
    outColor = vec4(color, 1.0);
}
