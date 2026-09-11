#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAlphaMeta;

layout(binding = 0) uniform sampler2D uOpaque;
layout(binding = 1) uniform sampler2DArray uAccum;
layout(binding = 2) uniform sampler2DArray uReveal;
layout(binding = 3) uniform sampler2D uGlobalReveal;
uniform int uBinCount;

void main() {
    vec3 background = texture(uOpaque, vUv).rgb;
    float transmittance = 1.0;
    vec3 weightedColor = vec3(0.0);
    float normalization = 0.0;

    for (int k = 0; k < uBinCount; ++k) {
        vec4 accum = texture(uAccum, vec3(vUv, float(k)));
        float reveal = clamp(texture(uReveal, vec3(vUv, float(k))).r, 0.0, 1.0);
        float alpha = 1.0 - reveal;
        if (accum.a > 1e-7 && alpha > 1e-7) {
            vec3 average = accum.rgb / accum.a;
            float contribution = alpha * transmittance;
            weightedColor += average * contribution;
            normalization += contribution;
        }
        transmittance *= reveal;
    }

    float globalReveal = clamp(texture(uGlobalReveal, vUv).r, 0.0, 1.0);
    float netAlpha = 1.0 - globalReveal;
    vec3 transparentColor = normalization > 1e-7 ? weightedColor / normalization : vec3(0.0);
    outColor = vec4(background * globalReveal + transparentColor * netAlpha, 1.0);
    outAlphaMeta = vec4(netAlpha, netAlpha > 1e-7 ? 1.0 : 0.0, netAlpha, 1.0);
}
