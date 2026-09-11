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

vec3 over(vec3 background, vec3 foreground, float alpha) {
    return mix(background, foreground, clamp(alpha, 0.0, 1.0));
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 opaqueColor = texelFetch(uOpaqueLit, pixel, 0).rgb;

    vec4 frontControl = texelFetch(uFrontControl, pixel, 0);
    vec4 backControl = texelFetch(uBackControl, pixel, 0);
    vec4 frontPosition = texelFetch(uFrontPosition, pixel, 0);
    vec4 backPosition = texelFetch(uBackPosition, pixel, 0);

    bool frontValid = frontPosition.a > 0.5 && frontControl.w > 0.5 && frontControl.y > 0.008;
    bool backValid = backPosition.a > 0.5 && backControl.w > 0.5 && backControl.y > 0.008;

    // The edge-aware attribute decoder has already reconstructed and gated the
    // surface fields. Do not blur lit radiance again here: final lighting/detail
    // should be created from reconstructed attributes and preserved through composition.
    vec3 color = opaqueColor;
    float alphaBack = backValid ? clamp(backControl.y, 0.0, 1.0) : 0.0;
    float alphaFront = frontValid ? clamp(frontControl.y, 0.0, 1.0) : 0.0;

    if (backValid) color = over(color, texelFetch(uBackLit, pixel, 0).rgb, alphaBack);
    if (frontValid) color = over(color, texelFetch(uFrontLit, pixel, 0).rgb, alphaFront);

    float totalAlpha = 1.0 - (1.0 - alphaFront) * (1.0 - alphaBack);
    float support = max(frontValid ? frontControl.z : 0.0, backValid ? backControl.z : 0.0);
    float coverage = max(frontValid ? frontControl.x : 0.0, backValid ? backControl.x : 0.0);

    outColor = vec4(color, 1.0);
    outAlphaMeta = vec4(coverage, support, alphaBack, totalAlpha);
}
