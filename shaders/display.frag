#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D uImage;
uniform int uToneMap;

vec3 tonemap(vec3 x) {
    x = max(x, vec3(0.0));
    return pow(x / (1.0 + x), vec3(1.0 / 2.2));
}

void main() {
    vec3 c = texture(uImage, vUv).rgb;
    outColor = vec4(uToneMap != 0 ? tonemap(c) : c, 1.0);
}
