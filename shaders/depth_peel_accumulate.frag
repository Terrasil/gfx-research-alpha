#version 460 core

in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D uLayer;

void main() {
    vec4 layer = texture(uLayer, vUv);
    if (layer.a <= 1e-7) discard;
    outColor = vec4(layer.rgb * layer.a, layer.a);
}
