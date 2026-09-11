#version 460 core

layout(triangles, invocations = 8) in;
layout(triangle_strip, max_vertices = 3) out;

in VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} gIn[];

out GS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
    flat int layerIndex;
} g;

void main() {
    for (int i = 0; i < 3; ++i) {
        gl_Position = gl_in[i].gl_Position;
        gl_Layer = gl_InvocationID;
        g.worldPosition = gIn[i].worldPosition;
        g.worldNormal = gIn[i].worldNormal;
        g.uv = gIn[i].uv;
        g.layerIndex = gl_InvocationID;
        EmitVertex();
    }
    EndPrimitive();
}
