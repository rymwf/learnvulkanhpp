#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec3 inColor[];
layout(location = 0) out vec3 outColor;

const vec2 offsets[3] = vec2[](
    vec2(-0.18, 0.16),
    vec2(0.18, 0.16),
    vec2(0.0, -0.18)
);

void main() {
    for (int index = 0; index < 3; ++index) {
        gl_Position = gl_in[0].gl_Position + vec4(offsets[index], 0.0, 0.0);
        outColor = inColor[0];
        EmitVertex();
    }
    EndPrimitive();
}
