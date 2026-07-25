#version 450

vec2 positions[3] = vec2[](
    vec2(-0.55, 0.0),
    vec2(0.0, 0.0),
    vec2(0.55, 0.0)
);

vec3 colors[3] = vec3[](
    vec3(0.95, 0.2, 0.2),
    vec3(0.2, 0.85, 0.35),
    vec3(0.2, 0.45, 1.0)
);

layout(location = 0) out vec3 outColor;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    outColor = colors[gl_VertexIndex];
}
