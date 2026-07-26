#version 450

layout(location = 0) out vec2 outUv;

const vec2 positions[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0)
);

void main() {
    const vec2 position = positions[gl_VertexIndex];
    gl_Position = vec4(position, 0.0, 1.0);
    outUv = vec2(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
}
