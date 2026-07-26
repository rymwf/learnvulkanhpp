#version 450

layout(location = 0) out vec2 outMoments;

void main() {
    const float depth = gl_FragCoord.z;
    outMoments = vec2(depth, depth * depth);
}
