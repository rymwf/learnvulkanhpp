#version 450

layout(set = 0, binding = 1) uniform sampler2D shadowDepth;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

void main() {
    const float depth = texture(shadowDepth, inUv).r;
    outColor = vec4(vec3(depth), 1.0);
}
