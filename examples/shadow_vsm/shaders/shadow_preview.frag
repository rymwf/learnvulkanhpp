#version 450

layout(set = 0, binding = 0) uniform sampler2D rawMoments;
layout(set = 0, binding = 1) uniform sampler2D filteredMoments;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

void main() {
    const bool showRawMoments = inUv.x < 0.5;
    const vec2 momentUv = vec2(showRawMoments ? inUv.x * 2.0 : inUv.x * 2.0 - 1.0, inUv.y);
    const float depth = (showRawMoments ? texture(rawMoments, momentUv) : texture(filteredMoments, momentUv)).r;
    outColor = vec4(vec3(depth), 1.0);
}
