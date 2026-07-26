#version 450

layout(set = 0, binding = 0) uniform sampler2D rawMoments;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec2 outMoments;

void main() {
    const vec2 texelSize = 1.0 / vec2(textureSize(rawMoments, 0));
    vec2 filtered = vec2(0.0);
    float weightSum = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            const float weight = 1.0 / (1.0 + float(x * x + y * y));
            filtered += texture(rawMoments, inUv + vec2(x, y) * texelSize).rg * weight;
            weightSum += weight;
        }
    }
    outMoments = filtered / weightSum;
}
