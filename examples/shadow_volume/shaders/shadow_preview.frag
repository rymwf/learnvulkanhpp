#version 450

layout(set = 0, binding = 1) uniform sampler2D shadowDepth;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

void main() {
    const vec2 previewMin = vec2(0.734375, 0.0208333333);
    const vec2 previewMax = vec2(0.984375, 0.3541666667);
    const vec2 shadowUv = (inUv - previewMin) / (previewMax - previewMin);
    const float depth = texture(shadowDepth, shadowUv).r;
    outColor = vec4(vec3(depth), 1.0);
}
