#version 450

layout(set = 0, binding = 0) uniform sampler2D rawMoments;
layout(set = 0, binding = 1) uniform sampler2D filteredMoments;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

void main() {
    const vec2 previewMin = vec2(0.734375, 0.0208333333);
    const vec2 previewMax = vec2(0.984375, 0.3541666667);
    const vec2 previewUv = (inUv - previewMin) / (previewMax - previewMin);
    const bool showRawMoments = previewUv.x < 0.5;
    const vec2 momentUv = vec2(showRawMoments ? previewUv.x * 2.0 : previewUv.x * 2.0 - 1.0, previewUv.y);
    const float depth = (showRawMoments ? texture(rawMoments, momentUv) : texture(filteredMoments, momentUv)).r;
    outColor = vec4(vec3(depth), 1.0);
}
