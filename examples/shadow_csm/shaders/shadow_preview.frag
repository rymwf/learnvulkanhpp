#version 450

layout(set = 0, binding = 2) uniform sampler2DArray shadowDepth;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

void main() {
    const float cascadeCoordinate = inUv.x * 3.0;
    const float cascadeIndex = min(floor(cascadeCoordinate), 2.0);
    const vec2 shadowUv = vec2(fract(cascadeCoordinate), inUv.y);
    const float depth = texture(shadowDepth, vec3(shadowUv, cascadeIndex)).r;
    const float divider = step(0.985, fract(cascadeCoordinate));
    outColor = mix(vec4(vec3(depth), 1.0), vec4(1.0), divider);
}
