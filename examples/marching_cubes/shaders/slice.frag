#version 450

layout(set = 0, binding = 0, r32f) uniform readonly image3D volumeImage;

layout(push_constant) uniform SlicePushConstants {
    float isoLevel;
    uint resolution;
    int axis;
    int sliceIndex;
} pushConstants;

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

vec3 heatMap(float value) {
    const vec3 dark = vec3(0.025, 0.045, 0.075);
    const vec3 cyan = vec3(0.05, 0.68, 0.78);
    const vec3 yellow = vec3(0.95, 0.82, 0.22);
    const vec3 red = vec3(0.92, 0.24, 0.16);
    if (value < 0.45) {
        return mix(dark, cyan, value / 0.45);
    }
    if (value < 0.75) {
        return mix(cyan, yellow, (value - 0.45) / 0.30);
    }
    return mix(yellow, red, (value - 0.75) / 0.25);
}

void main() {
    const int resolution = int(pushConstants.resolution);
    const ivec2 coordinate = clamp(ivec2(inUv * float(resolution + 1)), ivec2(0), ivec2(resolution));
    ivec3 sampleCoordinate;
    if (pushConstants.axis == 0) {
        sampleCoordinate = ivec3(pushConstants.sliceIndex, coordinate.x, coordinate.y);
    } else if (pushConstants.axis == 1) {
        sampleCoordinate = ivec3(coordinate.x, pushConstants.sliceIndex, coordinate.y);
    } else {
        sampleCoordinate = ivec3(coordinate.x, coordinate.y, pushConstants.sliceIndex);
    }

    const float density = imageLoad(volumeImage, sampleCoordinate).r;
    const float normalizedDensity = clamp(density / max(pushConstants.isoLevel * 2.0, 0.001), 0.0, 1.0);
    vec3 color = heatMap(normalizedDensity);
    const float contourDistance = abs(density - pushConstants.isoLevel) / max(pushConstants.isoLevel, 0.001);
    color = mix(vec3(0.96), color, smoothstep(0.015, 0.05, contourDistance));

    const vec2 borderDistance = min(inUv, vec2(1.0) - inUv);
    const float border = 1.0 - smoothstep(0.0, 0.012, min(borderDistance.x, borderDistance.y));
    color = mix(color, vec3(0.82), border * 0.8);
    outColor = vec4(color, 1.0);
}
