#version 450

layout(set = 0, binding = 0) uniform sampler2DArrayShadow shadowMap;

layout(set = 0, binding = 1, std140) uniform CascadeData {
    mat4 lightViewProjections[3];
    vec4 cascadeSplits;
} cascadeData;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec4 inLightClipPositions[3];
layout(location = 5) flat in float inViewDepth;

layout(location = 0) out vec4 outColor;

int selectCascade() {
    if (inViewDepth < cascadeData.cascadeSplits.x) {
        return 0;
    }
    if (inViewDepth < cascadeData.cascadeSplits.y) {
        return 1;
    }
    return 2;
}

float shadowVisibility(int cascadeIndex) {
    const vec3 lightNdc = inLightClipPositions[cascadeIndex].xyz / inLightClipPositions[cascadeIndex].w;
    if (abs(lightNdc.x) > 1.0 || abs(lightNdc.y) > 1.0 || lightNdc.z <= 0.0 || lightNdc.z >= 1.0) {
        return 1.0;
    }

    const vec2 shadowUv = lightNdc.xy * 0.5 + 0.5;
    return texture(shadowMap, vec4(shadowUv, float(cascadeIndex), lightNdc.z - 0.0015));
}

void main() {
    const int cascadeIndex = selectCascade();
    const vec3 normal = normalize(inNormal);
    const vec3 lightDirection = normalize(vec3(0.50, 1.0, -0.70));
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const float illumination = 0.18 + 0.82 * diffuse * shadowVisibility(cascadeIndex);
    const vec3 cascadeTint[3] = vec3[](
        vec3(0.92, 1.00, 0.92),
        vec3(0.92, 0.96, 1.00),
        vec3(1.00, 0.94, 0.90)
    );
    outColor = vec4(inColor * illumination * cascadeTint[cascadeIndex], 1.0);
}
