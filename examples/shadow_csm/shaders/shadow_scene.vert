#version 450

layout(set = 0, binding = 1, std140) uniform CascadeData {
    mat4 lightViewProjections[3];
    vec4 cascadeSplits;
} cascadeData;

layout(push_constant) uniform PushConstants {
    mat4 cameraViewProjection;
    mat4 cameraView;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outColor;
layout(location = 2) out vec4 outLightClipPositions[3];
layout(location = 5) flat out float outViewDepth;

void main() {
    const vec4 worldPosition = vec4(inPosition, 1.0);
    gl_Position = pushConstants.cameraViewProjection * worldPosition;
    outNormal = inNormal;
    outColor = inColor;
    for (int cascadeIndex = 0; cascadeIndex < 3; ++cascadeIndex) {
        outLightClipPositions[cascadeIndex] = cascadeData.lightViewProjections[cascadeIndex] * worldPosition;
    }
    outViewDepth = -(pushConstants.cameraView * worldPosition).z;
}
