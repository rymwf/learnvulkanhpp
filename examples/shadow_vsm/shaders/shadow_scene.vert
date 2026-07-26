#version 450

layout(push_constant) uniform PushConstants {
    mat4 cameraViewProjection;
    mat4 lightViewProjection;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outColor;
layout(location = 2) out vec4 outLightClipPosition;

void main() {
    gl_Position = pushConstants.cameraViewProjection * vec4(inPosition, 1.0);
    outNormal = inNormal;
    outColor = inColor;
    outLightClipPosition = pushConstants.lightViewProjection * vec4(inPosition, 1.0);
}
