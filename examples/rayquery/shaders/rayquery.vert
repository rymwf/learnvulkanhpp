#version 460

layout(push_constant) uniform PushConstants {
    mat4 viewProjection;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outColor;

void main() {
    outWorldPosition = inPosition;
    outNormal = inNormal;
    outColor = inColor;
    gl_Position = pushConstants.viewProjection * vec4(inPosition, 1.0);
}
