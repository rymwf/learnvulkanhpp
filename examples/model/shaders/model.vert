#version 450

layout(push_constant) uniform PushConstants {
    mat4 viewProjection;
} pushConstants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outNormal;

void main() {
    gl_Position = pushConstants.viewProjection * vec4(inPosition, 1.0);
    outNormal = inNormal;
}
