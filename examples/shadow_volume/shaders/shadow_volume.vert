#version 450

layout(push_constant) uniform PushConstants {
    mat4 cameraViewProjection;
    mat4 lightViewProjection;
} pushConstants;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outWorldPosition;

void main() {
    gl_Position = pushConstants.cameraViewProjection * vec4(inPosition, 1.0);
    outWorldPosition = inPosition;
}
