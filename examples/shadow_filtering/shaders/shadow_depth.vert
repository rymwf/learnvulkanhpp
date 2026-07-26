#version 450

layout(push_constant) uniform PushConstants {
    mat4 lightViewProjection;
} pushConstants;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = pushConstants.lightViewProjection * vec4(inPosition, 1.0);
}
