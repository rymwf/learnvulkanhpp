#version 450

const vec3 VOLUME_MIN = vec3(-1.5);
const vec3 VOLUME_MAX = vec3(1.5);

layout(set = 0, binding = 1) uniform sampler3D volumeTexture;

layout(push_constant) uniform SplatPushConstants {
    mat4 viewProjection;
    vec4 cameraRightAndRadius;
    vec4 cameraUpAndPadding;
    vec4 densityAndKernel;
    ivec4 gridParameters;
} pushConstants;

layout(location = 0) in uint inVoxelIndex;

layout(location = 0) flat out float outDensity;
layout(location = 1) out vec2 outSplatCoordinate;

void main() {
    const uint resolution = uint(pushConstants.gridParameters.x);
    const uint sliceSize = resolution * resolution;
    const uint z = inVoxelIndex / sliceSize;
    const uint remainder = inVoxelIndex - z * sliceSize;
    const uint y = remainder / resolution;
    const uint x = remainder - y * resolution;
    const vec3 textureCoordinate = (vec3(x, y, z) + vec3(0.5)) / float(resolution);
    const float density = textureLod(volumeTexture, textureCoordinate, 0.0).r;

    const vec2 corners[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0));
    const vec2 corner = corners[gl_VertexIndex];
    outDensity = density;
    outSplatCoordinate = corner;

    if (density <= pushConstants.densityAndKernel.x) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    const vec3 center = mix(VOLUME_MIN, VOLUME_MAX, textureCoordinate);
    const vec3 offset =
        (pushConstants.cameraRightAndRadius.xyz * corner.x +
         pushConstants.cameraUpAndPadding.xyz * corner.y) * pushConstants.cameraRightAndRadius.w;
    gl_Position = pushConstants.viewProjection * vec4(center + offset, 1.0);
}
