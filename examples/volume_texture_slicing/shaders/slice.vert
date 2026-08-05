#version 450

const float PLANE_HALF_EXTENT = 2.598076211;

layout(push_constant) uniform SlicePushConstants {
    mat4 viewProjection;
    vec4 cameraRightAndExtent;
    vec4 cameraUpAndSliceCount;
    vec4 cameraForwardAndOpacity;
    vec4 densityAndShading;
} pushConstants;

layout(location = 0) out vec3 outWorldPosition;

void main() {
    const vec2 corners[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, 1.0));
    const vec2 corner = corners[gl_VertexIndex];
    const float sliceCount = max(pushConstants.cameraUpAndSliceCount.w, 1.0);
    const float sliceSpacing = 2.0 * PLANE_HALF_EXTENT / sliceCount;
    const float planeOffset = PLANE_HALF_EXTENT - (float(gl_InstanceIndex) + 0.5) * sliceSpacing;
    const vec3 center = pushConstants.cameraForwardAndOpacity.xyz * planeOffset;
    outWorldPosition = center +
                       pushConstants.cameraRightAndExtent.xyz * corner.x * PLANE_HALF_EXTENT +
                       pushConstants.cameraUpAndSliceCount.xyz * corner.y * PLANE_HALF_EXTENT;
    gl_Position = pushConstants.viewProjection * vec4(outWorldPosition, 1.0);
}
