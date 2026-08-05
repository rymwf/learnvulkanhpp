#version 450

const vec3 VOLUME_MIN = vec3(-1.5);
const vec3 VOLUME_MAX = vec3(1.5);
const float PLANE_HALF_EXTENT = 2.598076211;

layout(set = 0, binding = 1) uniform sampler3D volumeTexture;

layout(push_constant) uniform SlicePushConstants {
    mat4 viewProjection;
    vec4 cameraRightAndExtent;
    vec4 cameraUpAndSliceCount;
    vec4 cameraForwardAndOpacity;
    vec4 densityAndShading;
} pushConstants;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 0) out vec4 outColor;

vec3 worldToTexture(vec3 worldPosition) {
    return (worldPosition - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN);
}

float normalizedDensity(float density) {
    const float windowWidth = max(pushConstants.densityAndShading.y - pushConstants.densityAndShading.x, 0.001);
    return clamp((density - pushConstants.densityAndShading.x) / windowWidth, 0.0, 1.0);
}

vec3 transferColor(float density) {
    const float value = normalizedDensity(density);
    const vec3 darkBlue = vec3(0.025, 0.075, 0.14);
    const vec3 cyan = vec3(0.04, 0.74, 0.82);
    const vec3 amber = vec3(0.98, 0.68, 0.16);
    const vec3 white = vec3(1.0, 0.96, 0.88);
    if (value < 0.38) {
        return mix(darkBlue, cyan, value / 0.38);
    }
    if (value < 0.76) {
        return mix(cyan, amber, (value - 0.38) / 0.38);
    }
    return mix(amber, white, (value - 0.76) / 0.24);
}

vec3 densityNormal(vec3 texturePosition) {
    const vec3 textureStep = 1.0 / vec3(textureSize(volumeTexture, 0));
    const float x0 = texture(volumeTexture, texturePosition - vec3(textureStep.x, 0.0, 0.0)).r;
    const float x1 = texture(volumeTexture, texturePosition + vec3(textureStep.x, 0.0, 0.0)).r;
    const float y0 = texture(volumeTexture, texturePosition - vec3(0.0, textureStep.y, 0.0)).r;
    const float y1 = texture(volumeTexture, texturePosition + vec3(0.0, textureStep.y, 0.0)).r;
    const float z0 = texture(volumeTexture, texturePosition - vec3(0.0, 0.0, textureStep.z)).r;
    const float z1 = texture(volumeTexture, texturePosition + vec3(0.0, 0.0, textureStep.z)).r;
    const vec3 outwardGradient = -vec3(x1 - x0, y1 - y0, z1 - z0);
    const float squaredLength = dot(outwardGradient, outwardGradient);
    return squaredLength > 1.0e-10 ? outwardGradient * inversesqrt(squaredLength) : vec3(0.0, 1.0, 0.0);
}

vec3 shadeSample(vec3 color, vec3 texturePosition) {
    if (pushConstants.densityAndShading.z < 0.5) {
        return color;
    }

    const vec3 normal = densityNormal(texturePosition);
    const vec3 lightDirection = normalize(vec3(0.45, 0.82, 0.36));
    const vec3 viewDirection = -pushConstants.cameraForwardAndOpacity.xyz;
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 2.0);
    return color * (0.20 + 0.78 * diffuse) + vec3(0.08, 0.16, 0.22) * rim;
}

void main() {
    if (any(lessThan(inWorldPosition, VOLUME_MIN)) || any(greaterThan(inWorldPosition, VOLUME_MAX))) {
        discard;
    }

    const vec3 texturePosition = worldToTexture(inWorldPosition);
    const float density = texture(volumeTexture, texturePosition).r;
    const float value = normalizedDensity(density);
    if (value <= 0.0) {
        discard;
    }

    const float referenceOpacity =
        clamp(pow(value, 1.35) * 0.085 * pushConstants.cameraForwardAndOpacity.w, 0.0, 0.95);
    const float sliceCount = max(pushConstants.cameraUpAndSliceCount.w, 1.0);
    const float sliceSpacing = 2.0 * PLANE_HALF_EXTENT / sliceCount;
    const float voxelLength = (VOLUME_MAX.x - VOLUME_MIN.x) / float(textureSize(volumeTexture, 0).x);
    const float alpha = 1.0 - pow(1.0 - referenceOpacity, max(sliceSpacing / voxelLength, 0.001));
    if (alpha <= 0.001) {
        discard;
    }

    const vec3 color = shadeSample(transferColor(density), texturePosition);
    outColor = vec4(color * alpha, alpha);
}
