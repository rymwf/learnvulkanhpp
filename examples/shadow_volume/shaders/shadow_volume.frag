#version 450

layout(set = 0, binding = 0) uniform sampler2DShadow shadowMap;

layout(push_constant) uniform PushConstants {
    mat4 cameraViewProjection;
    mat4 lightViewProjection;
} pushConstants;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 0) out vec4 outColor;

float shadowVisibility(vec3 worldPosition) {
    const vec4 lightClip = pushConstants.lightViewProjection * vec4(worldPosition, 1.0);
    const vec3 lightNdc = lightClip.xyz / lightClip.w;
    if (abs(lightNdc.x) > 1.0 || abs(lightNdc.y) > 1.0 || lightNdc.z <= 0.0 || lightNdc.z >= 1.0) {
        return 1.0;
    }
    const vec2 shadowUv = lightNdc.xy * 0.5 + 0.5;
    return texture(shadowMap, vec3(shadowUv, lightNdc.z - 0.0015));
}

void main() {
    const vec3 lightDirection = normalize(vec3(0.50, 1.0, -0.70));
    float integratedVisibility = 0.0;
    const int sampleCount = 12;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        const float t = (float(sampleIndex) + 0.5) / float(sampleCount);
        integratedVisibility += shadowVisibility(inWorldPosition + lightDirection * (t * 3.0));
    }
    integratedVisibility /= float(sampleCount);

    const float heightFade = smoothstep(0.05, 0.8, inWorldPosition.y) *
                             (1.0 - smoothstep(2.2, 3.25, inWorldPosition.y));
    const float density = 0.22 * heightFade;
    const vec3 scatteringTint = vec3(1.0, 0.82, 0.42);
    outColor = vec4(scatteringTint * (0.20 + 0.80 * integratedVisibility), density * integratedVisibility);
}
