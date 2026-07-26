#version 450

layout(set = 0, binding = 0) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 1) uniform sampler2D shadowDepth;

layout(set = 0, binding = 2) uniform FilteringSettings {
    vec4 controls;
} filteringSettings;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec4 inLightClipPosition;

layout(location = 0) out vec4 outColor;

bool projectShadow(out vec2 shadowUv, out float receiverDepth) {
    const vec3 lightNdc = inLightClipPosition.xyz / inLightClipPosition.w;
    if (abs(lightNdc.x) > 1.0 || abs(lightNdc.y) > 1.0 || lightNdc.z <= 0.0 || lightNdc.z >= 1.0) {
        return false;
    }

    shadowUv = lightNdc.xy * 0.5 + 0.5;
    receiverDepth = lightNdc.z - 0.0015;
    return true;
}

float hardShadow(vec2 shadowUv, float receiverDepth) {
    return texture(shadowMap, vec3(shadowUv, receiverDepth));
}

float pcfShadow(vec2 shadowUv, float receiverDepth) {
    const float texelSize = 1.0 / 1024.0;
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            visibility += texture(shadowMap, vec3(shadowUv + vec2(x, y) * texelSize, receiverDepth));
        }
    }
    return visibility / 9.0;
}

float pcssShadow(vec2 shadowUv, float receiverDepth) {
    const float texelSize = 1.0 / 1024.0;
    const float searchRadius = 18.0 * texelSize;
    float blockerDepthSum = 0.0;
    int blockerCount = 0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            const float sampleDepth = texture(shadowDepth, shadowUv + vec2(x, y) * searchRadius * 0.5).r;
            if (sampleDepth < receiverDepth) {
                blockerDepthSum += sampleDepth;
                blockerCount += 1;
            }
        }
    }
    if (blockerCount == 0) {
        return 1.0;
    }

    const float averageBlockerDepth = blockerDepthSum / float(blockerCount);
    const float penumbraRadius = clamp(
        (receiverDepth - averageBlockerDepth) / max(averageBlockerDepth, 0.001) * 42.0 * texelSize,
        2.0 * texelSize,
        28.0 * texelSize);
    float visibility = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            visibility += texture(shadowMap, vec3(shadowUv + vec2(x, y) * penumbraRadius * 0.5, receiverDepth));
        }
    }
    return visibility / 25.0;
}

float shadowVisibility() {
    vec2 shadowUv;
    float receiverDepth;
    if (!projectShadow(shadowUv, receiverDepth)) {
        return 1.0;
    }

    const int filteringMode = int(filteringSettings.controls.x + 0.5);
    if (filteringMode == 1) {
        return pcfShadow(shadowUv, receiverDepth);
    }
    if (filteringMode == 2) {
        return pcssShadow(shadowUv, receiverDepth);
    }
    return hardShadow(shadowUv, receiverDepth);
}

void main() {
    const vec3 normal = normalize(inNormal);
    const vec3 lightDirection = normalize(vec3(0.50, 1.0, -0.70));
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const float illumination = 0.18 + 0.82 * diffuse * shadowVisibility();
    outColor = vec4(inColor * illumination, 1.0);
}
