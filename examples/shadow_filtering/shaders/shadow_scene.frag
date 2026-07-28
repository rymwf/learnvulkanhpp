#version 450

layout(set = 0, binding = 0) uniform sampler2DShadow shadowMap;
layout(set = 0, binding = 1) uniform sampler2D shadowDepth;

layout(set = 0, binding = 2) uniform FilteringSettings {
    vec4 controls;
    vec4 pcssSearch;
    vec4 pcssPenumbra;
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
    receiverDepth = lightNdc.z - filteringSettings.controls.y;
    return true;
}

float hardShadow(vec2 shadowUv, float receiverDepth) {
    return texture(shadowMap, vec3(shadowUv, receiverDepth));
}

float pcfShadow(vec2 shadowUv, float receiverDepth) {
    const float texelSize = 1.0 / 1024.0;
    const int kernelRadius = clamp(int(filteringSettings.controls.z + 0.5), 0, 4);
    float visibility = 0.0;
    float sampleCount = 0.0;
    for (int y = -4; y <= 4; ++y) {
        for (int x = -4; x <= 4; ++x) {
            if (abs(x) > kernelRadius || abs(y) > kernelRadius) {
                continue;
            }
            visibility += texture(shadowMap, vec3(shadowUv + vec2(x, y) * texelSize, receiverDepth));
            sampleCount += 1.0;
        }
    }
    return visibility / sampleCount;
}

float pcssShadow(vec2 shadowUv, float receiverDepth) {
    const float texelSize = 1.0 / 1024.0;
    const int blockerKernelRadius = clamp(int(filteringSettings.pcssSearch.x + 0.5), 1, 4);
    const float blockerSampleStep =
        filteringSettings.pcssSearch.y * texelSize / float(blockerKernelRadius);
    float blockerDepthSum = 0.0;
    int blockerCount = 0;
    for (int y = -4; y <= 4; ++y) {
        for (int x = -4; x <= 4; ++x) {
            if (abs(x) > blockerKernelRadius || abs(y) > blockerKernelRadius) {
                continue;
            }
            const float sampleDepth = texture(shadowDepth, shadowUv + vec2(x, y) * blockerSampleStep).r;
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
        (receiverDepth - averageBlockerDepth) / max(averageBlockerDepth, 0.001) *
            filteringSettings.pcssSearch.z * texelSize,
        filteringSettings.pcssPenumbra.x * texelSize,
        filteringSettings.pcssPenumbra.y * texelSize);
    const int filterKernelRadius = clamp(int(filteringSettings.pcssSearch.w + 0.5), 1, 4);
    const float filterSampleStep = penumbraRadius / float(filterKernelRadius);
    float visibility = 0.0;
    float sampleCount = 0.0;
    for (int y = -4; y <= 4; ++y) {
        for (int x = -4; x <= 4; ++x) {
            if (abs(x) > filterKernelRadius || abs(y) > filterKernelRadius) {
                continue;
            }
            visibility +=
                texture(shadowMap, vec3(shadowUv + vec2(x, y) * filterSampleStep, receiverDepth));
            sampleCount += 1.0;
        }
    }
    return visibility / sampleCount;
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
