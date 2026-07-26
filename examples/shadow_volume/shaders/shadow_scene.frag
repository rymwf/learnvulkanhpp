#version 450

layout(set = 0, binding = 0) uniform sampler2DShadow shadowMap;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec4 inLightClipPosition;

layout(location = 0) out vec4 outColor;

float shadowVisibility() {
    const vec3 lightNdc = inLightClipPosition.xyz / inLightClipPosition.w;
    if (abs(lightNdc.x) > 1.0 || abs(lightNdc.y) > 1.0 || lightNdc.z <= 0.0 || lightNdc.z >= 1.0) {
        return 1.0;
    }

    const vec2 shadowUv = lightNdc.xy * 0.5 + 0.5;
    const float receiverBias = 0.0015;
    return texture(shadowMap, vec3(shadowUv, lightNdc.z - receiverBias));
}

void main() {
    const vec3 normal = normalize(inNormal);
    const vec3 lightDirection = normalize(vec3(0.50, 1.0, -0.70));
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const float illumination = 0.18 + 0.82 * diffuse * shadowVisibility();
    outColor = vec4(inColor * illumination, 1.0);
}
