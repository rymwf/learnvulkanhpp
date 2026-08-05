#version 450

layout(push_constant) uniform SplatPushConstants {
    mat4 viewProjection;
    vec4 cameraRightAndRadius;
    vec4 cameraUpAndPadding;
    vec4 densityAndKernel;
    ivec4 gridParameters;
} pushConstants;

layout(location = 0) flat in float inDensity;
layout(location = 1) in vec2 inSplatCoordinate;

layout(location = 0) out vec4 outColor;

float normalizedDensity(float density) {
    const float windowWidth = max(pushConstants.densityAndKernel.y - pushConstants.densityAndKernel.x, 0.001);
    return clamp((density - pushConstants.densityAndKernel.x) / windowWidth, 0.0, 1.0);
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

void main() {
    const float radiusSquared = dot(inSplatCoordinate, inSplatCoordinate);
    if (radiusSquared > 1.0) {
        discard;
    }

    const float value = normalizedDensity(inDensity);
    const float gaussian = exp(-pushConstants.densityAndKernel.w * radiusSquared);
    const float opticalDepth = value * pushConstants.densityAndKernel.z * gaussian * 0.12;
    const float alpha = 1.0 - exp(-opticalDepth);
    if (alpha <= 0.001) {
        discard;
    }

    const vec3 color = transferColor(inDensity);
    outColor = vec4(color * alpha, alpha);
}
