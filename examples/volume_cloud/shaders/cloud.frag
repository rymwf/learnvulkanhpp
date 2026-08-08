#version 450

const int MAX_VIEW_STEPS = 256;
const int MAX_LIGHT_STEPS = 16;
const float PI = 3.14159265358979323846;
const vec3 VOLUME_MIN = vec3(-1.5);
const vec3 VOLUME_MAX = vec3(1.5);

layout(set = 0, binding = 1) uniform sampler3D volumeTexture;

layout(push_constant) uniform CloudPushConstants {
    mat4 inverseViewProjection;
    vec4 lightDirectionAndPhase;
    vec4 mediumParameters;
    vec4 marchParameters;
    ivec4 options;
} pushConstants;

layout(location = 0) in vec2 inNdc;
layout(location = 0) out vec4 outColor;

vec3 safeReciprocal(vec3 value) {
    const float epsilon = 1.0e-6;
    return vec3(
        1.0 / (abs(value.x) < epsilon ? (value.x < 0.0 ? -epsilon : epsilon) : value.x),
        1.0 / (abs(value.y) < epsilon ? (value.y < 0.0 ? -epsilon : epsilon) : value.y),
        1.0 / (abs(value.z) < epsilon ? (value.z < 0.0 ? -epsilon : epsilon) : value.z));
}

bool intersectVolume(vec3 rayOrigin, vec3 rayDirection, out float entryDistance, out float exitDistance) {
    const vec3 inverseDirection = safeReciprocal(rayDirection);
    const vec3 distances0 = (VOLUME_MIN - rayOrigin) * inverseDirection;
    const vec3 distances1 = (VOLUME_MAX - rayOrigin) * inverseDirection;
    const vec3 nearDistances = min(distances0, distances1);
    const vec3 farDistances = max(distances0, distances1);
    entryDistance = max(nearDistances.x, max(nearDistances.y, nearDistances.z));
    exitDistance = min(farDistances.x, min(farDistances.y, farDistances.z));
    entryDistance = max(entryDistance, 0.0);
    return exitDistance > entryDistance;
}

vec3 worldToTexture(vec3 worldPosition) {
    return (worldPosition - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN);
}

float sampleDensity(vec3 worldPosition) {
    const vec3 texturePosition = worldToTexture(worldPosition);
    if (any(lessThan(texturePosition, vec3(0.0))) || any(greaterThan(texturePosition, vec3(1.0)))) {
        return 0.0;
    }
    return texture(volumeTexture, texturePosition).r;
}

float pixelJitter() {
    const vec2 seed = floor(gl_FragCoord.xy);
    return fract(52.9829189 * fract(dot(seed, vec2(0.06711056, 0.00583715))));
}

float henyeyGreenstein(float cosineTheta, float anisotropy) {
    const float g = clamp(anisotropy, -0.95, 0.95);
    const float gSquared = g * g;
    const float denominator = max(1.0 + gSquared - 2.0 * g * cosineTheta, 1.0e-4);
    return (1.0 - gSquared) / (4.0 * PI * pow(denominator, 1.5));
}

vec3 skyRadiance(vec3 rayDirection, vec3 lightDirection, float sunIntensity) {
    const float skyHeight = clamp(rayDirection.y * 0.5 + 0.5, 0.0, 1.0);
    const vec3 horizon = vec3(0.68, 0.80, 0.94);
    const vec3 zenith = vec3(0.13, 0.33, 0.68);
    vec3 color = mix(horizon, zenith, pow(skyHeight, 0.55));
    const float sunDisk = smoothstep(0.99955, 0.99992, dot(rayDirection, lightDirection));
    color += vec3(1.0, 0.82, 0.58) * sunDisk * (0.35 + 0.08 * sunIntensity);
    return color;
}

float lightTransmittance(vec3 worldPosition, vec3 lightDirection, float extinction, int lightSteps) {
    float entryDistance;
    float exitDistance;
    if (!intersectVolume(worldPosition + lightDirection * 1.0e-4, lightDirection,
                         entryDistance, exitDistance)) {
        return 1.0;
    }

    const float marchLength = max(exitDistance, 0.0);
    if (marchLength <= 1.0e-5) {
        return 1.0;
    }

    const float stepLength = marchLength / float(lightSteps);
    float opticalDepth = 0.0;
    for (int step = 0; step < MAX_LIGHT_STEPS && step < lightSteps; ++step) {
        const float distanceAlongRay = (float(step) + 0.5) * stepLength;
        opticalDepth += sampleDensity(worldPosition + lightDirection * distanceAlongRay) *
                        extinction * stepLength;
        if (opticalDepth >= 8.0) {
            break;
        }
    }
    return exp(-opticalDepth);
}

vec3 toneMap(vec3 color) {
    return vec3(1.0) - exp(-max(color, vec3(0.0)));
}

void renderDensityPreview(vec3 rayOrigin,
                          vec3 rayDirection,
                          float entryDistance,
                          float exitDistance,
                          float stepLength,
                          vec3 skyColor) {
    float maximumDensity = 0.0;
    float distanceAlongRay = entryDistance + pixelJitter() * stepLength;
    for (int step = 0; step < MAX_VIEW_STEPS && distanceAlongRay <= exitDistance; ++step) {
        maximumDensity = max(maximumDensity,
                             sampleDensity(rayOrigin + rayDirection * distanceAlongRay));
        distanceAlongRay += stepLength;
    }

    const float value = pow(clamp(maximumDensity, 0.0, 1.0), 0.65);
    const vec3 densityColor = mix(vec3(0.02, 0.08, 0.16), vec3(0.92, 0.97, 1.0), value);
    outColor = vec4(mix(toneMap(skyColor), densityColor, value), 1.0);
}

void renderCloud(vec3 rayOrigin,
                 vec3 rayDirection,
                 float entryDistance,
                 float exitDistance,
                 float stepLength,
                 vec3 skyColor) {
    const vec3 lightDirection = normalize(pushConstants.lightDirectionAndPhase.xyz);
    const float phase = henyeyGreenstein(dot(-rayDirection, lightDirection),
                                         pushConstants.lightDirectionAndPhase.w);
    const float extinction = max(pushConstants.mediumParameters.x, 0.001);
    const float scatteringAlbedo = clamp(pushConstants.mediumParameters.y, 0.0, 1.0);
    const float sunIntensity = max(pushConstants.mediumParameters.z, 0.0);
    const float ambientStrength = max(pushConstants.mediumParameters.w, 0.0);
    const int lightSteps = clamp(int(pushConstants.marchParameters.y + 0.5), 1, MAX_LIGHT_STEPS);

    const vec3 sunColor = vec3(1.0, 0.92, 0.78);
    const vec3 ambientColor = vec3(0.34, 0.50, 0.72);
    vec3 accumulatedRadiance = vec3(0.0);
    float transmittance = 1.0;
    float distanceAlongRay = entryDistance + pixelJitter() * stepLength;

    for (int step = 0; step < MAX_VIEW_STEPS && distanceAlongRay <= exitDistance; ++step) {
        const vec3 position = rayOrigin + rayDirection * distanceAlongRay;
        const float density = sampleDensity(position);
        if (density > 0.002) {
            const float segmentTransmittance = exp(-extinction * density * stepLength);
            const float segmentOpacity = 1.0 - segmentTransmittance;
            const float sunTransmittance = lightTransmittance(position, lightDirection,
                                                               extinction * 1.5, lightSteps);
            const float powder = 1.0 - exp(-density * 2.0);
            const float densityTowardSun =
                sampleDensity(position + lightDirection * stepLength * 2.0);
            const float sunFacingEdge = clamp((density - densityTowardSun) * 5.0, 0.0, 1.0);
            const vec3 directLight = sunColor * sunIntensity * sunTransmittance * phase *
                                     (0.75 + 0.25 * powder + 1.25 * sunFacingEdge);
            const vec3 ambientLight = ambientColor * ambientStrength * (0.55 + 0.45 * powder);
            accumulatedRadiance += transmittance * scatteringAlbedo *
                                   (directLight + ambientLight) * segmentOpacity;
            transmittance *= segmentTransmittance;
            if (transmittance < 0.01) {
                break;
            }
        }
        distanceAlongRay += stepLength;
    }

    outColor = vec4(toneMap(accumulatedRadiance + transmittance * skyColor), 1.0);
}

void main() {
    const vec4 nearHomogeneous = pushConstants.inverseViewProjection * vec4(inNdc, 0.0, 1.0);
    const vec4 farHomogeneous = pushConstants.inverseViewProjection * vec4(inNdc, 1.0, 1.0);
    const vec3 nearPosition = nearHomogeneous.xyz / nearHomogeneous.w;
    const vec3 farPosition = farHomogeneous.xyz / farHomogeneous.w;
    const vec3 rayDirection = normalize(farPosition - nearPosition);
    const vec3 lightDirection = normalize(pushConstants.lightDirectionAndPhase.xyz);
    const vec3 skyColor = skyRadiance(rayDirection, lightDirection,
                                      pushConstants.mediumParameters.z);

    float entryDistance;
    float exitDistance;
    if (!intersectVolume(nearPosition, rayDirection, entryDistance, exitDistance)) {
        outColor = vec4(toneMap(skyColor), 1.0);
        return;
    }

    const float voxelLength = (VOLUME_MAX.x - VOLUME_MIN.x) /
                              float(textureSize(volumeTexture, 0).x);
    const float stepLength = voxelLength * max(pushConstants.marchParameters.x, 0.25);
    if (pushConstants.options.x != 0) {
        renderDensityPreview(nearPosition, rayDirection, entryDistance, exitDistance,
                             stepLength, skyColor);
        return;
    }

    renderCloud(nearPosition, rayDirection, entryDistance, exitDistance, stepLength, skyColor);
}
