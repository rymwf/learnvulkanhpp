#version 450

const int MODE_DVR = 0;
const int MODE_MIP = 1;
const int MODE_ISOSURFACE = 2;
const int MAX_STEPS = 512;
const int ISO_REFINEMENT_STEPS = 5;
const vec3 VOLUME_MIN = vec3(-1.5);
const vec3 VOLUME_MAX = vec3(1.5);

layout(set = 0, binding = 1) uniform sampler3D volumeTexture;

layout(push_constant) uniform RaycastPushConstants {
    mat4 inverseViewProjection;
    vec4 rayParameters;
    vec4 densityWindow;
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
    return texture(volumeTexture, worldToTexture(worldPosition)).r;
}

float normalizedDensity(float density) {
    const float windowWidth = max(pushConstants.densityWindow.y - pushConstants.densityWindow.x, 0.001);
    return clamp((density - pushConstants.densityWindow.x) / windowWidth, 0.0, 1.0);
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

vec3 densityNormal(vec3 worldPosition) {
    const vec3 textureStep = 1.0 / vec3(textureSize(volumeTexture, 0));
    const vec3 texturePosition = worldToTexture(worldPosition);
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

vec3 shadeSample(vec3 color, vec3 worldPosition, vec3 rayDirection) {
    if (pushConstants.rayParameters.w < 0.5) {
        return color;
    }

    const vec3 normal = densityNormal(worldPosition);
    const vec3 lightDirection = normalize(vec3(0.45, 0.82, 0.36));
    const vec3 viewDirection = -rayDirection;
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const float rim = pow(1.0 - max(dot(normal, viewDirection), 0.0), 2.0);
    return color * (0.20 + 0.78 * diffuse) + vec3(0.08, 0.16, 0.22) * rim;
}

float pixelJitter() {
    const vec2 seed = floor(gl_FragCoord.xy);
    return fract(52.9829189 * fract(dot(seed, vec2(0.06711056, 0.00583715))));
}

void renderDvr(vec3 rayOrigin, vec3 rayDirection, float entryDistance, float exitDistance, float stepLength) {
    vec4 accumulated = vec4(0.0);
    float distanceAlongRay = entryDistance + pixelJitter() * stepLength;
    const float voxelLength = (VOLUME_MAX.x - VOLUME_MIN.x) / float(textureSize(volumeTexture, 0).x);

    for (int step = 0; step < MAX_STEPS && distanceAlongRay <= exitDistance; ++step) {
        const vec3 position = rayOrigin + rayDirection * distanceAlongRay;
        const float density = sampleDensity(position);
        const float value = normalizedDensity(density);
        if (value > 0.0) {
            const float referenceOpacity =
                clamp(pow(value, 1.35) * 0.085 * pushConstants.rayParameters.y, 0.0, 0.95);
            const float opacity =
                1.0 - pow(1.0 - referenceOpacity, max(stepLength / voxelLength, 0.001));
            vec3 color = transferColor(density);
            if (opacity > 0.004) {
                color = shadeSample(color, position, rayDirection);
            }
            const float contribution = (1.0 - accumulated.a) * opacity;
            accumulated.rgb += color * contribution;
            accumulated.a += contribution;
            if (accumulated.a >= 0.985) {
                break;
            }
        }
        distanceAlongRay += stepLength;
    }

    if (accumulated.a <= 0.002) {
        discard;
    }
    outColor = accumulated;
}

void renderMip(vec3 rayOrigin, vec3 rayDirection, float entryDistance, float exitDistance, float stepLength) {
    float maximumDensity = 0.0;
    float distanceAlongRay = entryDistance + pixelJitter() * stepLength;
    for (int step = 0; step < MAX_STEPS && distanceAlongRay <= exitDistance; ++step) {
        maximumDensity = max(maximumDensity, sampleDensity(rayOrigin + rayDirection * distanceAlongRay));
        distanceAlongRay += stepLength;
    }

    if (normalizedDensity(maximumDensity) <= 0.002) {
        discard;
    }
    outColor = vec4(transferColor(maximumDensity), 1.0);
}

void renderIsosurface(
    vec3 rayOrigin, vec3 rayDirection, float entryDistance, float exitDistance, float stepLength) {
    const float isoLevel = pushConstants.rayParameters.z;
    float previousDistance = entryDistance;
    float previousDensity = sampleDensity(rayOrigin + rayDirection * previousDistance);
    float distanceAlongRay = previousDistance + stepLength;

    for (int step = 1; step < MAX_STEPS && distanceAlongRay <= exitDistance; ++step) {
        const float density = sampleDensity(rayOrigin + rayDirection * distanceAlongRay);
        const bool crossedSurface = (previousDensity - isoLevel) * (density - isoLevel) <= 0.0 &&
                                    abs(density - previousDensity) > 1.0e-6;
        if (crossedSurface) {
            float lowerDistance = previousDistance;
            float upperDistance = distanceAlongRay;
            const bool lowerInside = previousDensity >= isoLevel;
            for (int refinement = 0; refinement < ISO_REFINEMENT_STEPS; ++refinement) {
                const float middleDistance = 0.5 * (lowerDistance + upperDistance);
                const float middleDensity = sampleDensity(rayOrigin + rayDirection * middleDistance);
                if ((middleDensity >= isoLevel) == lowerInside) {
                    lowerDistance = middleDistance;
                } else {
                    upperDistance = middleDistance;
                }
            }

            const vec3 hitPosition = rayOrigin + rayDirection * (0.5 * (lowerDistance + upperDistance));
            const vec3 color = shadeSample(transferColor(isoLevel), hitPosition, rayDirection);
            outColor = vec4(color, 1.0);
            return;
        }

        previousDistance = distanceAlongRay;
        previousDensity = density;
        distanceAlongRay += stepLength;
    }
    discard;
}

void main() {
    const vec4 nearHomogeneous = pushConstants.inverseViewProjection * vec4(inNdc, 0.0, 1.0);
    const vec4 farHomogeneous = pushConstants.inverseViewProjection * vec4(inNdc, 1.0, 1.0);
    const vec3 nearPosition = nearHomogeneous.xyz / nearHomogeneous.w;
    const vec3 farPosition = farHomogeneous.xyz / farHomogeneous.w;
    const vec3 rayDirection = normalize(farPosition - nearPosition);

    float entryDistance;
    float exitDistance;
    if (!intersectVolume(nearPosition, rayDirection, entryDistance, exitDistance)) {
        discard;
    }

    const float voxelLength = (VOLUME_MAX.x - VOLUME_MIN.x) / float(textureSize(volumeTexture, 0).x);
    const float stepLength = voxelLength * pushConstants.rayParameters.x;
    if (pushConstants.options.x == MODE_DVR) {
        renderDvr(nearPosition, rayDirection, entryDistance, exitDistance, stepLength);
    } else if (pushConstants.options.x == MODE_MIP) {
        renderMip(nearPosition, rayDirection, entryDistance, exitDistance, stepLength);
    } else {
        renderIsosurface(nearPosition, rayDirection, entryDistance, exitDistance, stepLength);
    }
}
