#version 450

layout(quads, equal_spacing, ccw) in;

layout(push_constant) uniform PushConstants {
    float elapsedSeconds;
    float aspectRatio;
} pushConstants;

layout(location = 0) out vec3 outColor;

void main() {
    float u = gl_TessCoord.x;
    float v = gl_TessCoord.y;

    vec2 lowerEdge = mix(gl_in[0].gl_Position.xy, gl_in[1].gl_Position.xy, u);
    vec2 upperEdge = mix(gl_in[2].gl_Position.xy, gl_in[3].gl_Position.xy, u);
    vec2 patchPosition = mix(lowerEdge, upperEdge, v);

    float wave = 0.18 * sin(patchPosition.x * 4.0 + pushConstants.elapsedSeconds * 1.25) *
                 cos(patchPosition.y * 3.0 - pushConstants.elapsedSeconds * 0.9);
    vec3 surfacePosition = vec3(patchPosition.x, wave, patchPosition.y);

    const float cosTilt = 0.819152;
    const float sinTilt = 0.573576;
    vec3 viewPosition = vec3(
        surfacePosition.x,
        surfacePosition.y * cosTilt - surfacePosition.z * sinTilt,
        surfacePosition.y * sinTilt + surfacePosition.z * cosTilt + 3.2
    );

    const float focalLength = 2.0;
    const float nearPlane = 0.1;
    float aspectRatio = max(pushConstants.aspectRatio, 0.001);
    gl_Position = vec4(
        focalLength * viewPosition.x / aspectRatio,
        -focalLength * viewPosition.y,
        viewPosition.z - nearPlane,
        viewPosition.z
    );

    float heightFactor = clamp(wave / 0.36 + 0.5, 0.0, 1.0);
    vec3 lowColor = vec3(0.10, 0.45, 1.00);
    vec3 highColor = vec3(1.00, 0.28, 0.12);
    outColor = mix(lowColor, highColor, heightFactor);
}
