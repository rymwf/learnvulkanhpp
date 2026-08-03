#version 450

layout(push_constant) uniform SurfacePushConstants {
    mat4 viewProjection;
    vec4 colorAndWireframe;
} pushConstants;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorldPosition;
layout(location = 2) noperspective in vec3 inBarycentric;

layout(location = 0) out vec4 outColor;

void main() {
    const vec3 normal = normalize(inNormal);
    const vec3 lightDirection = normalize(vec3(0.45, 0.82, 0.36));
    const vec3 fillDirection = normalize(vec3(-0.72, 0.20, -0.55));
    const float keyLight = max(dot(normal, lightDirection), 0.0);
    const float fillLight = max(dot(normal, fillDirection), 0.0);
    const float heightTint = smoothstep(-1.2, 1.2, inWorldPosition.y);
    const vec3 baseColor = mix(pushConstants.colorAndWireframe.rgb,
                               vec3(0.22, 0.78, 0.64), heightTint * 0.28);
    vec3 color = baseColor * (0.16 + 0.76 * keyLight + 0.20 * fillLight);

    const vec3 derivatives = fwidth(inBarycentric);
    const vec3 smoothed = smoothstep(vec3(0.0), derivatives * 1.25, inBarycentric);
    const float edge = 1.0 - min(smoothed.x, min(smoothed.y, smoothed.z));
    const float wireframe = pushConstants.colorAndWireframe.w;
    color = mix(color, vec3(1.0, 0.62, 0.20), edge * wireframe * 0.88);
    outColor = vec4(color, 1.0);
}
