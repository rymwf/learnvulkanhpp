#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 0) out vec4 outColor;

void main() {
    const vec3 baseColor = vec3(0.82, 0.72, 0.62);
    const vec3 lightDirection = normalize(vec3(-0.5, 1.0, 0.7));
    const float diffuse = max(dot(normalize(inNormal), lightDirection), 0.0);
    const float illumination = 0.18 + 0.82 * diffuse;
    outColor = vec4(baseColor * illumination, 1.0);
}
