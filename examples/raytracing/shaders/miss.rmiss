#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    float t = 0.5 * (normalize(gl_WorldRayDirectionEXT).y + 1.0);
    payload = mix(vec3(0.02, 0.03, 0.08), vec3(0.25, 0.45, 0.75), t);
}
