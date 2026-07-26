#version 460
#extension GL_EXT_ray_query : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec4 outColor;

bool isOccluded(vec3 origin, vec3 direction) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query,
                          topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                          0xff,
                          origin,
                          0.001,
                          direction,
                          100.0);
    while (rayQueryProceedEXT(query)) {
    }
    return rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

void main() {
    const vec3 normal = normalize(inNormal);
    const vec3 lightDirection = normalize(vec3(-0.55, 1.0, 0.35));
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const bool occluded = isOccluded(inWorldPosition + normal * 0.003, lightDirection);
    const float visibility = occluded ? 0.0 : 1.0;
    const float illumination = 0.16 + 0.84 * diffuse * visibility;
    outColor = vec4(inColor * illumination, 1.0);
}
