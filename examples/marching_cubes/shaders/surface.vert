#version 450

struct Vertex {
    vec4 position;
    vec4 normal;
};

layout(std430, set = 0, binding = 1) readonly buffer GeneratedVertices {
    Vertex vertices[];
};

layout(push_constant) uniform SurfacePushConstants {
    mat4 viewProjection;
    vec4 colorAndWireframe;
} pushConstants;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outWorldPosition;
layout(location = 2) noperspective out vec3 outBarycentric;

void main() {
    const Vertex vertex = vertices[gl_VertexIndex];
    gl_Position = pushConstants.viewProjection * vertex.position;
    outNormal = vertex.normal.xyz;
    outWorldPosition = vertex.position.xyz;

    const int triangleVertex = gl_VertexIndex % 3;
    outBarycentric = triangleVertex == 0 ? vec3(1.0, 0.0, 0.0)
                   : triangleVertex == 1 ? vec3(0.0, 1.0, 0.0)
                                         : vec3(0.0, 0.0, 1.0);
}
