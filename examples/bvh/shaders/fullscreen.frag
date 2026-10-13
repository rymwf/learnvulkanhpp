#version 450

layout(set = 0, binding = 1) uniform sampler2D outputImage;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texelFetch(outputImage, ivec2(gl_FragCoord.xy), 0);
}
