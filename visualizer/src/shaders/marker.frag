#version 330 core

layout (location = 0) out vec4 fragColor;

uniform vec3 u_color;

void main() {
    fragColor = vec4(u_color, 1.0);
}
