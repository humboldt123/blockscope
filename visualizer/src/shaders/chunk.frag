#version 330 core

layout (location = 0) out vec4 fragColor;

const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = 1 / gamma;

uniform sampler2DArray u_texture_array_0;
uniform vec3 bg_color;

in vec2 uv;
in float shading;
in vec3 tint_color;
in float frag_alpha;

flat in int face_id;
flat in int frag_tex_id;

void main() {
    vec4 tex_sample = texture(u_texture_array_0, vec3(uv, frag_tex_id));

    // Alpha test - discard nearly invisible pixels
    if (tex_sample.a < 0.1) discard;

    vec3 tex_col = tex_sample.rgb;
    tex_col = pow(tex_col, gamma);

    // Apply biome tint color
    tex_col *= tint_color;

    tex_col *= shading;

    // fog
    float fog_dist = gl_FragCoord.z / gl_FragCoord.w;
    tex_col = mix(tex_col, bg_color, (1.0 - exp2(-0.00001 * fog_dist * fog_dist)));

    tex_col = pow(tex_col, inv_gamma);
    fragColor = vec4(tex_col, frag_alpha);
}
