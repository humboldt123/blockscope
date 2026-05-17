#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in float in_tex_id;
layout (location = 2) in float in_packed_face_ao;
layout (location = 3) in vec3 in_tint_color;
layout (location = 4) in float in_alpha;

uniform mat4 m_proj;
uniform mat4 m_view;

flat out int frag_tex_id;
flat out int face_id;

out vec2 uv;
out float shading;
out vec3 tint_color;
out float frag_alpha;

const float ao_values[4] = float[4](0.1, 0.25, 0.5, 1.0);

const float face_shading[12] = float[12](
    1.0, 0.5,  // top bottom
    0.5, 0.8,  // right left
    0.5, 0.8,  // front back
    1.0, 1.0,  // unshaded (face_id + 6)
    1.0, 1.0,
    1.0, 1.0
);

const vec2 uv_coords[4] = vec2[4](
    vec2(0, 0), vec2(0, 1),
    vec2(1, 0), vec2(1, 1)
);

const int uv_indices[24] = int[24](
    1, 0, 2, 1, 2, 3,  // even face
    3, 0, 2, 3, 1, 0,  // odd face
    3, 1, 0, 3, 0, 2,  // even flipped
    1, 2, 3, 1, 0, 2   // odd flipped
);

void main() {
    int pack_val = int(in_packed_face_ao);
    face_id = pack_val / 16;
    int ao_id = (pack_val % 16) / 2;
    int flip_id = pack_val % 2;

    frag_tex_id = int(in_tex_id);
    tint_color = in_tint_color;
    frag_alpha = in_alpha;

    // UV uses face parity (& 1) so face_id 6-11 works like 0-5
    int uv_index = gl_VertexID % 6 + ((face_id & 1) + flip_id * 2) * 6;
    uv = uv_coords[uv_indices[uv_index]];

    shading = face_shading[face_id] * ao_values[ao_id];

    gl_Position = m_proj * m_view * vec4(in_position, 1.0);
}
