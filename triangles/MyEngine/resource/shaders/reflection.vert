/*************************************************************************
 *
 * reflection.vert
 *
 * Created by dmitry
 * 24.04.2021
 *
 ***/

#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shader_draw_parameters : require


struct ObjectInfo
{
    mat4 model_matrix;
    vec3 color;
};


// push constants
//=======================================================================================
layout( push_constant ) uniform constants
{
    mat4 viewProjMatrix;
    vec3 cameraPosition;
} pushConstants;
//=======================================================================================
//=======================================================================================


// set 0
//=======================================================================================
layout(set = 0, binding = 0) readonly buffer object_transform
{
    ObjectInfo object_info[];
};
//=======================================================================================
//=======================================================================================


// input vertex atributes
//=======================================================================================
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
//=======================================================================================
//=======================================================================================


// output atributes
//=======================================================================================
layout(location = 0) out vec3 outUVW;
layout(location = 1) out vec3 outNormal;
//=======================================================================================
//=======================================================================================

void main()
{
    //===================================================================================
    // "We are using gl_BaseInstance to access the object buffer. This is due to how Vulkan works
    // on its normal draw calls. All the draw commands in Vulkan request “first Instance” and
    // “instance count”. We are not doing instanced rendering, so instance count is always 1. But
    // we can still change the “first instance” parameter, and this way get gl_BaseInstance as a
    // integer we can use for whatever use we want to in the shader. This gives us a simple way
    // to send a single integer to the shader without setting up pushconstants or descriptors."
    // Source: https://vkguide.dev/docs/chapter-4/storage_buffers/
    //===================================================================================
    //===================================================================================
    mat4 model_matrix = object_info[  gl_BaseInstance  ].model_matrix;
    vec3 model_color = object_info[  gl_BaseInstance  ].color;

    vec4 worldPosition = model_matrix * vec4(inPosition, 1.0);

    outNormal = normalize((model_matrix * vec4(inNormal, 0.f)).xyz);

    gl_Position = pushConstants.viewProjMatrix * worldPosition;

    outUVW = worldPosition.xyz;
}