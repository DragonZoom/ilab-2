/*************************************************************************
 *
 * Driver.h
 *
 * Created by dmitry
 * 22.02.2021
 *
 ***/


//
/// Engine
///======================================================================================
/// A humble engine based on the Vulkan API.
///
/// In this version, the following features:
/// - Drawing a grid of triangles of different sizes.
/// - Maximum objects == 10000 (configurable by default in the header of the current
///   file).
/// - Ability to load new objects after engine initialization.
/// - Rendering only those objects that are passed to the render function.
/// - All objects must be inherited from the Mesh class. In this class, you can override
///   methods that return the model matrix and color.
///======================================================================================
///======================================================================================
//


#pragma once


#include "vkCore.h"
#include "CameraView.h"
#include "DeleteonQueue.hpp"

#include "Renderable/Renderable.hpp"
#include "Types.hpp"



#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>


#include <fstream>
#include <functional>
#include <deque>
#include <array>
#include <algorithm>
#include <iterator>


#include <OtherLibs/timer.h>


const size_t ezg_default_max_objects = 10000;


namespace ezg::engine
{

    class Engine
    {
        struct FrameData
        {
            VkSemaphore presentSemaphore = nullptr;
            VkSemaphore renderSemaphore = nullptr;
            VkFence renderFence = nullptr;

            VkCommandPool commandPool = nullptr;
            VkCommandBuffer mainCommandBuffer = nullptr;

            VkDescriptorSet globalDescriptor = nullptr;
            AllocatedBuffer objectBuffer = {};
            AllocatedBuffer sceneBuffer = {};
        };//struct FrameData

    private:

        struct GPUObjectData
        {
            alignas(16) glm::mat4 model;
            alignas(16) glm::vec3 color;
        };


    private:

        const Window& m_rWindow;
        Core m_core;

        VmaAllocator m_allocator = nullptr;

        std::vector< FrameData > m_frames;

        std::vector< VkImage > m_images;
        std::vector< VkImageView > m_views;
        std::vector< VkFramebuffer > m_frameBuffers;

        VkSwapchainKHR m_swapChainKHR = nullptr;
        VkQueue m_queue = nullptr;

        VkRenderPass m_renderPass = nullptr;

        std::unordered_map< RenderMaterial::Type, RenderMaterial >
                m_renderMaterials;


        VkImageView m_depthImageView = nullptr;
        AllocatedImage m_depthImage = {};


        VkDescriptorPool m_descriptorPool = nullptr;
        VkDescriptorSetLayout m_globalSetLayout = nullptr;
        //VkDescriptorSetLayout          m_objectSetLayout = nullptr; //todo

        Timer m_time;

        size_t m_currentFrame = 0;
        size_t m_maxFramesInFlight = 1;

        // TODO increase if necessary
        size_t m_numObjects = ezg_default_max_objects;

        CameraView m_cameraView;


        DeletionQueue_t m_deletionQueue;


        VkDescriptorSetLayout m_reflectionDescriptorSetLayout = nullptr;

        const std::string m_reflectionVertShaderPathName = "resource/shaders/reflection.vert.spv";
        const std::string m_reflectionFragShaderPathName = "resource/shaders/reflection.frag.spv";

        void createReflectionPipeLine_t();

        ///========================================================================
        /// TODO it is experimental
        size_t         m_maxNumLight = 1;
        AllocatedImage m_shadowMap         = {};
        VkImageView    m_shadowMapView     = nullptr;
        VkSampler      m_shadowSampler       = nullptr;

        VkDescriptorSetLayout m_shadowSetLayout = nullptr;
        VkDescriptorSet m_shadowDescriptorSet = nullptr;

        std::vector< frameBufferData > m_shadowFrameBuffers = {};
        AllocatedBuffer m_lightInfoBuffer = {};

        VkExtent3D  m_shadowExtent = { 2024, 2024, 1 };

        const std::string m_depthFragShaderPathName = "resource/shaders/depth.frag.spv";



        VkRenderPass  m_shadowRenderPass   = nullptr;
        void createShadowRenderPass_t();
        void createShadowFrameBuffer_t();
        void createShadowDescriptors_t();


        GPUSceneInfo m_sceneInfo = { 0.3, 4 };
        ///========================================================================
        ///========================================================================


    public:

        Engine(const Engine&) = delete; //not supported
        Engine& operator=(const Engine&) = delete; //not supported
        Engine(Engine&&) = delete; //not supported
        Engine& operator=(Engine&&) = delete; //not supported


        Engine(const Window& window);


        ~Engine();


    public:

        void upload_object(Renderable& mesh);
        void unload_object(Renderable& mesh);

        void render_meshes(const std::vector< Renderable* >& objects);


        void cleanup();


        void setCameraView(const CameraView& camera) { m_cameraView = camera; }


    private:

        void init_(GLFWwindow* window);


        void upload_mesh(Mesh& mesh);
        void unload_mesh(Mesh& mesh);

        void upload_mirror(Mirror& mirror);
        void unload_mirror(Mirror& mirror);


        FrameData& getCurFrame() { return m_frames[m_currentFrame % m_maxFramesInFlight]; }

        AllocatedBuffer create_buffer_(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

        AllocatedImage create_image_(VkImageType type
                                     , VkFormat format
                                     , VkImageCreateFlags flags
                                     , VkSampleCountFlagBits samples
                                     , VkImageTiling tiling
                                     , VkImageUsageFlags usage
                                     , VmaMemoryUsage memoryUsage
                                     , VkExtent3D extent
                                     , uint32_t layers
                                     , uint32_t mipLevels);


        VkShaderModule createShaderModule_(const std::string& source_);

        void createSwapChain_();
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

        void initDescriptors();

        void updateUniformBuffer_(uint32_t currentImage_, const std::vector< Renderable* >& meshes);

        void createCommandBuffer_();

        void createFramebuffer_();

        void createDepthResources_();

        void createRenderPass_();

        void createPipeline_();

        void createSyncObjects_();


        bool hasStencilComponent(VkFormat format)
        {
            return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT;
        }
    };



    std::array< CameraView, 6 > getCameraViewCube(const glm::vec3& position);

}//namespace vks