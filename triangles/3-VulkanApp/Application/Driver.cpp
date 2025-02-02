/*************************************************************************
 *
 * Driver.cpp
 *
 * Created by dmitry
 * 22.02.2021
 *
 ***/

#include "Engine.h"

#include <cassert>
#include <cstring>
#include <chrono>


const std::string vert_shader_fname = "resource/shaders/vert.spv";
const std::string frag_shader_fname = "resource/shaders/frag.spv";

namespace vks
{

	void VulkanDriver::Init(GLFWwindow* window)
	{
		try
		{
            m_pWindow = window;

			m_core.Init(m_pWindow);

			vkGetDeviceQueue(m_core.getDevice(), m_core.getQueueFamily(), 0, &m_queue);

			createSwapChain_();

			createRenderPass_();
            createDescriptorSetLayout_();

			createCommandBuffer_();
			createDepthResources_();
			createFramebuffer_();
			createVertexBuffer_();

			createPipeline_();

			createUniformBuffers_();
			createDescriptorPool_();
			createDescriptorSets_();

			recordCommandBuffers_();

			createSyncObjects_();

            m_worldCoords.resize(vertices.size());
        }
		catch (std::exception& exc_)
		{
			std::cerr << "Fatal error in Engine::init_():\n";
			std::cerr << "What(): " << exc_.what() << std::endl;
			std::cerr << "Called Engine::cleanup()" << std::endl;
			cleanup();
		}
	}



    void VulkanDriver::addObject(const ObjectInfo& info)
    {
	    UniformModel cur {};
	    cur.model = info.model_matrix;
	    cur.color = info.color;
	    m_modelData.push_back(cur);


	    for(const auto& vert : info.vertices) {
	        vertices.push_back(vert);
	    }
    }
    void VulkanDriver::setObjectInfo(size_t objectID, const ObjectInfo& info)
    {
        auto& md = m_modelData.at(objectID);

        md.model = info.model_matrix;
        md.color = info.color;
    }


    std::vector< glm::vec3 > VulkanDriver::getWorldCoords(size_t objectID)
    {
	    //todo objects of different sizes
	    const size_t num_vertices = 3;
	    std::vector< glm::vec3 > res(num_vertices);

	    for(size_t i = 0; i < num_vertices; i++) {
	        res.at(i) = m_worldCoords[num_vertices * objectID + i];
	    }

	    return res;
    }



    void VulkanDriver::createSwapChain_()
	{
		const VkSurfaceCapabilitiesKHR& SurfaceCaps = m_core.getSurfaceCaps();
		assert(SurfaceCaps.currentExtent.width != -1);
		
		uint32_t numImages = SurfaceCaps.minImageCount + 1;
		assert(numImages <= SurfaceCaps.maxImageCount);

		VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
		swapChainCreateInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		swapChainCreateInfo.surface          = m_core.getSurface();
		swapChainCreateInfo.minImageCount    = numImages;
		swapChainCreateInfo.imageFormat      = m_core.getSurfaceFormat().format;
		swapChainCreateInfo.imageColorSpace  = m_core.getSurfaceFormat().colorSpace;
		swapChainCreateInfo.imageExtent      = SurfaceCaps.currentExtent;
		swapChainCreateInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		swapChainCreateInfo.preTransform     = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
		swapChainCreateInfo.imageArrayLayers = 1;
		swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		swapChainCreateInfo.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
		swapChainCreateInfo.clipped          = VK_TRUE;
		swapChainCreateInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		if (vkCreateSwapchainKHR(m_core.getDevice(), &swapChainCreateInfo, nullptr, &m_swapChainKHR) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create swap chain!"));
		}

		uint32_t numSwapChainImages = 0;
		if (VK_SUCCESS != vkGetSwapchainImagesKHR(m_core.getDevice(), m_swapChainKHR, &numSwapChainImages, nullptr)) {
            throw std::runtime_error(DEBUG_MSG("failed to get swap chain ImageKHR!"));
        }

		m_images.resize(numSwapChainImages);
		m_cmdBufs.resize(numSwapChainImages);
		m_views.resize(numSwapChainImages);

        if (VK_SUCCESS != vkGetSwapchainImagesKHR(m_core.getDevice(), m_swapChainKHR, &numSwapChainImages, m_images.data())) {
            throw std::runtime_error(DEBUG_MSG("failed to get swap chain ImageKHR!"));
        }
	}


	void VulkanDriver::recreateSwapChain_()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(m_pWindow, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(m_pWindow, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(m_core.getDevice());

		cleanupSwapChain();

		m_core.updataPhysDivicesProp();
		createSwapChain_();

		createRenderPass_();
		createPipeline_();
		createDepthResources_();

		createFramebuffer_();

		createUniformBuffers_();
		createDescriptorPool_();
		createDescriptorSets_();

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = m_cmdBufPool;
		allocInfo.commandBufferCount = m_cmdBufs.size();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(m_core.getDevice(), &allocInfo, m_cmdBufs.data()) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to allocate command buffers!"));
		}
		recordCommandBuffers_();
	}


	void VulkanDriver::cleanupSwapChain()
	{
		const auto device_ = m_core.getDevice();
		if (device_ == nullptr) { return; }

		vkDestroyImageView(device_, m_depthImageView, nullptr);
		vkDestroyImage(device_, m_depthImage, nullptr);
		vkFreeMemory(device_, m_depthImageMemory, nullptr);

		for (size_t i = 0, mi = m_fbs.size(); i < mi; i++) {
			vkDestroyFramebuffer(device_, m_fbs[i], nullptr);
		}
		m_fbs.clear();

		vkFreeCommandBuffers(device_, m_cmdBufPool, static_cast<uint32_t>(m_cmdBufs.size()), m_cmdBufs.data());

		vkDestroyPipeline(device_, m_pipeline, nullptr);
		vkDestroyPipelineLayout(device_, m_pipelineLayout, nullptr);
		vkDestroyRenderPass(device_, m_renderPass, nullptr);

		for (size_t i = 0, mi = m_views.size(); i < mi; i++) {
			vkDestroyImageView(device_, m_views[i], nullptr);
		}
		m_views.clear();

		vkDestroySwapchainKHR(device_, m_swapChainKHR, nullptr);

		for (size_t i = 0, mi = m_uniformBuffersFromUbo.size(); i < mi; i++)
		{
			vkDestroyBuffer(device_, m_uniformBuffersFromUbo[i], nullptr);
			vkFreeMemory(device_, m_uniformBuffersMemoryFromUbo[i], nullptr);
		}
		m_uniformBuffersFromUbo.clear();
		m_uniformBuffersMemoryFromUbo.clear();
		m_images.clear();

		vkDestroyDescriptorPool(device_, m_descriptorPool, nullptr);
	}


	void VulkanDriver::cleanup()
	{
		cleanupSwapChain();

		const auto device_ = m_core.getDevice();
		if (device_ == nullptr) { return; }

		vkDestroyDescriptorSetLayout(device_, m_descriptorSetLayout, nullptr);
		m_descriptorSetLayout = nullptr;

		vkDestroyBuffer(device_, m_vertexBuffer, nullptr);
		m_vertexBuffer = nullptr;
		vkFreeMemory(device_, m_vertexBufferMemory, nullptr);
		m_vertexBufferMemory = nullptr;

		for (size_t i = 0, mi = std::min(m_renderFinishedSem.size(), m_maxFramesInFlight); i < mi; i++)
		{
			vkDestroySemaphore(device_, m_renderFinishedSem[i], nullptr);
			vkDestroySemaphore(device_, m_imageAvailableSem[i], nullptr);
			vkDestroyFence(device_, m_inFlightFences[i], nullptr);
		}
		m_renderFinishedSem.clear();
		m_imageAvailableSem.clear();
		m_inFlightFences.clear();

		vkDestroyCommandPool(device_, m_cmdBufPool, nullptr);
		m_cmdBufPool = nullptr;

		//glfwDestroyWindow(m_pWindow);
		m_pWindow = nullptr;

		m_core.cleanup();
	}

	
	void VulkanDriver::createCommandBuffer_()
	{
		VkCommandPoolCreateInfo cmdPoolCreateInfo = {};
		cmdPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolCreateInfo.queueFamilyIndex = m_core.getQueueFamily();

		if (vkCreateCommandPool(m_core.getDevice(), &cmdPoolCreateInfo, nullptr, &m_cmdBufPool) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create graphics command pool!"));
		}

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = m_cmdBufPool;
		allocInfo.commandBufferCount = m_cmdBufs.size();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(m_core.getDevice(), &allocInfo, m_cmdBufs.data()) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to allocate command buffers!"));
		}
	}


	void VulkanDriver::recordCommandBuffers_()
	{
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		//beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkImageSubresourceRange imageRange = {};
		imageRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageRange.levelCount = 1;
		imageRange.layerCount = 1;

		int wHeight = 0, wWidth = 0;
		glfwGetWindowSize(m_pWindow, &wWidth, &wHeight);

		for (size_t k = 0, mk = m_cmdBufs.size(); k < mk; k++)
		{
			if (vkBeginCommandBuffer(m_cmdBufs[k], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error(DEBUG_MSG("failed to begin recording command buffer!"));
			}

			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = m_renderPass;
			renderPassInfo.framebuffer = m_fbs[k];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent.height = wHeight;
			renderPassInfo.renderArea.extent.width = wWidth;
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(m_cmdBufs[k], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(m_cmdBufs[k], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

			VkBuffer vertexBuffers[] = {
				m_vertexBuffer
			};
			VkDeviceSize offsets[] = {
				0,
			};
			vkCmdBindVertexBuffers(m_cmdBufs[k], 0, 1, vertexBuffers, offsets);
            //todo
            uint32_t dynamicOffsets[] = {
                    0,
                    0
            };
			vkCmdBindDescriptorSets(m_cmdBufs[k], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorSets[k], 2, dynamicOffsets);

			size_t numVec = vertices.size();
            for (size_t curT = 1; curT * 3 <= numVec; curT++) {
                // curT - 1 == gl_BaseInstance in vert shader
                vkCmdDraw(m_cmdBufs[k], 3u, 1, (curT - 1) * 3, curT - 1);
            }

			vkCmdEndRenderPass(m_cmdBufs[k]);

			if (vkEndCommandBuffer(m_cmdBufs[k]) != VK_SUCCESS) {
				throw std::runtime_error(DEBUG_MSG("failed to record command buffer!"));
			}
		}
	}


	void VulkanDriver::createDepthResources_()
	{
		VkFormat depthFormat = m_core.findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
		int wHeight = 0, wWidth = 0;
		glfwGetWindowSize(m_pWindow, &wWidth, &wHeight);

		m_core.createImage(wWidth, wHeight,
			depthFormat, 
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			m_depthImage,
			m_depthImageMemory
			);

		m_depthImageView = m_core.createImageView(m_depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
	}


	void VulkanDriver::createVertexBuffer_()
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer = {};
		VkDeviceMemory stagingBufferMemory = {};
		m_core.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer, stagingBufferMemory);
		
		void* data = nullptr;
		auto device_ = m_core.getDevice();
		vkMapMemory(device_, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device_, stagingBufferMemory);

		m_core.createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertexBuffer, m_vertexBufferMemory);
		
		copyBuffer_(stagingBuffer, m_vertexBuffer, bufferSize);

		vkDestroyBuffer(device_, stagingBuffer, nullptr);
		vkFreeMemory(device_, stagingBufferMemory, nullptr);

	}


	void VulkanDriver::copyBuffer_(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}


	VkCommandBuffer VulkanDriver::beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_cmdBufPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(m_core.getDevice(), &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void VulkanDriver::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(m_queue);

		vkFreeCommandBuffers(m_core.getDevice(), m_cmdBufPool, 1, &commandBuffer);
	}


	void VulkanDriver::createUniformBuffers_()
	{
	    ////////////////////////////////////////////
	    //create buffers from ubo
		VkDeviceSize bufferSize1 = sizeof(UniformView);
		m_uniformBuffersFromUbo.resize(m_images.size());
		m_uniformBuffersMemoryFromUbo.resize(m_images.size());

		for (size_t i = 0, mi = m_images.size(); i < mi; i++)
		{
			m_core.createBuffer(bufferSize1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                m_uniformBuffersFromUbo[i], m_uniformBuffersMemoryFromUbo[i]);
		}
        ////////////////////////////////////////////

        ////////////////////////////////////////////
        //create buffers from model info
        VkDeviceSize bufferSize2 = sizeof(UniformModel) * vertices.size() / 3;
        m_uniformBuffersFromModel.resize(m_images.size());
        m_uniformBuffersMemoryFromModel.resize(m_images.size());

        for (size_t i = 0, mi = m_images.size(); i < mi; i++)
        {
            m_core.createBuffer(bufferSize2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                m_uniformBuffersFromModel[i], m_uniformBuffersMemoryFromModel[i]);
        }
        ////////////////////////////////////////////

        ////////////////////////////////////////////
        // create buffers from world coords
        VkDeviceSize bufferSize3 = sizeof(OutputVertShader) * vertices.size();
        m_storageBufferWorldCoords.resize(m_images.size());
        m_storageBufferMemoryWorldCoords.resize(m_images.size());

        for (size_t i = 0, mi = m_images.size(); i < mi; i++)
        {
            m_core.createBuffer(bufferSize3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                m_storageBufferWorldCoords[i], m_storageBufferMemoryWorldCoords[i]);
        }
        ////////////////////////////////////////////
    }


	void VulkanDriver::updateUniformBuffer_(uint32_t currentImage_)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration< float, std::chrono::seconds::period >(currentTime - startTime).count();
	
		UniformView ubo = {};
		ubo.view = m_cameraView.getViewMatrix();
		ubo.proj = m_cameraView.getProjectionMatrix();

		void* data = nullptr;
		auto device = m_core.getDevice();
		vkMapMemory(device, m_uniformBuffersMemoryFromUbo[currentImage_], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, m_uniformBuffersMemoryFromUbo[currentImage_]);
        data = nullptr;


        vkMapMemory(device, m_uniformBuffersMemoryFromModel[currentImage_], 0,
                    sizeof(UniformModel) * m_modelData.size(), 0, &data);

        auto* pModelData = static_cast< UniformModel* >(data);
        for (size_t i = 0, mi = m_modelData.size(); i < mi; i++) {
            pModelData[i] = m_modelData.at(i);
        }
        vkUnmapMemory(device, m_uniformBuffersMemoryFromModel[currentImage_]);

        //todo
        updateWorldCoordsData(currentImage_);
	}


    void VulkanDriver::updateWorldCoordsData(uint32_t currentImage_)
    {
        void* data = nullptr;
        auto device = m_core.getDevice();
        m_worldCoords.resize(vertices.size());

        vkMapMemory(device, m_storageBufferMemoryWorldCoords[currentImage_], 0,
                sizeof(OutputVertShader) * vertices.size(), 0, &data);

        auto* pOutData = static_cast< OutputVertShader* >(data);
        for (size_t i = 0, mi = vertices.size(); i < mi; ++i)
        {
            m_worldCoords[i] = pOutData[i].world_coord;
        }
        vkUnmapMemory(device, m_storageBufferMemoryWorldCoords[currentImage_]);
    }


    void VulkanDriver::createDescriptorSetLayout_()
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
        uboLayoutBinding.binding            = 0;
        uboLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount    = 1;
        uboLayoutBinding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding modelLayoutBinding = {};
        modelLayoutBinding.binding            = 1;
        modelLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        modelLayoutBinding.descriptorCount    = 1;
        modelLayoutBinding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
        modelLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding coordsLayoutBinding = {};
        coordsLayoutBinding.binding            = 2;
        coordsLayoutBinding.descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        coordsLayoutBinding.descriptorCount    = 1;
        coordsLayoutBinding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
        coordsLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutBinding bindings[] {
                uboLayoutBinding,
                modelLayoutBinding,
                coordsLayoutBinding
        };

		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 3;
		layoutInfo.pBindings = bindings;

		if (vkCreateDescriptorSetLayout(m_core.getDevice(), &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create descriptor set layput!"));
		}
	}


	void VulkanDriver::createDescriptorPool_()
	{
		VkDescriptorPoolSize poolSizeUbo = {};
		poolSizeUbo.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizeUbo.descriptorCount = static_cast<uint32_t> (m_images.size());

        VkDescriptorPoolSize poolSizeModel = {};
        poolSizeModel.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        poolSizeModel.descriptorCount = static_cast<uint32_t> (m_images.size());

        VkDescriptorPoolSize poolSizeCoords = {};
        poolSizeCoords.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        poolSizeCoords.descriptorCount = static_cast<uint32_t> (m_images.size());

        VkDescriptorPoolSize poolSizes[] = {
                poolSizeUbo,
                poolSizeModel,
                poolSizeCoords
        };

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 3;
		poolInfo.pPoolSizes = poolSizes;
		poolInfo.maxSets = static_cast<uint32_t> (m_images.size());

		if (vkCreateDescriptorPool(m_core.getDevice(), &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create descriptor pool!"));
		}
	}


	void VulkanDriver::createDescriptorSets_()
	{
		std::vector< VkDescriptorSetLayout > layputs(m_images.size(), m_descriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = m_descriptorPool;
		allocInfo.descriptorSetCount = static_cast< uint32_t >(m_images.size());
		allocInfo.pSetLayouts = layputs.data();

		auto device_ = m_core.getDevice();

		m_descriptorSets.resize(m_images.size());
		if (vkAllocateDescriptorSets(device_, &allocInfo, m_descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to allocate descriptor sets!"));
		}

		for (size_t i = 0, mi = m_images.size(); i < mi; i++)
		{
			VkDescriptorBufferInfo bufferInfoUbo = {};
            bufferInfoUbo.buffer = m_uniformBuffersFromUbo[i];
            bufferInfoUbo.offset = 0;
            bufferInfoUbo.range = sizeof(UniformView);

			VkWriteDescriptorSet descriptorWriteUbo = {};
            descriptorWriteUbo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteUbo.dstSet = m_descriptorSets[i];
            descriptorWriteUbo.dstBinding = 0;
            descriptorWriteUbo.dstArrayElement = 0;
            descriptorWriteUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWriteUbo.descriptorCount = 1;
            descriptorWriteUbo.pBufferInfo = &bufferInfoUbo;
            descriptorWriteUbo.pImageInfo = nullptr;
            descriptorWriteUbo.pTexelBufferView = nullptr;

            ///

            VkDescriptorBufferInfo bufferInfoModel1 = {};
            bufferInfoModel1.buffer = m_uniformBuffersFromModel[i];
            bufferInfoModel1.offset = 0;
            bufferInfoModel1.range = sizeof(UniformModel) * vertices.size() / 3;

            VkWriteDescriptorSet descriptorWriteModel1 = {};
            descriptorWriteModel1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteModel1.dstSet = m_descriptorSets[i];
            descriptorWriteModel1.dstBinding = 1;
            descriptorWriteModel1.dstArrayElement = 0;
            descriptorWriteModel1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
            descriptorWriteModel1.descriptorCount = 1;
            descriptorWriteModel1.pBufferInfo = &bufferInfoModel1;
            descriptorWriteModel1.pImageInfo = nullptr;
            descriptorWriteModel1.pTexelBufferView = nullptr;

            ///

            VkDescriptorBufferInfo bufferInfoColors = {};
            bufferInfoColors.buffer = m_storageBufferWorldCoords[i];
            bufferInfoColors.offset = 0;
            bufferInfoColors.range = sizeof(OutputVertShader) * vertices.size();

            VkWriteDescriptorSet descriptorWriteColors = {};
            descriptorWriteColors.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWriteColors.dstSet = m_descriptorSets[i];
            descriptorWriteColors.dstBinding = 2;
            descriptorWriteColors.dstArrayElement = 0;
            descriptorWriteColors.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
            descriptorWriteColors.descriptorCount = 1;
            descriptorWriteColors.pBufferInfo = &bufferInfoColors;
            descriptorWriteColors.pImageInfo = nullptr;
            descriptorWriteColors.pTexelBufferView = nullptr;


            VkWriteDescriptorSet descriptorWrites[] = {
                    descriptorWriteUbo,
                    descriptorWriteModel1,
                    descriptorWriteColors
            };

			vkUpdateDescriptorSets(device_, 3, descriptorWrites, 0, nullptr);
		}
	}


	void VulkanDriver::render()
	{
		vkWaitForFences(m_core.getDevice(), 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex = 0;
		VkResult result = vkAcquireNextImageKHR(m_core.getDevice(), m_swapChainKHR, UINT64_MAX, m_imageAvailableSem[m_currentFrame], VK_NULL_HANDLE, &imageIndex); 
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			m_framebufferResized = false;
			recreateSwapChain_();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error(DEBUG_MSG("failed to acquire swap chain image!"));
		}
		

		// Check if a previous frame is using this image (i.e. there is its fence to wait on)
		if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
			vkWaitForFences(m_core.getDevice(), 1, &m_imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
		}
		// Mark the image as now being in use by this frame
		m_imagesInFlight[imageIndex] = m_inFlightFences[m_currentFrame];

		//todo
		updateUniformBuffer_(imageIndex);
		//todo

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { m_imageAvailableSem[m_currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &m_cmdBufs[imageIndex];

		VkSemaphore signalSemaphores[] = { m_renderFinishedSem[m_currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		vkResetFences(m_core.getDevice(), 1, &m_inFlightFences[m_currentFrame]);

		if (VK_SUCCESS != vkQueueSubmit(m_queue, 1, &submitInfo, m_inFlightFences[m_currentFrame])) {
			throw std::runtime_error(DEBUG_MSG("failed to submit draw command buffer!"));
		}

		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &m_swapChainKHR;
		presentInfo.pImageIndices = &imageIndex;

		presentInfo.pResults = nullptr;

		result = vkQueuePresentKHR(m_queue, &presentInfo); 
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
			m_framebufferResized = false;
			recreateSwapChain_();
		} 
		else if (result != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to present swap chain image!"));
		}

		//vkQueueWaitIdle(m_queue);
		m_currentFrame = (m_currentFrame + 1) % m_maxFramesInFlight;
	}


	void VulkanDriver::createRenderPass_()
	{
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format         = m_core.getSurfaceFormat().format;
		colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		//colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference attachRef = {};
		attachRef.attachment = 0;
		attachRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthAttachment = {};
		depthAttachment.format = m_core.findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		VkSubpassDescription subpassDesc = {};
		subpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDesc.colorAttachmentCount = 1;
		subpassDesc.pColorAttachments = &attachRef;
		subpassDesc.pDepthStencilAttachment = &depthAttachmentRef;

		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		std::array< VkAttachmentDescription, 2 > attachments = { colorAttachment, depthAttachment };

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDesc;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(m_core.getDevice(), &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create render pass!"));
		}
	}


	void VulkanDriver::createFramebuffer_()
	{
		m_fbs.resize(m_images.size());

		int wHeight = 0, wWidth = 0;
		glfwGetWindowSize(m_pWindow, &wWidth, &wHeight);

		for (uint32_t i = 0, mi = m_images.size(); i < mi; i++)
		{
			m_views[i] = m_core.createImageView(m_images[i], m_core.getSurfaceFormat().format, VK_IMAGE_ASPECT_COLOR_BIT);

			std::array< VkImageView, 2 > attachments = {
				m_views[i],
				m_depthImageView
			};

			VkFramebufferCreateInfo fbInfo = {};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.renderPass = m_renderPass;
			fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			fbInfo.pAttachments = attachments.data();
			fbInfo.width = wWidth;
			fbInfo.height = wHeight;
			fbInfo.layers = 1;

			if (VK_SUCCESS != vkCreateFramebuffer(m_core.getDevice(), &fbInfo, NULL, &m_fbs[i])) {
				throw std::runtime_error(DEBUG_MSG("failed to create frame buffer"));
			}
		}
	}


	VkShaderModule VulkanDriver::createShaderModule_(const std::vector< char >& source_)
	{
		VkShaderModuleCreateInfo createInfo = {}; 
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = source_.size();
		createInfo.pCode = reinterpret_cast< const uint32_t* >(source_.data());

		VkShaderModule sModule = nullptr;
		if (vkCreateShaderModule(m_core.getDevice(), &createInfo, nullptr, &sModule) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create shader module!"));
		}

		return sModule;
	}


	void VulkanDriver::createPipeline_()
	{
		auto vsCode = readFile(vert_shader_fname);
		auto fsCode = readFile(frag_shader_fname);

		VkShaderModule vertShader = createShaderModule_(vsCode);
		VkShaderModule fragShader = createShaderModule_(fsCode);


		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShader;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShader;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = {
			vertShaderStageInfo,
			fragShaderStageInfo
		};

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescription();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription; // Optional
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data(); // Optional

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		int wHeight = 0, wWidth = 0;
		glfwGetWindowSize(m_pWindow, &wWidth, &wHeight);

		VkViewport viewport = {};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = wWidth;
		viewport.height = wHeight;
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent.width = wWidth;
		scissor.extent.height = wHeight;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;
		
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.f;
		//todo
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		//rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		//end todo
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.f;
		rasterizer.depthBiasClamp = 0.f;
		rasterizer.depthBiasSlopeFactor = 0.f;

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.minDepthBounds = 0.0f;
		depthStencil.maxDepthBounds = 1.0f;
		depthStencil.stencilTestEnable = VK_FALSE;
		depthStencil.front = {};
		depthStencil.back = {};

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
			| VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		//VkDynamicState dynamicStates[] = {
		//	VK_DYNAMIC_STATE_VIEWPORT,
		//};
		//VkPipelineDynamicStateCreateInfo dynamicState = {};
		//dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		//dynamicState.dynamicStateCount = 1;
		//dynamicState.pDynamicStates = dynamicStates;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;

		if (vkCreatePipelineLayout(m_core.getDevice(), &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create pipeline layout!"));
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;

		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr;

		pipelineInfo.layout = m_pipelineLayout;

		pipelineInfo.renderPass = m_renderPass;
		pipelineInfo.subpass = 0;

		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(m_core.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
			throw std::runtime_error(DEBUG_MSG("failed to create graphics pipeline!"));
		}

		vkDestroyShaderModule(m_core.getDevice(), fragShader, nullptr);
		vkDestroyShaderModule(m_core.getDevice(), vertShader, nullptr);
	}


	void VulkanDriver::createSyncObjects_()
	{
		m_imageAvailableSem.resize(m_maxFramesInFlight);
		m_renderFinishedSem.resize(m_maxFramesInFlight);
		m_inFlightFences.resize(m_maxFramesInFlight);
		m_imagesInFlight.resize(m_images.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < m_maxFramesInFlight; i++) {
			if (vkCreateSemaphore(m_core.getDevice(), &createInfo, nullptr, &m_imageAvailableSem[i]) != VK_SUCCESS ||
				vkCreateSemaphore(m_core.getDevice(), &createInfo, nullptr, &m_renderFinishedSem[i]) != VK_SUCCESS ||
				vkCreateFence(m_core.getDevice(), &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS) {

				throw std::runtime_error(DEBUG_MSG("failed to create semaphores!"));
			}
		}
	}

}//namespace vks
