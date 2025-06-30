/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */
#include "nir/radv_meta_nir.h"
#include "radv_cs.h"
#include "radv_formats.h"
#include "radv_meta.h"

/* Copy memory->memory. */
static VkResult
get_compute_copy_memory_indirect_preprocess_pipeline(struct radv_device *device, VkPipeline *pipeline_out,
                                                     VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_MEMORY_INDIRECT_PREPROCESS_CS;
   VkResult result;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 24,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_copy_memory_indirect_preprocess_cs(device);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
radv_compute_copy_memory_indirect_preprocess(struct radv_cmd_buffer *cmd_buffer,
                                             const VkCopyMemoryIndirectInfoKHR *pCopyMemoryIndirectInfo,
                                             uint64_t upload_addr)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint32_t copy_count = pCopyMemoryIndirectInfo->copyCount;
   struct radv_meta_saved_state saved_state;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_compute_copy_memory_indirect_preprocess_pipeline(device, &pipeline, &layout);
   if (result != VK_SUCCESS)
      return result;

   radv_meta_save(&saved_state, cmd_buffer, RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS);

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   const uint32_t constants[6] = {
      pCopyMemoryIndirectInfo->copyAddressRange.address,
      pCopyMemoryIndirectInfo->copyAddressRange.address >> 32,
      pCopyMemoryIndirectInfo->copyAddressRange.stride,
      pCopyMemoryIndirectInfo->copyAddressRange.stride >> 32,
      upload_addr,
      upload_addr >> 32,
   };

   const VkPushConstantsInfoKHR pc_info = {
      .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
      .layout = layout,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(constants),
      .pValues = constants,
   };

   radv_CmdPushConstants2(radv_cmd_buffer_to_handle(cmd_buffer), &pc_info);

   radv_unaligned_dispatch(cmd_buffer, copy_count, 1, 1);

   radv_meta_restore(&saved_state, cmd_buffer);

   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                   radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                         VK_ACCESS_2_SHADER_WRITE_BIT, 0, NULL, NULL);

   return VK_SUCCESS;
}

static VkResult
get_compute_copy_memory_indirect_pipeline(struct radv_device *device, VkPipeline *pipeline_out,
                                          VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_MEMORY_INDIRECT_CS;
   VkResult result;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 8,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_copy_memory_indirect_cs(device);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

void
radv_compute_copy_memory_indirect(struct radv_cmd_buffer *cmd_buffer,
                                  const VkCopyMemoryIndirectInfoKHR *pCopyMemoryIndirectInfo)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint32_t copy_count = pCopyMemoryIndirectInfo->copyCount;
   struct radv_meta_saved_state saved_state;
   VkPipelineLayout layout;
   uint32_t alloc_offset;
   uint32_t *alloc_ptr;
   VkPipeline pipeline;
   VkResult result;

   if (!radv_cmd_buffer_upload_alloc_aligned(cmd_buffer, copy_count * sizeof(VkDispatchIndirectCommand), 4,
                                             &alloc_offset, (void *)&alloc_ptr)) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const uint64_t upload_addr = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + alloc_offset;

   result = radv_compute_copy_memory_indirect_preprocess(cmd_buffer, pCopyMemoryIndirectInfo, upload_addr);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   result = get_compute_copy_memory_indirect_pipeline(device, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_meta_save(&saved_state, cmd_buffer, RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS);

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   for (uint32_t i = 0; i < copy_count; i++) {
      const uint64_t copy_addr =
         pCopyMemoryIndirectInfo->copyAddressRange.address + i * pCopyMemoryIndirectInfo->copyAddressRange.stride;

      const VkPushConstantsInfoKHR pc_info = {
         .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
         .layout = layout,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .offset = 0,
         .size = sizeof(copy_addr),
         .pValues = &copy_addr,
      };

      radv_CmdPushConstants2(radv_cmd_buffer_to_handle(cmd_buffer), &pc_info);

      const struct radv_dispatch_info info = {
         .indirect_va = upload_addr + i * sizeof(VkDispatchIndirectCommand),
         .unaligned = true,
      };

      radv_compute_dispatch(cmd_buffer, &info);
   }

   radv_meta_restore(&saved_state, cmd_buffer);
}

/* Copy memory->image. */
static VkResult
get_compute_copy_memory_to_image_indirect_preprocess_pipeline(struct radv_device *device, VkPipeline *pipeline_out,
                                                              VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_MEMORY_TO_IMAGE_INDIRECT_PREPROCESS_CS;
   VkResult result;

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 32,
   };

   result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, NULL, &pc_range, &key, sizeof(key),
                                        layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_copy_memory_to_image_indirect_preprocess_cs(device);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
radv_compute_copy_memory_to_image_indirect_preprocess(
   struct radv_cmd_buffer *cmd_buffer, const VkCopyMemoryToImageIndirectInfoKHR *pCopyMemoryToImageIndirectInfo,
   uint64_t upload_addr)
{
   VK_FROM_HANDLE(radv_image, dst_image, pCopyMemoryToImageIndirectInfo->dstImage);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const uint32_t copy_count = pCopyMemoryToImageIndirectInfo->copyCount;
   struct radv_meta_saved_state saved_state;
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   result = get_compute_copy_memory_to_image_indirect_preprocess_pipeline(device, &pipeline, &layout);
   if (result != VK_SUCCESS)
      return result;

   radv_meta_save(&saved_state, cmd_buffer, RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS);

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   const struct util_format_description *fmt = vk_format_description(dst_image->vk.format);

   const uint32_t constants[8] = {
      pCopyMemoryToImageIndirectInfo->copyAddressRange.address,
      pCopyMemoryToImageIndirectInfo->copyAddressRange.address >> 32,
      pCopyMemoryToImageIndirectInfo->copyAddressRange.stride,
      pCopyMemoryToImageIndirectInfo->copyAddressRange.stride >> 32,
      upload_addr,
      upload_addr >> 32,
      fmt->block.width,
      fmt->block.height,
   };

   const VkPushConstantsInfoKHR pc_info = {
      .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
      .layout = layout,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(constants),
      .pValues = constants,
   };

   radv_CmdPushConstants2(radv_cmd_buffer_to_handle(cmd_buffer), &pc_info);

   radv_unaligned_dispatch(cmd_buffer, copy_count, 1, 1);

   radv_meta_restore(&saved_state, cmd_buffer);

   cmd_buffer->state.flush_bits |= RADV_CMD_FLAG_CS_PARTIAL_FLUSH | RADV_CMD_FLAG_INV_VCACHE |
                                   radv_src_access_flush(cmd_buffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                         VK_ACCESS_2_SHADER_WRITE_BIT, 0, NULL, NULL);

   return VK_SUCCESS;
}

static VkResult
get_compute_copy_memory_to_image_indirect_pipeline_layout(struct radv_device *device, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_MEMORY_TO_IMAGE_INDIRECT_CS;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 48,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_copy_memory_to_image_indirect_key {
   enum radv_meta_object_key_type type;
   bool is_3d;
};

static VkResult
get_compute_copy_memory_to_image_indirect_pipeline(struct radv_device *device, const struct radv_image *image,
                                                   VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   struct radv_copy_memory_to_image_indirect_key key;
   VkResult result;

   result = get_compute_copy_memory_to_image_indirect_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = RADV_META_OBJECT_KEY_COPY_MEMORY_TO_IMAGE_INDIRECT_CS;
   key.is_3d = is_3d;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_copy_memory_to_image_indirect_cs(device, is_3d);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static VkResult
get_compute_copy_memory_to_image_r32g32b32_indirect_pipeline_layout(struct radv_device *device,
                                                                    VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_MEMORY_TO_IMAGE_R32G32B32_INDIRECT_CS;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = 44,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

static VkResult
get_compute_copy_memory_to_image_r32g32b32_indirect_pipeline(struct radv_device *device, const struct radv_image *image,
                                                             VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_COPY_MEMORY_TO_IMAGE_R32G32B32_INDIRECT_CS;
   VkResult result;

   result = get_compute_copy_memory_to_image_r32g32b32_indirect_pipeline_layout(device, layout_out);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *cs = radv_meta_nir_build_copy_memory_to_image_r32g32b32_indirect_cs(device);

   const VkPipelineShaderStageCreateInfo stage_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = vk_shader_module_handle_from_nir(cs),
      .pName = "main",
      .pSpecializationInfo = NULL,
   };

   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = stage_info,
      .flags = 0,
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device, &pipeline_info, &key, sizeof(key),
                                            pipeline_out);

   ralloc_free(cs);
   return result;
}

static void
radv_compute_copy_memory_to_image_r32g32b32_indirect(
   struct radv_cmd_buffer *cmd_buffer, const VkCopyMemoryToImageIndirectInfoKHR *pCopyMemoryToImageIndirectInfo)
{
   VK_FROM_HANDLE(radv_image, dst_image, pCopyMemoryToImageIndirectInfo->dstImage);
   const uint32_t copy_count = pCopyMemoryToImageIndirectInfo->copyCount;
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;
   struct radv_meta_saved_state saved_state;
   VkPipelineLayout layout;
   uint32_t alloc_offset;
   uint32_t *alloc_ptr;
   VkPipeline pipeline;
   VkResult result;

   assert(dst_image->vk.mip_levels == 1 && dst_image->vk.array_layers == 1);

   if (!radv_cmd_buffer_upload_alloc_aligned(cmd_buffer, copy_count * sizeof(VkDispatchIndirectCommand), 4,
                                             &alloc_offset, (void *)&alloc_ptr)) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const uint64_t upload_addr = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + alloc_offset;

   result =
      radv_compute_copy_memory_to_image_indirect_preprocess(cmd_buffer, pCopyMemoryToImageIndirectInfo, upload_addr);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   result = get_compute_copy_memory_to_image_r32g32b32_indirect_pipeline(device, dst_image, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_meta_save(&saved_state, cmd_buffer,
                  RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS | RADV_META_SAVE_DESCRIPTORS);

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   const struct util_format_description *fmt = vk_format_description(dst_image->vk.format);

   for (uint32_t i = 0; i < copy_count; i++) {
      const VkImageSubresourceLayers *imageSubresource = &pCopyMemoryToImageIndirectInfo->pImageSubresources[i];
      const VkImageAspectFlags aspect_mask = imageSubresource->aspectMask;
      const unsigned bind_idx = dst_image->disjoint ? radv_plane_from_aspect(aspect_mask) : 0;

      radv_cs_add_buffer(device->ws, cs->b, dst_image->bindings[bind_idx].bo);

      struct radv_meta_blit2d_surf img_bsurf = radv_blit_surf_for_image_level_layer(
         dst_image, pCopyMemoryToImageIndirectInfo->dstImageLayout, imageSubresource);

      radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1,
                                 (VkDescriptorGetInfoEXT[]){{
                                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                                    .data.pStorageTexelBuffer =
                                       &(VkDescriptorAddressInfoEXT){
                                          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                          .address = dst_image->bindings[0].addr,
                                          .range = dst_image->size,
                                          .format = radv_meta_get_96bit_channel_format(dst_image->vk.format),
                                       },
                                 }});

      uint32_t buffer_view_desc[4];
      radv_make_texel_buffer_descriptor(device, 0, img_bsurf.format, ~0, buffer_view_desc);

      const uint64_t copy_addr = pCopyMemoryToImageIndirectInfo->copyAddressRange.address +
                                 i * pCopyMemoryToImageIndirectInfo->copyAddressRange.stride;

      unsigned stride = dst_image->planes[0].surface.u.gfx9.surf_pitch;

      const uint32_t constants[11] = {
         copy_addr,           copy_addr >> 32, fmt->block.width,    fmt->block.height,   fmt->block.depth,
         fmt->block.bits / 8, stride,          buffer_view_desc[0], buffer_view_desc[1], buffer_view_desc[2],
         buffer_view_desc[3],
      };

      const VkPushConstantsInfoKHR pc_info = {
         .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
         .layout = layout,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .offset = 0,
         .size = sizeof(constants),
         .pValues = constants,
      };

      radv_CmdPushConstants2(radv_cmd_buffer_to_handle(cmd_buffer), &pc_info);

      const struct radv_dispatch_info info = {
         .indirect_va = upload_addr + i * sizeof(VkDispatchIndirectCommand),
         .unaligned = true,
      };

      radv_compute_dispatch(cmd_buffer, &info);
   }

   radv_meta_restore(&saved_state, cmd_buffer);
}

void
radv_compute_copy_memory_to_image_indirect(struct radv_cmd_buffer *cmd_buffer,
                                           const VkCopyMemoryToImageIndirectInfoKHR *pCopyMemoryToImageIndirectInfo)
{
   VK_FROM_HANDLE(radv_image, dst_image, pCopyMemoryToImageIndirectInfo->dstImage);
   const uint32_t copy_count = pCopyMemoryToImageIndirectInfo->copyCount;
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   struct radv_cmd_stream *cs = cmd_buffer->cs;
   struct radv_meta_saved_state saved_state;
   VkPipelineLayout layout;
   uint32_t alloc_offset;
   uint32_t *alloc_ptr;
   VkPipeline pipeline;
   VkResult result;

   if (vk_format_is_96bit(dst_image->vk.format)) {
      radv_compute_copy_memory_to_image_r32g32b32_indirect(cmd_buffer, pCopyMemoryToImageIndirectInfo);
      return;
   }

   if (!radv_cmd_buffer_upload_alloc_aligned(cmd_buffer, copy_count * sizeof(VkDispatchIndirectCommand), 4,
                                             &alloc_offset, (void *)&alloc_ptr)) {
      vk_command_buffer_set_error(&cmd_buffer->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const uint64_t upload_addr = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + alloc_offset;

   result =
      radv_compute_copy_memory_to_image_indirect_preprocess(cmd_buffer, pCopyMemoryToImageIndirectInfo, upload_addr);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   result = get_compute_copy_memory_to_image_indirect_pipeline(device, dst_image, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   radv_meta_save(&saved_state, cmd_buffer,
                  RADV_META_SAVE_COMPUTE_PIPELINE | RADV_META_SAVE_CONSTANTS | RADV_META_SAVE_DESCRIPTORS);

   radv_CmdBindPipeline(radv_cmd_buffer_to_handle(cmd_buffer), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   for (uint32_t i = 0; i < copy_count; i++) {
      const VkImageSubresourceLayers *imageSubresource = &pCopyMemoryToImageIndirectInfo->pImageSubresources[i];
      const VkImageAspectFlags aspect_mask = imageSubresource->aspectMask;
      const unsigned bind_idx = dst_image->disjoint ? radv_plane_from_aspect(aspect_mask) : 0;
      VkFormat format = vk_format_get_aspect_format(dst_image->vk.format, imageSubresource->aspectMask);
      const struct util_format_description *fmt = vk_format_description(format);
      struct radv_image_view dst_iview;

      radv_cs_add_buffer(device->ws, cs->b, dst_image->bindings[bind_idx].bo);

      struct radv_meta_blit2d_surf img_bsurf = radv_blit_surf_for_image_level_layer(
         dst_image, pCopyMemoryToImageIndirectInfo->dstImageLayout, imageSubresource);

      if (!radv_is_buffer_format_supported(img_bsurf.format, NULL))
         img_bsurf.format = vk_format_for_size(vk_format_get_blocksize(img_bsurf.format));

      const uint32_t slice_count = vk_image_subresource_layer_count(&dst_image->vk, imageSubresource);

      for (uint32_t slice = 0; slice < slice_count; slice++) {
         const VkImageViewUsageCreateInfo iview_usage_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
         };

         radv_image_view_init(&dst_iview, device,
                              &(VkImageViewCreateInfo){
                                 .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .pNext = &iview_usage_info,
                                 .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                                 .image = radv_image_to_handle(dst_image),
                                 .viewType = radv_meta_get_view_type(dst_image),
                                 .format = img_bsurf.format,
                                 .subresourceRange = {.aspectMask = img_bsurf.aspect_mask,
                                                      .baseMipLevel = img_bsurf.level,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = img_bsurf.layer + slice,
                                                      .layerCount = 1},
                              },
                              NULL);

         radv_meta_bind_descriptors(
            cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1,
            (VkDescriptorGetInfoEXT[]){{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        .data.pStorageImage = (VkDescriptorImageInfo[]){
                                           {
                                              .sampler = VK_NULL_HANDLE,
                                              .imageView = radv_image_view_to_handle(&dst_iview),
                                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                           },
                                        }}});

         uint32_t buffer_view_desc[4];
         radv_make_texel_buffer_descriptor(device, 0, img_bsurf.format, ~0, buffer_view_desc);

         const uint64_t copy_addr = pCopyMemoryToImageIndirectInfo->copyAddressRange.address +
                                    i * pCopyMemoryToImageIndirectInfo->copyAddressRange.stride;

         const uint32_t constants[12] = {
            copy_addr,           copy_addr >> 32, fmt->block.width,    fmt->block.height,   fmt->block.depth,
            fmt->block.bits / 8, slice,           buffer_view_desc[0], buffer_view_desc[1], buffer_view_desc[2],
            buffer_view_desc[3], img_bsurf.layer};

         const VkPushConstantsInfoKHR pc_info = {
            .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
            .layout = layout,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(constants),
            .pValues = constants,
         };

         radv_CmdPushConstants2(radv_cmd_buffer_to_handle(cmd_buffer), &pc_info);

         const struct radv_dispatch_info info = {
            .indirect_va = upload_addr + i * sizeof(VkDispatchIndirectCommand),
            .unaligned = true,
         };

         radv_compute_dispatch(cmd_buffer, &info);

         radv_image_view_finish(&dst_iview);
      }
   }

   radv_meta_restore(&saved_state, cmd_buffer);
}
