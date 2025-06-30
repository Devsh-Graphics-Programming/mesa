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
