/*
 * Copyright © 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "vk_nir_lower_descriptor_heaps.h"

static bool
fixup_derefs(nir_builder *b, nir_intrinsic_instr *intrin, void *Data)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_deref:
   case nir_intrinsic_store_deref:
   case nir_intrinsic_load_deref_block_intel:
   case nir_intrinsic_store_deref_block_intel:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap: {
      nir_deref_path path;
      nir_deref_path_init(&path, nir_def_as_deref(intrin->src[0].ssa), NULL);
      bool is_broken_ubo = false;
      for (uint32_t i = 0; path.path[i] != NULL; i++) {
         if ((path.path[i]->modes & nir_var_uniform) == 0)
            break;
         nir_intrinsic_instr *intrin;
         if (path.path[i]->deref_type == nir_deref_type_cast &&
             (intrin = nir_src_as_intrinsic(path.path[i]->parent)) &&
             intrin->intrinsic == nir_intrinsic_load_buffer_ptr_deref &&
             nir_intrinsic_resource_type(intrin) == VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT) {
            is_broken_ubo = true;
            break;
         }
      }

      if (is_broken_ubo) {
         for (uint32_t i = 0; path.path[i] != NULL; i++) {
            path.path[i]->modes &= ~nir_var_uniform;
            path.path[i]->modes |= nir_var_mem_ubo;
         }
      }

      nir_deref_path_finish(&path);
      return false;
   }

   case nir_intrinsic_memcpy_deref: {
   }

   default:
      return false;
   }
}

bool
vk_nir_fixup_ubo_derefs(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, fixup_derefs, nir_metadata_all, NULL);
}
