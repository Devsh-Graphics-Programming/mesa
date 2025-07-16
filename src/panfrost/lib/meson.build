# Copyright © 2018 Rob Clark
# Copyright © 2019 Collabora
# SPDX-License-Identifier: MIT

if with_gallium_panfrost or with_panfrost_vk
  subdir('kmod')
endif

pixel_format_versions = ['5', '6', '7', '9', '10', '12', '13']
libpanfrost_pixel_format = []

deps_for_libpanfrost = [dep_libdrm, idep_pan_packers, idep_mesautil]

foreach ver : pixel_format_versions
  libpanfrost_pixel_format += static_library(
    'pan-format-v' + ver,
    ['pan_format.c', pan_packers],
    include_directories : [inc_src, inc_include, inc_panfrost],
    c_args : ['-DPAN_ARCH=' + ver],
    gnu_symbol_visibility : 'hidden',
    dependencies : [deps_for_libpanfrost, dep_valgrind],
)
endforeach

libpanfrost_per_arch = []

libpanfrost_per_arch_files = [ 'pan_blend.c', 'pan_shader.c', 'pan_texture.c' ]
if with_gallium_panfrost or with_panfrost_vk
  libpanfrost_per_arch_files += 'pan_desc.c'
endif

foreach ver : ['4', '5', '6', '7', '9', '10', '12', '13']
  libpanfrost_per_arch += static_library(
    'pan-arch-v' + ver, libpanfrost_per_arch_files,
    include_directories : [inc_include, inc_src, inc_panfrost],
    c_args : ['-DPAN_ARCH=' + ver],
    gnu_symbol_visibility : 'hidden',
    dependencies : [deps_for_libpanfrost, idep_nir],
  )
endforeach

libpanfrost_lib_files = files(
  'pan_encoder.h',

  'pan_blend.c',
  'pan_clear.c',
  'pan_earlyzs.c',
  'pan_samples.c',
  'pan_shader.c',
  'pan_tiler.c',
  'pan_layout.c',
  'pan_scratch.c',
  'pan_util.c',
  'pan_afbc.c',
)

if with_gallium_panfrost or with_panfrost_vk
  libpanfrost_lib_files += files('pan_props.c')
endif

libpanfrost_link_with = [libpanfrost_pixel_format, libpanfrost_per_arch]
if with_gallium_panfrost or with_panfrost_vk
  libpanfrost_link_with += libpankmod_lib
endif

libpanfrost_lib = static_library(
  'panfrost_lib',
  [libpanfrost_lib_files, pan_packers],
  include_directories : [inc_include, inc_src, inc_panfrost],
  c_args : [no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  dependencies: [dep_libdrm, idep_nir, idep_mesautil],
  build_by_default : false,
  link_with: libpanfrost_link_with,
)

libpanfrost_dependencies = [deps_for_libpanfrost, idep_nir]
if with_gallium_panfrost or with_panfrost_vk
  libpanfrost_dependencies = libpankmod_dep
endif

libpanfrost_dep = declare_dependency(
  link_with: [libpanfrost_lib, libpanfrost_decode, libpanfrost_midgard, libpanfrost_bifrost, libpanfrost_pixel_format, libpanfrost_per_arch],
  include_directories: [inc_include, inc_src, inc_panfrost],
  dependencies: libpanfrost_dependencies,
)

if with_tests
  test(
    'panfrost_attribute',
    executable(
      'panfrost_attribute_test',
      files('tests/test-attribute.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_blend',
    executable(
      'panfrost_blend_test',
      files('tests/test-blend.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_clear',
    executable(
      'panfrost_clear_test',
      files('tests/test-clear.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_tests',
    executable(
      'panfrost_tests',
      files(
        'tests/test-earlyzs.cpp',
        'tests/test-layout.cpp',
      ),
      cpp_args : [cpp_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src, inc_panfrost],
      dependencies: [idep_gtest, libpanfrost_dep],
    ),
    suite : ['panfrost'],
    protocol : 'gtest',
  )
endif
