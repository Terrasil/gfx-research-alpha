cmake_minimum_required(VERSION 3.25)

project(gfx-research-alpha VERSION 3.2.1 LANGUAGES C CXX)

set(GFX_RESEARCH_BASE_GIT_TAG main CACHE STRING "gfx-research-base Git revision")
set(GFX_RESEARCH_BASE_LOCAL_PATH "" CACHE PATH "Optional local gfx-research-base checkout")

if(GFX_RESEARCH_BASE_LOCAL_PATH)
    add_subdirectory("${GFX_RESEARCH_BASE_LOCAL_PATH}" "${CMAKE_BINARY_DIR}/gfx-research-base")
else()
    if(NOT DEFINED CPM_SOURCE_CACHE)
        set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/.cpm-cache" CACHE PATH "CPM source cache")
    endif()

    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
    set(GFX_RESEARCH_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty" CACHE PATH "Third-party source directory")

    CPMAddPackage(
        NAME gfx_research_base
        GITHUB_REPOSITORY Terrasil/gfx-research-base
        GIT_TAG "${GFX_RESEARCH_BASE_GIT_TAG}"
        SOURCE_DIR "${GFX_RESEARCH_THIRD_PARTY_DIR}/gfx-research-base"
        OPTIONS
            "GFX_RESEARCH_BASE_BUILD_EXAMPLE OFF"
    )
endif()

# gfx-research-base main exports gfx::research_base. Keep fallbacks only for older checkouts.
if(TARGET gfx::research_base)
    set(GFX_RESEARCH_BASE_TARGET gfx::research_base)
elseif(TARGET gfx-research-base)
    set(GFX_RESEARCH_BASE_TARGET gfx-research-base)
elseif(TARGET gfx::research-base)
    set(GFX_RESEARCH_BASE_TARGET gfx::research-base)
else()
    message(FATAL_ERROR "gfx-research-base was added, but no supported library target was exported")
endif()
message(STATUS "gfx-research-alpha: linking ${GFX_RESEARCH_BASE_TARGET}")

# The current base project intentionally places downloaded third-party sources
# under ${CMAKE_SOURCE_DIR}/thirdparty. Use the path exported by the base instead
# of assuming they live inside the gfx-research-base checkout.
if(NOT DEFINED GFX_RESEARCH_BASE_THIRD_PARTY_DIR)
    set(GFX_RESEARCH_BASE_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty")
endif()

add_executable(gfx-research-alpha
    src/layered_wboit_target.cpp
    src/main.cpp
    src/textured_obj_scene.cpp
    src/wboit_target.cpp
)
target_compile_features(gfx-research-alpha PRIVATE cxx_std_20)
target_include_directories(gfx-research-alpha PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GFX_RESEARCH_BASE_THIRD_PARTY_DIR}/tinyobjloader"
)
target_link_libraries(gfx-research-alpha PRIVATE ${GFX_RESEARCH_BASE_TARGET})
target_compile_definitions(gfx-research-alpha PRIVATE GFX_RESEARCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

if(COMMAND gfx_research_configure_executable)
    gfx_research_configure_executable(gfx-research-alpha)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(gfx-research-alpha-reference-tests tests/reference_math.cpp)
    target_compile_features(gfx-research-alpha-reference-tests PRIVATE cxx_std_20)
    add_test(NAME gfx-research-alpha-reference COMMAND gfx-research-alpha-reference-tests)
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_SUPPRESS_REGENERATION ON CACHE BOOL "Disable automatic build-system regeneration" FORCE)
endif()
