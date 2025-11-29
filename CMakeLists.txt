cmake_minimum_required(VERSION 3.5)

project(SlicerKonfAI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/vboussot/SlicerKonfAI")
set(EXTENSION_CATEGORY "Deep Learning")
set(EXTENSION_CONTRIBUTORS "Valentin Boussot (University of Rennes, France)")
set(EXTENSION_DESCRIPTION "An extension enabling fast and configurable deep learning inference in 3D Slicer using KonfAI models from the Hugging Face repository.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/SlicerKonfAI.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/docs/Inference.png")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerKonfAI)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
