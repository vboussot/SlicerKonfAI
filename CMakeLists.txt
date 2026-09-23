cmake_minimum_required(VERSION 3.5)

project(KonfAI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/vboussot/SlicerKonfAI")
set(EXTENSION_CATEGORY "Pipelines")
set(EXTENSION_CONTRIBUTORS "Valentin Boussot (University of Rennes, France)")
set(EXTENSION_DESCRIPTION "Run KonfAI Apps (segmentation, synthetic CT, registration) from Hugging Face or local folders on the volumes of the scene, with evaluation, uncertainty estimation, and local or remote GPU execution.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/KonfAI.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/Screenshots/main.png https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/Screenshots/tutorial/03-result.jpg https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/Screenshots/tutorial/04-evaluation.jpg https://raw.githubusercontent.com/vboussot/SlicerKonfAI/main/Screenshots/tutorial/09-remote.jpg")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(KonfAI)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
