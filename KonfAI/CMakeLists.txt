#-----------------------------------------------------------------------------
set(MODULE_NAME KonfAI)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/UI/${MODULE_NAME}Core.ui
  Resources/UI/${MODULE_NAME}AppTemplate.ui
  Resources/UI/${MODULE_NAME}MetricsPanel.ui
  
  Resources/Icons/download.png
  Resources/Icons/folder.png
  Resources/Icons/gear.png
  Resources/Icons/${MODULE_NAME}.png
  Resources/Icons/refresh.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()