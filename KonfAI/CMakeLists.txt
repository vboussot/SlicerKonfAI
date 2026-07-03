#-----------------------------------------------------------------------------
set(MODULE_NAME KonfAI)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/platform_utils.py
  ${MODULE_NAME}Lib/logic/__init__.py
  ${MODULE_NAME}Lib/logic/dependencies.py
  ${MODULE_NAME}Lib/logic/hf.py
  ${MODULE_NAME}Lib/logic/mrml.py
  ${MODULE_NAME}Lib/logic/process.py
  ${MODULE_NAME}Lib/logic/servers.py
  ${MODULE_NAME}Lib/widgets/__init__.py
  ${MODULE_NAME}Lib/widgets/app_template.py
  ${MODULE_NAME}Lib/widgets/chip_selector.py
  ${MODULE_NAME}Lib/widgets/core_widget.py
  ${MODULE_NAME}Lib/widgets/helpers.py
  ${MODULE_NAME}Lib/widgets/io_selectors.py
  ${MODULE_NAME}Lib/widgets/metrics_panel.py
  ${MODULE_NAME}Lib/widgets/dialogs/__init__.py
  ${MODULE_NAME}Lib/widgets/dialogs/download_files.py
  ${MODULE_NAME}Lib/widgets/dialogs/remote_server.py
  ${MODULE_NAME}Lib/widgets/panels/__init__.py
  ${MODULE_NAME}Lib/widgets/panels/base.py
  ${MODULE_NAME}Lib/widgets/panels/inference.py
  ${MODULE_NAME}Lib/widgets/panels/qa.py
  ${MODULE_NAME}Lib/widgets/panels/selection.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/UI/${MODULE_NAME}Core.ui
  Resources/UI/${MODULE_NAME}AppInference.ui
  Resources/UI/${MODULE_NAME}AppQA.ui
  Resources/UI/${MODULE_NAME}AppSelection.ui
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