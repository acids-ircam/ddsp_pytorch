


# overwrites project so that it can be registered
macro(project)
  set(parent_project_name "${PROJECT_NAME}")
  _project(${ARGN})
  set(project_name "${PROJECT_NAME}") 
  project_register(${ARGN})
  event_emit("project" ${ARGN})
endmacro()




# function(project name)
#   set(parent_project_name "${PROJECT_NAME}")
#   _project("${name}" ${ARGN})
#   set(project_name "${PROJECT_NAME}") 
#   project_register("${name}" ${ARGN})
#   event_emit("project" "${name}" ${ARGN})

#   promote(
#     PROJECT_SOURCE_DIR
#     PROJECT_BINARY_DIR
#     PROJECT_VERSION
#     PROJECT_VERSION_MAJOR
#     PROJECT_VERSION_MINOR
#     PROJECT_VERSION_PATCH
#     PROJECT_VERSION_TWEAK
#     "${name}_SOURCE_DIR"
#     "${name}_BINARY_DIR"
#     "${name}_VERSION"
#     "${name}_VERSION_MAJOR"
#     "${name}_VERSION_MINOR"
#     "${name}_VERSION_PATCH"
#     "${name}_VERSION_TWEAK"
#     )

# endfunction()

