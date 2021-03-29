# returns the specified path component for the passed path
# posibble components are
# --file-name NAME_WE
# --file-name-ext NAME
# --parent-dir PATH
# @todo: create own components 
# e.g. parts dirs extension etc. consider creating an uri type
function(path_component path path_component)
  if("${path_component}" STREQUAL "--parent-dir")
    set(path_component PATH)
  elseif("${path_component}" STREQUAL "--file-name")
    set(path_component NAME_WE)
  elseif("${path_component}" STREQUAL "--file-name-ext")
    set(path_component NAME)
  endif()
  get_filename_component(res "${path}" "${path_component}")
  return_ref(res)
endfunction()



