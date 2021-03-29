## `(<relative path> <file>...)-> null` 
##
##  sets the source group for the specified files to their relative dirs 
function(source_group_relative  base_dir)
  foreach(file ${ARGN})
    path_parent_dir("${file}")
    ans(parent_dir)
    
    path_relative("${base_dir}" "${parent_dir}")
    ans(relative_dir_path)

    if(NOT "${relative_dir_path}" STREQUAL ".")
        string(REPLACE "../" "" relative_dir_path "${relative_dir_path}")
        string(REPLACE "/" "\\" relative_dir_path "${relative_dir_path}" )
        
        source_group(${relative_dir_path} FILES ${file})

    endif()

    



  endforeach()
  return()
endfunction()
