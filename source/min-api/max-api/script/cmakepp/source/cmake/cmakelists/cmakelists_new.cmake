  function(cmakelists_new source)
    set(cmakelists_path "${ARGN}")
    if(NOT cmakelists_path)
        set(cmakelists_path .)
    endif()
    if(NOT "${cmakelists_path}" MATCHES "CMakeLists\\.txt$")
        set(cmakelists_path "${cmakelists_path}/CMakeLists.txt")
    endif()
    path_qualify(cmakelists_path)

    map_new()
    ans(cmakelists)

    cmake_token_range("${source}")

    ans_extract(begin end)    
    map_set(${cmakelists} begin ${begin})
    map_set(${cmakelists} end ${end} )
    map_set(${cmakelists} range ${begin} ${end})
    map_set(${cmakelists} path "${cmakelists_path}")

    return_ref(cmakelists)
  endfunction()
