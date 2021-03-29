
## writes a file_map to the pwd.
## empty directories are not created
## fm is parsed according to obj()
function(file_map_write fm)


  # define callbacks for building result
  function(fmw_dir_begin)
    map_tryget(${context} current_key)
    ans(key)
    if("${map_length}" EQUAL 0)
      return()
    endif()
    if(key)
      pushd("${key}" --create)
    else()
      pushd()
    endif()
  endfunction()
  function(fmw_dir_end)
    if(NOT "${map_length}" EQUAL 0)    
      popd()
    endif()
  endfunction()
  function(fmw_path_change)
    map_set(${context} current_key "${map_element_key}")
  endfunction()

  function(fmw_file)
    map_get(${context} current_key) 
    ans(key)
    fwrite("${key}" "${node}")
  endfunction()

   map()
    kv(value              fmw_file)
    kv(map_begin          fmw_dir_begin)
    kv(map_end            fmw_dir_end)
    kv(list_begin         fmw_file)
    kv(map_element_begin  fmw_path_change)
  end()
  ans(file_map_write_cbs)
  function_import_table(${file_map_write_cbs} file_map_write_callback)

  # function definition
  function(file_map_write fm)            
    obj("${fm}")
    ans(fm)

    map_new()
    ans(context)
    dfs_callback(file_map_write_callback ${fm} ${ARGN})
    map_tryget(${context} files)
    return_ans()  
  endfunction()
  #delegate
  file_map_write(${fm} ${ARGN})
  return_ans()
endfunction()

function(file_map_read)
  path("${ARGN}")
  ans(path)
  message("path ${path}")
  
  file(GLOB_RECURSE paths RELATIVE "${path}" ${path}/**)

  message("paths ${paths}")



  return_ans()

endfunction()