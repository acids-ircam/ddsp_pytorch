## async(<callable>(<args...)) -> <process handle>
##
## executes a callable asynchroniously 
##
## todo: 
##   capture necessary scope vars
##   include further files for custom functions     
##   environment vars
  function(async callable)
    cmakepp_config(base_dir)
    ans(base_dir)
    set(args ${ARGN})
    list_pop_front(args)
    list_pop_back(args)
    qm_serialize(${args})
    ans(arguments)
    path_temp()
    ans(result_file)
    pwd()
    ans(pwd)
    set(code
      "
        include(\"${base_dir}/cmakepp.cmake\")
        cd(\"${pwd}\")
        ${arguments}
        ans(arguments)
        address_get(\"\${arguments}\")
        ans(arguments)
        function_import(\"${callable}\" as __async_call)
        message(\${arguments})
        __async_call(\${arguments})
        ans(async_result)
        qm_write(\"${result_file}\" \"\${async_result}\")
      ")
    process_start_script("${code}")
    ans(process_handle)
    map_set(${process_handle} result_file "${result_file}")
    return_ref(process_handle)
  endfunction()
