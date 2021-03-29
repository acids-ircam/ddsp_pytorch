function(cmakepp_tool)
  set(args ${ARGN})
  list_pop_front(args)
  ans(path)

  pushd("${path}")
    cd("build" --create)
    cmake(
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin 
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=bin 
      .. --process-handle)
    ans(handle)

    cmake(--build . --process-handle)
    ans(handle)

    json_print(${handle})

  popd()
  execute_process(COMMAND "${path}/build/bin/tool")
  return_ans()
endfunction()