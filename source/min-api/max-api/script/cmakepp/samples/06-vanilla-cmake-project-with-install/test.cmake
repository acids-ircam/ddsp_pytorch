function(test)

  sample_copy("06")

  pushd("./exporting_project")
  mkdir(build)
  cd(build)
    cmake("-DCMAKE_INSTALL_PREFIX=${test_dir}/prefix_dir" ..)
    cmake(--build . --target install)

  popd()

  assert(EXISTS "${test_dir}/prefix_dir/lib/")


  pushd("./importing_project")
    mkdir(build)
    cd(build)
    cmake(
      -DCMAKE_INSTALL_PREFIX=${test_dir}/prefix_dir
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin 
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=bin
    ..)
    ans(msg)
    message("message ${msg}")
    cmake(--build .)
    execute("bin/myexe")
    ans(res)
    assert("${res}" MATCHES "8")
  popd()


endfunction()