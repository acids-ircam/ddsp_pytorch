function(test)



  sample_copy("05")

  ## compile cmakepp to a single file in project dir
  cmakepp_compile("project_dir/cmakepp.cmake")



  ## write the package descriptor in json format
  ## this package descriptor tells cmakepp to include all files 
  ## in the cmake folder and to load the my_sample_function.cmake
  ## file when the package is used
  ##
  ## `cmakepp.hooks.on_materialilzed` tells cmakepp to execute the specified 
  ##    script after the projects files downloaded.
  ##    when the package is loaded 
  ## `content` tells the package source which files
  ##    are included in the package (the default is **) 
  ## the reason writing it here is an educational one. else you
  ## would create the file in the directory
  fwrite_data("my_package/package.cmake" --json 
  "{
    content:['my_install_script.cmake','src/**','include/**'],
    cmakepp:{
      cmakepp:{
        hooks:{
          on_materialize:'my_install_script.cmake'
        }
      }
    }    
  }")

return()
  ## push the package into a compressed file
  ## if the package decriptor does not specify which files
  ## are to be pushed all files of th package directory are used
  package_source_push_archive("./mypackage/" "my_package.tgz")

  ## check that the package was created
  assert(EXISTS "${test_dir}/my_package.tgz")



  ### install package using the package manager

  cd("project_dir")

  ## when cmakepp is installed you can use this command from the console
  ## during this process the install script is invoked
  pkg(install "../my_package.tgz")


  ## create a build dir 
  ## configure project 
  ## build project 
  ## execute resulting executable
  mkdir("build")
    cd("build")
      cmake(
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=bin 
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG=bin  
      ..)
      cmake(--build .)

      execute("bin/myexe")
      ans(res)


 ## check that output matches the expected value
 assertf("{res.stdout}" MATCHES 42)







endfunction()

