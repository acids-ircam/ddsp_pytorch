parameter_definition(target_add_auto
    <--name{"the name of the target, if empty or '.' the current folder name is used as project name"}:<string>=.>
    [--cppFeatures:<string>=cxx_variadic_templates;cxx_override]
    [--version{"the version of this target"}:<semver>=0.0.1]
    [--include-dir{"directory containing the public headers"}=>includeDir:<path>=include]
    [--source-dir{"directory containing the source files and private headers"}=>sourceDir:<string>=src]
    [--install{"generate install targets"}:<bool>=true]
    [--shared{"generate shared lib target"}:<bool>=true]
    [--static{"generate static lib target"}:<bool>=true]
    [--tests:<bool>=true]
    [--executable:<bool>=false]
    [--linkLibraries:<string>=]
    [--verbose]
    "#generates automatic targets for the current folder. assumes default package layout:
     # * directory `src` containing compilable files
     # * directory `header` containing public header files (which are installed)
     # * asdasd"    
 )

function(target_add_auto)
    arguments_extract_defined_values(0 ${ARGC} target_add_auto)
    

    if("${name}_" STREQUAL "_" OR "${name}_" STREQUAL "._")
      ## get target name from current path
    path_file_name(.)
    ans(targetName)
  else()
    set(targetName "${name}")
    endif()
  



  string(TOUPPER "${targetName}" targetNameUpper)



  glob("include/*.h" --recurse)
  ans(headers)
  glob("src/*.cpp" --recurse)
  ans(sources)





    if(verbose)
      print_vars(name cppFeatures verbose)
      message(INFO "adding auto target '${targetName}'...")
      message(INFO "target name: ${targetName}")
      message(INFO "c++ features: ${cppFeatures}")
      message(FORMAT INFO "version: {version.string}")
      list_length(headers)
      ans(nHeaders)
      list_length(sources)
      ans(nSources)
      message(INFO "include dir: ./${includeDir} (${nHeaders} files)")
      message(INFO "source dir: ./${sourceDir} (${nSources} files)")
    endif()


  set(static_suffix "_static")
  set(shared_suffix "_shared")

  source_group_relative("include" ${headers})
  source_group_relative("src" ${sources})


  write_compiler_detection_header(
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${targetName}/compiler_detection.h"
    PREFIX ${targetNameUpper}
    COMPILERS GNU MSVC
    VERSION 3.5
    FEATURES
      ${cppFeatures}
  )


  if(shared)

    ## shared version
    add_library("${targetName}${shared_suffix}" SHARED ${sources} ${headers})


    target_include_directories("${targetName}${shared_suffix}" PUBLIC   
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
      "$<INSTALL_INTERFACE:${targetName}/include>"
      )
    target_include_directories("${targetName}${shared_suffix}" PRIVATE src )
    target_version_info("${targetName}${shared_suffix}" --version "${version}")
    target_export_header("${targetName}${shared_suffix}" "${targetName}")
    if(MSVC)
      target_compile_options("${targetName}${shared_suffix}" PRIVATE  /MP /W4 /Zi)
      target_link_libraries("${targetName}${shared_suffix}" version.lib Iphlpapi.lib)
    endif()

  endif()

  if(static)

    ## static version
    add_library("${targetName}${static_suffix}" STATIC ${sources} ${headers})

    target_include_directories("${targetName}${static_suffix}" PUBLIC   
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
      "$<INSTALL_INTERFACE:${targetName}/include>"
      )
    target_include_directories("${targetName}${static_suffix}" PRIVATE src )
    target_export_header("${targetName}${static_suffix}" "${targetName}")
    if(MSVC)
      target_compile_options("${targetName}${static_suffix}" PRIVATE  /MP /W4 /Zi)
      target_link_libraries("${targetName}${static_suffix}" version.lib Iphlpapi.lib)
    endif()

  endif()


  ## default library
  add_library(${targetName} ALIAS "${targetName}${shared_suffix}")


  if(install)
    install(DIRECTORY include DESTINATION ${targetName})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${targetName}/compiler_detection.h DESTINATION ${targetName}/include/${targetName})
    install(TARGETS ${targetName}${static_suffix} ${targetName}${shared_suffix}
      EXPORT ${targetName} 
      RUNTIME DESTINATION ${targetName}/bin
      LIBRARY DESTINATION ${targetName}/lib
      ARCHIVE DESTINATION ${targetName}/lib
      )
    install(EXPORT ${targetName} NAMESPACE toeb:: DESTINATION ${targetName})
  endif()

  if(tests)

    ## tests
    glob_ignore("${CMAKE_CURRENT_LIST_DIR}/tests/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/tests/*.h"  "!${CMAKE_CURRENT_LIST_DIR}/bench.cpp")
    ans(test_sources)

    if(test_sources)
      add_executable("${targetName}_test" ${test_sources})
      target_link_libraries("${targetName}_test" "${targetName}${shared_suffix}" gtest)
      target_include_directories("${targetName}_test" PRIVATE ".")
      target_copy_shared_libraries_to_output("${targetName}_test")

      if(MSVC)
        target_compile_options("${targetName}_test" PRIVATE /MP /Zi)
      endif()
    endif()

  endif()
  


endfunction()

