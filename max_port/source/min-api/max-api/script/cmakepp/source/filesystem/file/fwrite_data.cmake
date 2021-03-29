  ## fwrite_data(<path> ([--mimetype <mime type>]|[--json]|[--qm]) <~structured data?>) -> <structured data>
  ##
  ## writes the specified data into the specified target file (overwriting it if it exists)
  ##
  ## fails if no format could be chosen
  ##
  ## format:  if you do not specify a format by passing a mime-type
  ##          or type flag the mime-type is chosen by analysing the 
  ##          file extension - e.g. *.qm files serialize to quickmap
  ##          *.json files serialize to json
  ##
  function(fwrite_data target_file)
    
    set(args ${ARGN})

    ## check first arg is a map with a stored $target_file hidden property
    ## then we can use this as the target file
    ## this field only exists if fread_data was used to read data
    map_source_file_get("${target_file}")
    ans(new_target_file)
    if(new_target_file)
      set(args ${target_file})
      set(target_file "${new_target_file}")      
    endif()   




    ## choose mime type
    list_extract_labelled_value(args --mime-type)
    ans(mime_types)

    list_extract_flag(args --json)
    ans(json)

    list_extract_flag(args --qm)
    ans(quickmap)

    list_extract_flag(args --cmake)
    ans(cmake)

    if(cmake)
      set(mime_types application/x-serializedcmake)
    endif()

    if(json)
      set(mime_types application/json)
    endif()

    if(quickmap)
      set(mime_types application/x-quickmap)
    endif()


    if(NOT mime_types)
      mime_type_from_filename("${target_file}")
      ans(mime_types)
      if(NOT mime_types)
        set(mime_types "application/json")
      endif()
    endif()

    ## parse data
    data(${args})
    ans(data)


    ## serialize data
    if("${mime_types}" MATCHES "application/json")
      json_indented("${data}")
      ans(serialized)
    elseif("${mime_types}" MATCHES "application/x-serializedcmake")
      cmake_serialize("${data}")
      ans(serialized)
    elseif("${mime_types}" MATCHES "application/x-quickmap")
      qm_serialize("${data}")
      ans(serialized)
    else()
      message(FATAL_ERROR "serialization to '${mime_types}' is not supported")
    endif()

    ## write and return data
    fwrite("${target_file}" "${serialized}")

    map_source_file_set("${data}" "${target_file}")    

    return_ref(data)
  endfunction()
