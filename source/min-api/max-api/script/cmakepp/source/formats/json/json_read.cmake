# reads a json file from the specified location
# the location may be relative (see explanantion of path() function)
# returns a map or nothing if reading fails 
function(json_read file)
    path("${file}")
    ans(file)
    if(NOT EXISTS "${file}")
      return()
    endif()
    checksum_file("${file}")
    ans(cache_key)
    file_cache_return_hit("${cache_key}")

    file(READ "${file}" data)
    json_deserialize("${data}")
    ans(data)

    file_cache_update("${cache_key}" "${data}")

    return_ref(data)
endfunction()