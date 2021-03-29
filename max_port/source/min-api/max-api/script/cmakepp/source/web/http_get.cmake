## http_get(<~uri> <?content:<structured data>> [--progress] [--response] [--exit-code] )-> <http response>
##
##
## flags: 
##   --json         flag deserializes the content and returns it 
##   --show-progress     flag prints the progress of the download to the console
##   --response     flag
##   --exit-code  flag
##   --silent-fail  flag
##  
function(http_get uri)
  set(args ${ARGN})
  list_extract_flag(args --json)
  ans(return_json)
  list_extract_flag(args --show-progress)
  ans(show_progress)
  list_extract_flag(args --response)
  ans(return_response)
  list_extract_flag(args --exit-code)
  ans(return_error)
  list_extract_flag(args --silent-fail)
  ans(silent_fail)

  set(show_progress)
  if(show_progress)
    set(show_progress SHOW_PROGRESS)
  endif()
  
  path_temp()
  ans(target_path)

  list_pop_front(args)
  ans(content)

  obj("${content}")
  ans(content)

  uri("${uri}")
  ans(uri)

  uri_format("${uri}" "${content}")
  ans(uri)

  if(return_response)
    set(log LOG http_log)
  endif()

  event_emit(on_http_get "${uri}")
  ans(modified_uri)

  if(modified_uri)
    set(uri "${modified_uri}")
  endif()

  ## actual request - uses file DOWNLOAD which 
  ## uses cUrl internally 
  file(DOWNLOAD 
    "${uri}" 
    "${target_path}" 
    STATUS status 
    ${log}
    ${show_progress}
    TLS_VERIFY OFF 
    ${args}
  )

  # split status into client_status and client_message
  list_extract(status client_status client_message)

  ## return only error code if requested
  if(return_error)
    return_ref(client_status)
  endif()

  ## read content if client was executed correctly
  ## afterwards delete file
  if(NOT client_status)
    fread("${target_path}")
    ans(content)
  else()
    error("http_get failed for '{uri}': ${client_message}")
    if(NOT silent_fail AND NOT return_response)
      rm("${target_path}")
      if("$ENV{TRAVIS}")
        ## do not show the query if travis build because it could contain sensitive
        ## data
        uri_format("${uri}" --no-query)
        ans(uri)
      endif()
      message(FATAL_ERROR "http_get failed for '${uri}': ${client_message}")
    elseif(silent_fail AND NOT return_response)
      rm("${target_path}")
      return()
    endif()
  endif()
  rm("${target_path}")

  ## if the response is not to be returnd
  ## check if deserialization is wished and 
  ## and return content
  if(NOT return_response)
    if(return_json)
      json_deserialize("${content}")
      ans(content)
    endif()
    return_ref(content)
  endif()

  ## parse response and set further fields
  http_last_response_parse("${http_log}")
  ans(response)

  map_set(${response} content "${content}")
  map_set(${response} client_status "${client_status}")
  map_set(${response} client_message "${client_message}")
  map_set(${response} request_url "${uri}")

  string(LENGTH "${content}" content_length)
  map_set(${response} content_length "${content_length}")
  map_set(${response} http_log "${http_log}")

  return_ref(response)
endfunction()