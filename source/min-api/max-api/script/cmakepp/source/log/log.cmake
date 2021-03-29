## `log(<message:<string>> <refs...> [--error]|[--warning]|[--info]|[--debug]) -> <void>`
##
## This is the base function on which all of the logging depends. It transforms
## every log message into a object which can be consumed by listeners or filtered later
##
## *Note*: in its current state this function is not ready for use
##
## * returns
##   * the reference to the `<log entry>`
## * parameters
##   * `<message>` a `<string>` containing the message which is to be logged the data may be formatted (see `format()`)
##   * `<refs...>` you may pass variable references which will be captured so you can later check the state of the application when the message was logged
## * flags
##   * `--error`    flag indicates that errors occured
##   * `--warning`  flag indicates warnings
##   * `--info`     flag indicates a info output
##   * `--debug`    flag indicates a debug output
## * values
##   * `--error-code <code>` 
##   * `--level <n>` 
##   * `--push <section>` depth+1
##   * `--pop <section>`  depth-1
## * events
##   * `on_log_message`
##
## *Examples*
## ```
## log("this is a simple error" --error) => <% 
##   log("this is a simple error" --error) 
##   template_out_json("${__ans}")
## %>
## ```
function(log)

  map_tryget(global log)
  ans(log)
  if(NOT log)
    map_new()
    ans(log)
    map_set(global log "${log}")
  endif()

  event_handlers(on_log_message)
  ans(has_handlers)
  if(NOT has_handlers)
    return()
  endif()


  set(args ${ARGN})
  list_extract_flag(args --warning)
  list_extract_flag(args --info)
  list_extract_flag(args --debug)
  list_extract_flag(args --aftereffect)
  list_extract_flag(args --trace)
  ans(aftereffect)
  list_extract_flag(args --error)
  ans(is_error)
  list_extract_labelled_value(args --level)
  list_extract_labelled_value(args --push)
  list_extract_labelled_value(args --pop)
  list_extract_labelled_value(args --error-code)
  list_extract_labelled_value(args --function)
  ans(function)

  if(function)
    set(member_function ${function})
  endif()
  if(__current_function_name)
    set(member_function "${__current_function_name}")
  endif()


  ans(error_code)
  map_new()
  ans(entry)
  set(message "${args}")
  format("${message}")
  ans(message)
  if(aftereffect)
    log_last_error_entry()
    ans(last_error)
    map_set(${entry} preceeding_error ${last_error})
  endif()
  map_set(${entry} message ${message})
  ##map_set(${entry} args this ${args})
  map_set(${entry} function ${member_function})
  map_set(${entry} error_code ${error_code})
  set(type)
  if(is_error OR NOT error_code STREQUAL "")
    set(type error)
  endif()
  event_emit(on_log_message ${entry})
  map_set(${entry} type ${type})
  address_append(log_record ${entry})
  return_ref(entry)
endfunction()
