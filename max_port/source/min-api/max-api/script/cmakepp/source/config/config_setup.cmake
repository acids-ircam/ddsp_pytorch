

function(config_setup name definition)
  map_get(global unused_command_line_args)
  ans(args)
  structured_list_parse("${definition}" ${args})
  ans(config)
  map_tryget(${config} unused)
  ans(args)
  map_set(global unused_command_line_args ${args})
  #curry(config_function("${config}" "${definition}" /1) as "${name}")
  curry3("${name}"(a) => config_function("${config}" "${definition}" /a))
endfunction()