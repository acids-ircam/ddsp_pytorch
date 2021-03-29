function(test)

  ## shows that the file frome where a map comes is stored along the data
  
  fwrite_data("test.json" "{ value1:'asd', value2:'bsd'}")
  ans(obj)

  map_set(${obj} hello "world")

  fwrite_data("${obj}")


  fread_data("test.json")
  ans(obj2)


  json_print(${obj2})

  ##fwrite_data("${obj}")


  assertf("{obj2.hello}" STREQUAL "world")

  map_set("${obj2}" next value)
  map_source_file_get("${obj2}")
  ans(paht)
  message("asd ${paht}")

  fwrite_data("${obj2}")


  fread_data(test.json)
  ans(obj3)

  json_print(${obj3})





endfunction()