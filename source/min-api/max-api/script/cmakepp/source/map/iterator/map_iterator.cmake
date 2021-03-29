## initializes a new mapiterator
  function(map_iterator map)
    map_keys("${map}")
    ans(keys)
    set(iterator "${map}" before_start ${keys})
    return_ref(iterator)    
  endfunction()

