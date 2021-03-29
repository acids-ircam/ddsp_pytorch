## generates the ascii table and stores it in the global ascii_table variable  
  function(ascii_generate_table)
    foreach(i RANGE 1 255)
      string(ASCII ${i} c)
      map_set(ascii_table "'${char}'" "${i}")
      map_set(ascii_table "${i}" "${char}")
    endforeach()
    function(ascii_generate_table)
    endfunction()
  endfunction()

