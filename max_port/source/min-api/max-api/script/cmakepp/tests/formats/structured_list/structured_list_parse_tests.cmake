function(test)



  map()
    kv(option1 LABELS --option1 -o1 MIN 0 MAX 0 DESCRIPTION "a single option" DISPLAY_NAME "Option Number One")
    kv(single1 LABELS --single1 -s1 MIN 1 MAX 1 DESCRIPTION "a single value" DISPLAY_NAME "Single Number One")
    kv(multi1  LABELS --multi1 -m1 MIN 0 MAX * DESCRIPTION "a multi value" DISPLAY_NAME "Multi Number One")   
  end( )
  ans(structure_map)

  structured_list_parse(${structure_map} -o1 --single1 "hello there" --multi1 muhahaha muha mumu)
  ans(res)

  # simple structure map
  assert(res)
  assert(DEREF EQUALS "{res.single1}"  "hello there")
  assert(DEREF EQUALS "{res.option1}"  true )
  assert(DEREF EQUALS "{res.multi1}"  muhahaha muha mumu)

endfunction()