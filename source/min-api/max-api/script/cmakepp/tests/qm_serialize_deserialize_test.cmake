function(test)

  qm_serialize("")
  ans(res)

  assert("${res}" STREQUAL "#qm/1.0\nref()\n val(\"\")\nend()\n")


  qm_serialize("asd")
  ans(res)
  assert(${res} STREQUAL  "#qm/1.0\nref()\n val(\"asd\")\nend()\n")


  data("{id:'asd'}")
  ans(res)
  qm_serialize(${res})
  ans(res)

  assert(${res} STREQUAL "#qm/1.0\nref()\nmap()\n key(\"id\")\n  val(\"asd\")\nend()\nend()\n")


  
endfunction()