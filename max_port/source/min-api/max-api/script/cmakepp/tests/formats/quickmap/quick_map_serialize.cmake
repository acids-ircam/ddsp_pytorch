function(test)

map()
  kv("heloo" "youuu")
  val("asd")
  map("wooot")
kv("heloo" "youuu")
  val("asd")
  
  end()
end()
ans(res)


qm_serialize(${res})
ans(res)

qm_deserialize(${res})

ans(res)

message("${res}")


map()
kv("val1" "[^\n\t\r   ()()(123]")
end()
ans(in)
qm_serialize("${in}")
ans(res)
message("${res}")
qm_deserialize("${res}")
ans(res)
json_print(${res})
endfunction()