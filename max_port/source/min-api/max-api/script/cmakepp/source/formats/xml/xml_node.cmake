# creates a new xml node
# {
#   tag:'tag string'
#   //child_nodes:[xml_node, ...]
#   //parent:xml_node
#   attrs: {  }
#   value: 'string'
#   
# }
function(xml_node tag value attrs)
  obj("${attrs}")
  ans(attrs)
  map()
    kv(tag "${tag}")
    kv(value "${value}")
    kv(attrs "${attrs}")
  end()
  ans(res)
  return_ref(res)
endfunction()