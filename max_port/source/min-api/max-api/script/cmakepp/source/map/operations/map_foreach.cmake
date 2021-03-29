# executes action (key, value)->void
# on every key value pair in map
# exmpl: map = {id:'1',val:'3'}
# map_foreach("${map}" "(k,v)-> message($k $v)")
# prints 
#  id;1
#  val;3
function(map_foreach map action)
	map_keys("${map}")
	ans(keys)
	foreach(key ${keys})
		map_tryget("${map}" "${key}")
		ans(val)
		call("${action}"("${key}" "${val}"))
	endforeach()
endfunction()