function(obj_get_test)




# get simple
address_new()
ans(obj)
obj_set(${obj} k1 v1)
obj_get(${obj} k1)
ans(res)
assert("${res}" STREQUAL "v1")


# get non existing
address_new()
ans(obj)
obj_get(${obj} k1)
ans(res)
assert(NOT res)

# get inherited
map_new()
ans(obj)
map_new()
ans(parent)
obj_set(${parent} k1 v1)
map_set_special("${obj}" prototype "${parent}")
obj_get(${obj} k1)
ans(res)
assert("${res}" STREQUAL "v1") 


endfunction()