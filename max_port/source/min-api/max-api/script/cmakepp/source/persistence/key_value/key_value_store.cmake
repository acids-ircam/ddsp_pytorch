
  function(key_value_store key_function)
    set(args ${ARGN})
    list_pop_front(args)
    ans(store_dir)

    path_qualify(store_dir)

    map_new()
    ans(this)

    assign(this.store_dir = store_dir)
    assign(this.save = 'key_value_store_save')
    assign(this.load = 'key_value_store_load')
    assign(this.list = 'key_value_store_list')
    assign(this.keys = 'key_value_store_keys')
    assign(this.delete = 'key_value_store_delete')
    assign(this.key = key_function)
    return(${this})
  endfunction()