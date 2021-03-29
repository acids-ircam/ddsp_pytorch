# Module <!-- group --> `attributes`



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`c74::min::attribute_base`](#classc74_1_1min_1_1attribute__base)    | 
`class `[`c74::min::attribute_threadsafe_helper`](#classc74_1_1min_1_1attribute__threadsafe__helper)    | 
`class `[`c74::min::attribute`](#classc74_1_1min_1_1attribute)    | 
`class `[`c74::min::attribute_threadsafe_helper< T, threadsafe::yes >`](#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1yes_01_4)    | 
`class `[`c74::min::attribute_threadsafe_helper< T, threadsafe::no >`](#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1no_01_4)    | 
# class `c74::min::attribute_base` {#classc74_1_1min_1_1attribute__base}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  attribute_base(`[`object_base`](#classc74_1_1min_1_1object__base)` & an_owner,std::string a_name)` | 
`public attribute_base & operator=(atoms args)` | set the value of the attribute
`public void set(const atoms & args,bool notify,bool override_readonly)` | set the value of the attribute
`public  operator atoms() const` | get the value of the attribute
`public inline `[`symbol`](#classc74_1_1min_1_1symbol)` datatype() const` | fetch the name of the datatype
`public inline `[`symbol`](#classc74_1_1min_1_1symbol)` name() const` | 
`public inline bool writable() const` | 
`public inline const char * label_string()` | fetch the title/label as a string
`public inline std::string description_string() const` | 
`public inline style editor_style() const` | 
`public inline `[`symbol`](#classc74_1_1min_1_1symbol)` editor_category() const` | 
`public inline long editor_order() const` | 
`public std::string range_string()` | fetch the range in string format, values separated by spaces
`public void create(max::t_class * c,max::method getter,max::method setter,bool isjitclass)` | Create the Max attribute and add it to the Max class.
`public inline size_t size_offset()` | calculate the offset of the size member as required for array/vector attributes
`public inline void touch()` | 
`protected `[`object_base`](#classc74_1_1min_1_1object__base)` & m_owner` | 
`protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_name` | 
`protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_title` | 
`protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_datatype` | 
`protected setter m_setter` | 
`protected getter m_getter` | 
`protected bool m_readonly` | 
`protected size_t m_size` | 
`protected description m_description` | size of array/vector if attr is array/vector
`protected style m_style` | 
`protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_category` | 
`protected long m_order` | 
`protected inline long flags(bool isjitclass)` | 

## Members

#### `public inline  attribute_base(`[`object_base`](#classc74_1_1min_1_1object__base)` & an_owner,std::string a_name)` {#classc74_1_1min_1_1attribute__base_1ab6f99affa124456e781d4aeccd2c84a7}





#### `public attribute_base & operator=(atoms args)` {#classc74_1_1min_1_1attribute__base_1a01304b71dc8a72e2d5f5caf769d8934d}

set the value of the attribute



#### `public void set(const atoms & args,bool notify,bool override_readonly)` {#classc74_1_1min_1_1attribute__base_1afcbbe2c39b31d6acdd731478ca94fd91}

set the value of the attribute



#### `public  operator atoms() const` {#classc74_1_1min_1_1attribute__base_1ac03486f00d3c1cda4d71716464ae86b3}

get the value of the attribute



#### `public inline `[`symbol`](#classc74_1_1min_1_1symbol)` datatype() const` {#classc74_1_1min_1_1attribute__base_1a2b67f447c67be9c173a9eeea222f1b89}

fetch the name of the datatype



#### `public inline `[`symbol`](#classc74_1_1min_1_1symbol)` name() const` {#classc74_1_1min_1_1attribute__base_1ab4d16967fd1ba6b886ef2bdd6a908b64}





#### `public inline bool writable() const` {#classc74_1_1min_1_1attribute__base_1a6aea667268fe90f890a8ad92338f92f4}





#### `public inline const char * label_string()` {#classc74_1_1min_1_1attribute__base_1acef40a1f2052a16bf4ae450cdbc62d92}

fetch the title/label as a string



#### `public inline std::string description_string() const` {#classc74_1_1min_1_1attribute__base_1ab3ebc9e89653fc1cfd128749b21fcfa3}





#### `public inline style editor_style() const` {#classc74_1_1min_1_1attribute__base_1a22e7afc1c80be6b9a7d15a3f4f23a9de}





#### `public inline `[`symbol`](#classc74_1_1min_1_1symbol)` editor_category() const` {#classc74_1_1min_1_1attribute__base_1ae0cd04bf46a000960f58513a27944f50}





#### `public inline long editor_order() const` {#classc74_1_1min_1_1attribute__base_1aca2957ee31fb692bfbc2ae5f80b58e4b}





#### `public std::string range_string()` {#classc74_1_1min_1_1attribute__base_1aee8573362dc8a98dfe855dae8aeb6697}

fetch the range in string format, values separated by spaces



#### `public void create(max::t_class * c,max::method getter,max::method setter,bool isjitclass)` {#classc74_1_1min_1_1attribute__base_1aa88c3914f44c289d800078ebcb0e31ab}

Create the Max attribute and add it to the Max class.



#### `public inline size_t size_offset()` {#classc74_1_1min_1_1attribute__base_1a5735ebe470f1446b28d8e14795b5b53a}

calculate the offset of the size member as required for array/vector attributes



#### `public inline void touch()` {#classc74_1_1min_1_1attribute__base_1a10fd8340b670756c940397c5a1145d4f}





#### `protected `[`object_base`](#classc74_1_1min_1_1object__base)` & m_owner` {#classc74_1_1min_1_1attribute__base_1a1159c1d5f13f255a13e639903fe7862b}





#### `protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_name` {#classc74_1_1min_1_1attribute__base_1a108a0854bfafda67fbb09cdaf815dfea}





#### `protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_title` {#classc74_1_1min_1_1attribute__base_1aa38cfca64fcef81891fa02e9afd94c94}





#### `protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_datatype` {#classc74_1_1min_1_1attribute__base_1ab4b04b90a7b184fd4c240062a82c9a54}





#### `protected setter m_setter` {#classc74_1_1min_1_1attribute__base_1a4999234f1f8b4a7455879cf490a54494}





#### `protected getter m_getter` {#classc74_1_1min_1_1attribute__base_1abd564867fd87dc250a6b22a597d9306f}





#### `protected bool m_readonly` {#classc74_1_1min_1_1attribute__base_1ae6b77adb28b79047036e0b02444bb0b4}





#### `protected size_t m_size` {#classc74_1_1min_1_1attribute__base_1a5e0c0fe94dea732dbc631b7065164dfc}





#### `protected description m_description` {#classc74_1_1min_1_1attribute__base_1a364093315951414ca759523d16c9d7a6}

size of array/vector if attr is array/vector



#### `protected style m_style` {#classc74_1_1min_1_1attribute__base_1af88958ad6c73279a2cc290562836ff6a}





#### `protected `[`symbol`](#classc74_1_1min_1_1symbol)` m_category` {#classc74_1_1min_1_1attribute__base_1a7ac33f29f7ac5908a829576a8cd5f41d}





#### `protected long m_order` {#classc74_1_1min_1_1attribute__base_1a6ea88a96a74ac9b03ed4630d04d3a08c}





#### `protected inline long flags(bool isjitclass)` {#classc74_1_1min_1_1attribute__base_1a973ad81a1f03d892324bd51c639d9f44}





# class `c74::min::attribute_threadsafe_helper` {#classc74_1_1min_1_1attribute__threadsafe__helper}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------

## Members

# class `c74::min::attribute` {#classc74_1_1min_1_1attribute}

```
class c74::min::attribute
  : public c74::min::attribute_base
```  



default is `threadsafe::no`

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public template<typename... ARGS>`  <br/>` attribute(`[`object_base`](#classc74_1_1min_1_1object__base)` * an_owner,std::string a_name,T a_default_value,ARGS... args)` | 
`public inline `[`attribute`](#classc74_1_1min_1_1attribute)` & operator=(const T arg)` | Set the attribute value using the native type of the attribute.
`public inline `[`attribute`](#classc74_1_1min_1_1attribute)` & operator=(atoms args)` | Set the attribute value using atoms.
`public template<class U,typename enable_if< is_enum< U >::value, int >::type>`  <br/>`inline `[`attribute`](#classc74_1_1min_1_1attribute)` & operator=(`[`symbol`](#classc74_1_1min_1_1symbol)` arg)` | 
`public template<class U,typename enable_if< is_enum< U >::value, int >::type>`  <br/>`inline void assign(const atoms & args)` | 
`public inline void set(const atoms & args,bool notify,bool override_readonly)` | Set the attribute value.
`public T range_apply(const T & value)` | 
`public void create(max::t_class * c,max::method getter,max::method setter,bool isjitclass)` | Create the Max attribute and add it to the Max class.
`public inline  operator atoms() const` | 
`public inline  operator const T &() const` | 
`public inline  operator T &()` | 
`public template<class U,typename enable_if< is_same< U, time_value >::value, int >::type>`  <br/>`inline  operator double() const` | 
`public std::string range_string()` | 
`public inline enum_map get_enum_map()` | 
`public inline atoms get_range_args()` | 
`public inline std::vector< T > & range_ref()` | 
`public inline bool disabled() const` | 
`public inline void disable(bool value)` | 
`public template<>`  <br/>` attribute(`[`object_base`](#classc74_1_1min_1_1object__base)` * an_owner,std::string a_name,time_value a_default_value,ARGS... args)` | 
`public template<>`  <br/>`void create(max::t_class * c,max::method getter,max::method setter,bool isjitclass)` | 
`public template<>`  <br/>`std::string range_string()` | 

## Members

#### `public template<typename... ARGS>`  <br/>` attribute(`[`object_base`](#classc74_1_1min_1_1object__base)` * an_owner,std::string a_name,T a_default_value,ARGS... args)` {#classc74_1_1min_1_1attribute_1a21c4f22473f88a94eec1f411dfe09fef}



Constructor 
#### Parameters
* `an_owner` The instance pointer for the owning C++ class, typically you will pass 'this' 


* `a_name` A string specifying the name of the attribute when dynamically addressed or inspected. 


* `a_default_value` The default value of the attribute, which will be set when the instance is created. 


* `...args` N arguments specifying optional properties of an attribute such as setter, label, style, etc.

#### `public inline `[`attribute`](#classc74_1_1min_1_1attribute)` & operator=(const T arg)` {#classc74_1_1min_1_1attribute_1a6dcd1f653263e13260549db74167be93}

Set the attribute value using the native type of the attribute.



#### `public inline `[`attribute`](#classc74_1_1min_1_1attribute)` & operator=(atoms args)` {#classc74_1_1min_1_1attribute_1a28b824bd76a47d760ec6b367365997cb}

Set the attribute value using atoms.



#### `public template<class U,typename enable_if< is_enum< U >::value, int >::type>`  <br/>`inline `[`attribute`](#classc74_1_1min_1_1attribute)` & operator=(`[`symbol`](#classc74_1_1min_1_1symbol)` arg)` {#classc74_1_1min_1_1attribute_1a224a9bd357dd0b3bed82933a5e3a6dfa}





#### `public template<class U,typename enable_if< is_enum< U >::value, int >::type>`  <br/>`inline void assign(const atoms & args)` {#classc74_1_1min_1_1attribute_1a569f0d7eadf44d2eb37053a74489fd3c}





#### `public inline void set(const atoms & args,bool notify,bool override_readonly)` {#classc74_1_1min_1_1attribute_1afa031df8b8691d43a4a47b74164a9f0e}

Set the attribute value.



#### `public T range_apply(const T & value)` {#classc74_1_1min_1_1attribute_1acd319c3c5e92ba6b37c0c3087e4c1ed0}





#### `public void create(max::t_class * c,max::method getter,max::method setter,bool isjitclass)` {#classc74_1_1min_1_1attribute_1a0c2017a88a63fa4b6721ff7cad7489b5}

Create the Max attribute and add it to the Max class.



#### `public inline  operator atoms() const` {#classc74_1_1min_1_1attribute_1a689160d5dcf5d1d07f1f4257cf4f2535}





#### `public inline  operator const T &() const` {#classc74_1_1min_1_1attribute_1a3106e67fc08644fa035574a34c985e91}





#### `public inline  operator T &()` {#classc74_1_1min_1_1attribute_1a5983a3072f02d3ac774dc44d842fa095}





#### `public template<class U,typename enable_if< is_same< U, time_value >::value, int >::type>`  <br/>`inline  operator double() const` {#classc74_1_1min_1_1attribute_1a0b87ab13666dad469fbd1ec82acbb73c}





#### `public std::string range_string()` {#classc74_1_1min_1_1attribute_1a685372013217bfb58d651b66ba0545a4}





#### `public inline enum_map get_enum_map()` {#classc74_1_1min_1_1attribute_1a3bb72ec985820254bdcdaacda1af487e}





#### `public inline atoms get_range_args()` {#classc74_1_1min_1_1attribute_1a258a1fe46cfd81809c15b528475927c6}





#### `public inline std::vector< T > & range_ref()` {#classc74_1_1min_1_1attribute_1a8fbcba9e868af75e7f56d8f5b127864b}





#### `public inline bool disabled() const` {#classc74_1_1min_1_1attribute_1a11e5c7e5a06c7227e4c4ced83a009da5}





#### `public inline void disable(bool value)` {#classc74_1_1min_1_1attribute_1a8afdcb1d9cf935075b6c38f030e59a12}





#### `public template<>`  <br/>` attribute(`[`object_base`](#classc74_1_1min_1_1object__base)` * an_owner,std::string a_name,time_value a_default_value,ARGS... args)` {#classc74_1_1min_1_1attribute_1a3ea9be79b8f1332bbd90a08d8deba36f}





#### `public template<>`  <br/>`void create(max::t_class * c,max::method getter,max::method setter,bool isjitclass)` {#classc74_1_1min_1_1attribute_1a7049c9b6d47655d8f6dd51522aec9a84}





#### `public template<>`  <br/>`std::string range_string()` {#classc74_1_1min_1_1attribute_1a3748cb8b23a7d72c412941439aff1ce9}





# class `c74::min::attribute_threadsafe_helper< T, threadsafe::yes >` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1yes_01_4}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  explicit attribute_threadsafe_helper(`[`attribute`](#classc74_1_1min_1_1attribute)`< T, threadsafe::yes > * an_attribute)` | 
`public inline void set(const atoms & args)` | 

## Members

#### `public inline  explicit attribute_threadsafe_helper(`[`attribute`](#classc74_1_1min_1_1attribute)`< T, threadsafe::yes > * an_attribute)` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1yes_01_4_1a499f1630ef711737ab4a7353aa211a9a}





#### `public inline void set(const atoms & args)` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1yes_01_4_1a77ad4a7541e18ccfe4ffa0fdca305075}





# class `c74::min::attribute_threadsafe_helper< T, threadsafe::no >` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1no_01_4}






## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  explicit attribute_threadsafe_helper(`[`attribute`](#classc74_1_1min_1_1attribute)`< T, threadsafe::no > * an_attribute)` | 
`public inline  ~attribute_threadsafe_helper()` | 
`public inline void set(const atoms & args)` | 

## Members

#### `public inline  explicit attribute_threadsafe_helper(`[`attribute`](#classc74_1_1min_1_1attribute)`< T, threadsafe::no > * an_attribute)` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1no_01_4_1ac77b696598422c3225c49eb4f81be4c2}





#### `public inline  ~attribute_threadsafe_helper()` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1no_01_4_1a3cfd3b98eebec64878008fed4fc871a8}





#### `public inline void set(const atoms & args)` {#classc74_1_1min_1_1attribute__threadsafe__helper_3_01_t_00_01threadsafe_1_1no_01_4_1af73522d29d361ff4eb6a8a95a179a5e4}





