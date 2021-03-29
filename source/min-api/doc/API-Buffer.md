# Module <!-- group --> `buffers`



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`c74::min::buffer_reference`](#classc74_1_1min_1_1buffer__reference)    | 
`class `[`c74::min::buffer_lock`](#classc74_1_1min_1_1buffer__lock)    | 
# class `c74::min::buffer_reference` {#classc74_1_1min_1_1buffer__reference}




A reference to a buffer~ object. The [buffer_reference](#classc74_1_1min_1_1buffer__reference) automatically adds the management hooks required for your object to work with a buffer~. This includes adding a 'set' message and a 'dblclick' message as well as dealing with notifications and binding.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline  buffer_reference(`[`object_base`](#classc74_1_1min_1_1object__base)` * an_owner,const function & a_function)` | 
`public inline  ~buffer_reference()` | 
`public inline void set(`[`symbol`](#classc74_1_1min_1_1symbol)` name)` | 

## Members

#### `public inline  buffer_reference(`[`object_base`](#classc74_1_1min_1_1object__base)` * an_owner,const function & a_function)` {#classc74_1_1min_1_1buffer__reference_1ad084fd9e298ebad0524f7fad4609d6c0}





#### `public inline  ~buffer_reference()` {#classc74_1_1min_1_1buffer__reference_1ae71e4ab9a938ce7c69a681469b12c8f9}





#### `public inline void set(`[`symbol`](#classc74_1_1min_1_1symbol)` name)` {#classc74_1_1min_1_1buffer__reference_1ac15cef312233271071c39a8368e74e76}





# class `c74::min::buffer_lock` {#classc74_1_1min_1_1buffer__lock}




A lock guard and accessor for buffer~ access from the audio thread.

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public  buffer_lock(`[`buffer_reference`](#classc74_1_1min_1_1buffer__reference)` & a_buffer_ref)` | 
`public  ~buffer_lock()` | 
`public inline bool valid()` | 
`public inline size_t framecount()` | 
`public inline int channelcount()` | 
`public inline float & operator[](long index)` | 
`public inline float & lookup(size_t frame,int channel)` | 
`public inline double samplerate()` | 
`public inline double length_in_seconds()` | 
`public inline void dirty()` | 
`public template<bool U,typename enable_if< U==false, int >::type>`  <br/>`inline void resize(double length_in_ms)` | 
`public template<>`  <br/>` buffer_lock(`[`buffer_reference`](#classc74_1_1min_1_1buffer__reference)` & a_buffer_ref)` | 
`public template<>`  <br/>` buffer_lock(`[`buffer_reference`](#classc74_1_1min_1_1buffer__reference)` & a_buffer_ref)` | 
`public template<>`  <br/>` ~buffer_lock()` | 
`public template<>`  <br/>` ~buffer_lock()` | 

## Members

#### `public  buffer_lock(`[`buffer_reference`](#classc74_1_1min_1_1buffer__reference)` & a_buffer_ref)` {#classc74_1_1min_1_1buffer__lock_1a65074e03f557b281cd0c77d3603af611}





#### `public  ~buffer_lock()` {#classc74_1_1min_1_1buffer__lock_1a38e83266d0506fbef887f1f2d1c4f1df}





#### `public inline bool valid()` {#classc74_1_1min_1_1buffer__lock_1a14f7ccbad215af8793a06178d8e0c36a}





#### `public inline size_t framecount()` {#classc74_1_1min_1_1buffer__lock_1a57850578fb5136a9838a17cc02271d00}





#### `public inline int channelcount()` {#classc74_1_1min_1_1buffer__lock_1aea9adc99a985545f3f3bd661f7b854ba}





#### `public inline float & operator[](long index)` {#classc74_1_1min_1_1buffer__lock_1a7c1cb3a542b5fadae426eaae48a94f35}





#### `public inline float & lookup(size_t frame,int channel)` {#classc74_1_1min_1_1buffer__lock_1a299b143bfbb2ced307a06d3e42673376}





#### `public inline double samplerate()` {#classc74_1_1min_1_1buffer__lock_1acc045c7097529a6683dce5f8f626aecd}





#### `public inline double length_in_seconds()` {#classc74_1_1min_1_1buffer__lock_1ab389ce424002f4510489797727f34649}





#### `public inline void dirty()` {#classc74_1_1min_1_1buffer__lock_1af3180662a533362c17c195e571d8e26c}





#### `public template<bool U,typename enable_if< U==false, int >::type>`  <br/>`inline void resize(double length_in_ms)` {#classc74_1_1min_1_1buffer__lock_1a7b768dd169b5206dae17592f2036dd23}



resize a buffer. only available for non-audio thread access.

#### `public template<>`  <br/>` buffer_lock(`[`buffer_reference`](#classc74_1_1min_1_1buffer__reference)` & a_buffer_ref)` {#classc74_1_1min_1_1buffer__lock_1a790cb7f0bc71dd4146ffe5b39dd32e69}





#### `public template<>`  <br/>` buffer_lock(`[`buffer_reference`](#classc74_1_1min_1_1buffer__reference)` & a_buffer_ref)` {#classc74_1_1min_1_1buffer__lock_1a0ac74369109d48424c4cb7e3e59502ab}





#### `public template<>`  <br/>` ~buffer_lock()` {#classc74_1_1min_1_1buffer__lock_1af2c7155b58925df1f5df7db2f86e8886}





#### `public template<>`  <br/>` ~buffer_lock()` {#classc74_1_1min_1_1buffer__lock_1a6f24289449141c8ecff6392f31b89746}





