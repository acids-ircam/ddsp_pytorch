# Writing Audio Objects with Min

This documentation extends the [Min Guide to Writing Objects](./GuideToWritingObjects.md) by adding information specific to objects for processing audio.

## Audio Processing in Max

When you turn on the audio in Max (e.g. by clicking on an **ezdac~** object) it sets in motion a series of events.  First, a signal chain is compiled by sending a 'dspsetup' message to all objects in the patcher. Any object that responds to that message can then add a vector audio processing routine to the signal chain.

With the signal chain completed, Max will now begin to receive callbacks from the operating system on an *audio thread*. This callback will copy a block (vector) of samples from Max to the audio output device for your system. The number of samples in the vector is determined by the *Input/Output Vector Size* in Max's Audio Settings. Max further divides this vector into smaller blocks determined by the setting simply named *Vector Size*. Max will call your vector audio processing routine on the audio thread once for each of these smaller blocks.

Processing audio in vectors instead of one sample at a time yields tremendous gains in computational performance. The Min API offers the ability to process audio vectors by inheriting from the `vector_operator<>` class.

Additionally, The Min API provides a simpler `sample_operator<>` class from which you may inherit. The `sample_operator<>` allows you to define an operation for a single sample. The boiler-plate code for dealing with vectors is implemented in template classes that are inlined at compile time.

## Class Definition

In addition to inheriting from the `min::object<>` class, your audio objects will now also inherit from `vector_operator<>` or `sample_operator<>`. The later of these requires two template arguments: the number of audio inputs and the number of audio outputs.

In many cases using `sample_operator<>` will be desirable as it simplifies the code. In cases where you need to obtain a shared resource, gain better control of how variables are cached, or work on an object with a dynamic number of inputs/outputs a `vector_operator<>` will provide that additional flexibility.

There following **buffer_loop** example is in the Min-DevKit. The **dcblocker** example is in 

[the Filter Package]: https://github.com/Cycling74/filter/blob/master/source/projects/filter.dcblock_tilde/filter.dcblock_tilde.cpp

. The **ease** example is in 

[the Ease Package]: https://github.com/Cycling74/ease/blob/master/source/projects/ease_tilde/ease_tilde.cpp

.

```c++
class ease : public object<ease>, public sample_operator<1,1> {
public:
	/// ...
};
```

```c++
class buffer_loop : public object<buffer_loop>, public vector_operator<> {
public:
	/// ...
};
```

## Inlets and Outlets

Your audio object will define inlets and outlets as with non-audio objects. However, any audio outlets must have the type specified as "signal".

```c++
class dcblocker : public object<dcblocker>, public sample_operator<1,1> {
public:
	inlet<>			input	{ this, "(signal) Input" };
	outlet<>		output	{ this, "(signal) Output", "signal" };

	/// ...
```

Note that you define your inlets and outlets for both `vector_operator<>` and `sample_operator<>` classes even though`sample_operator<>` classes will have previously indicated the number of inputs and outputs.

### Attribute-Mapped Audio Inlets

Audio inlets may optionally be mapped to attributes of your class. To do this, pass the member attribute as an argument following the description of the inlet. Now, if an audio signal is connected to that inlet then the attribute value will be set by the including audio.

```c++
	inlet<>  m_inlet_attack		{this, "(signal) attack",	m_attack_time};
	inlet<>  m_inlet_release	{this, "(signal) release",	m_release_time};
```

## Messages

There are no required messages for either `vector_operator<>` or `sample_operator<>` classes. You may optionally define a 'dspsetup' message which will be called when Max is compiling the signal chain. The message will be passed two arguments: the sample rate and the vector size.

```c++
message<> dspsetup { this, "dspsetup", 
    MIN_FUNCTION {
		number samplerate = args[0];
		int vectorsize = args[1];

		m_one_over_samplerate = 1.0 / samplerate;
		return {};
	}
};
```
## Buffers

To access a **buffer~** object from your class all you need is to create an instance of a `buffer_reference`, initializing it with a pointer to an instance of your class.

```c++
buffer_reference my_buffer { this };
```

All of the neccessary methods (e.g. `set` and `dblclick`), notification handling, etc. will be provided for you automatically.

If you wish to receive notifications when the **buffer~** content changes you can provide an optional callback to be triggered when a change occurs.

```c++
buffer_reference my_buffer { this, 
	MIN_FUNCTION {
	  // do something in response to the change...
	  return {};
	}
};
```

To access the **buffer~** contents in your audio routine, see the example below for `vector_operator<>` function call implementation.

## Audio Operator Functions

Your object must define a function call operator where the samples of audio will be calculated. The implementation of this will be different depending on whether your audio object is a `sample_operator<>` or a `vector_operator<>`.

### Sample Operators

For `sample_operator<>` classes, the function call operator will take N `sample` arguments as input and return either a `sample` or a container `samples<>` as output.  

The **filter.dcblocker~** example processes a single input and produces a single output.

```c++
sample operator()(sample x) {
	auto y = x - x_1 + y_1 * 0.9997;
	y_1 = y;
	x_1 = x;
	return y;
}
```

The **min.panner~** example has two audio inputs and produces two audio outputs. This is specified in the initial class definition, and the function signature of the call operator must match — meaning that it must take two `sample` arguments and return two samples in a `samples<2>`  container. 

```c++
class panner : public signal_routing_base<panner>, public sample_operator<2,2> {
public:
  
// ...
  
	samples<2> operator()(sample input, sample position = 0.5) {
		auto weight1 = this->weight1;
		auto weight2 = this->weight2;
		
		if (in_pos.has_signal_connection())
			std::tie(weight1, weight2) = calculate_weights(mode, position);
		
		return {{ input * weight1, input * weight2 }};
	}

// ...
```

The `samples<N>` container is a type alias of `std::array<sample,N>`. We construct this container in the return statement. For release builds the compiler optimizes this away as this function will typically be inlined into the vector-processing template that calls it.

### Vector Operators

For `vector_operator<>` classes, the function call operator will take two `audio_bundle` arguments, one each for input and output. 

The number of channels and the size of the vectors are properties of the `audio_bundle`.  Use the `channelcount()` and `framecount()` methods to access the dimensions and the `samples()` method to gain access to the vector data for a specified channel.

The example below is from the **min.buffer.index~** example object. It demonstrates both access to a **buffer~** and implementation of a `vector_operator<>`. Remembering that buffer access is using a shared-resource and must perform atomic operations for threadsafety, the `vector_operator<>` is a much better choice than a `sample_operator<>` because the buffer only needs to be "locked" (and "unlocked") once for the entire vector instead of for each sample.

```c++
void operator()(audio_bundle input, audio_bundle output) {
	auto			in = input.samples(0);	// get vector for channel 0 (first channel)
	auto			out = output.samples(0);// get vector for channel 0 (first channel)
	buffer_lock<>	b(buffer);				// gain access to the buffer~ content
	auto			chan = std::min<int>(channel-1, b.channelcount()); // 1-based channel attr

	if (b.valid()) {
		for (auto i=0; i<input.framecount(); ++i) {
			auto frame = size_t(in[i] + 0.5);
			out[i] = b.lookup(frame, chan);
		}
	}
	else {
		output.clear();
	}
}
```

