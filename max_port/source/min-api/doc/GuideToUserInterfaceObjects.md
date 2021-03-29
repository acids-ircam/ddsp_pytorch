# Writing User Interface (UI) Objects with Min

This documentation extends the [Min Guide to Writing Objects](./GuideToWritingObjects) by adding information specific to user interface objects.

## UI Objects

User Interface objects in Max are, in most respects, like any other object in Max: they respond to messages, possess attributes, and communicate with inlets and outlets. What is different is that they customize the display of their box instead of using the typical text-filled box. UI objects also may respond to gestures made with the mouse or other input devices.

## Class Definition

In addition to inheriting from the `object<>` class, your audio objects will now also inherit from `ui_operator<>`. This requires two template arguments which are the default width and the default height of your UI object.

The *min.textslider* examples in the Min-DevKit provides an excellent starting point.

```c++
class min_textslider : public object<min_textslider>, public ui_operator<140,24> {
public:
	/// ...
};
```

## Inlets and Outlets

Your UI object will define inlets and outlets as with non-ui objects. 

```c++
class min_textslider : public object<min_textslider>, public ui_operator<140,24> {
public:
	inlet<>	input	{ this, "(number) value to set" };
	outlet<> output	{ this, "(number) value" };

	/// ...
```

## Messages

There are a variety of messages  to which `ui_operator<>` classes may wish to respond. The only message that is *required* is the **paint** message which will define how the object will be drawn on the screen.

Supported messages:

* paint (required)
* mousedoubleclick
* mousedragdelta
* mouseup
* mousedown
* mouseleave
* mouseenter
* notify





### mouseenter

If defined, the mousenter message is called when the mouse enters the box for your UI object.

```c++
	message<> mouseenter { this, "mouseenter",
		MIN_FUNCTION {
			m_mouseover = true;
			if (m_showvalue)
				update_text();
			return {};
		}
	};
```





### mousedown

If defined, the mousedown message is called when the mouse is clicked (button pressed down) in the box for your UI object.

```c++
	message<> mousedown { this, "mousedown",
		MIN_FUNCTION {
			ui::target	t { args };
			number		x { args[2] };
			number		y { args[3] };
			int			modifiers { args[4] };

			// cache mouse position so we can restore it after we are done
			m_mouse_position[0] = t.x() + x;
			m_mouse_position[1] = t.y() + y;

			// Jump to new value on mouse down?
			if (m_clickjump) {
				auto delta = MIN_CLAMP((x - 1.0), 0.0, t.width() - 3.0);		// substract for borders
				delta = delta / (t.width() - 2.0) * m_range_delta + m_range[0];
				if (modifiers & c74::max::eCommandKey)
					m_number(static_cast<long>(delta));							//when command-key pressed, jump to the nearest integer-value
				else
					m_number(delta);											// otherwise jump to a float value
			}

			m_anchor = m_value;
//			c74::max::jbox_set_mousedragdelta(maxobj(), 1);
			return {};
		}
	};
```

The message will be passed four arguments: the target, the x-coordinate, the y-coordinate, and a mask that defines what modifier keys are being held down during the click.


