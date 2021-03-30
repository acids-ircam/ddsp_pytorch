/// @file
///	@ingroup 	minapi
///	@copyright	Copyright (c) 2016, Cycling '74
///	@license	Usage of this file and its contents is governed by the MIT License

#pragma once

namespace c74::min {

    using tagged_attribute = std::pair<const symbol, attribute_base*>;

    class ui_operator_base {
    public:
		virtual void add_color_attribute(const tagged_attribute a_color_attr) = 0;
        virtual void update_colors() = 0;
    };


    template<int default_width_type = 20, int default_height_type = 20>
    class ui_operator : public ui_operator_base {
    public:
        explicit ui_operator(object_base* instance, const atoms& args)
        : m_instance { instance }
        {
            if (!m_instance->maxobj()) // box will be a nullptr when being dummy-constructed
                return;

            long flags = 0
                | c74::max::JBOX_DRAWFIRSTIN		// 0
                | c74::max::JBOX_NODRAWBOX		// 1
                | c74::max::JBOX_DRAWINLAST		// 2
             //	| JBOX_TRANSPARENT		// 3
             //	| JBOX_NOGROW			// 4
             //	| JBOX_GROWY			// 5
                | c74::max::JBOX_GROWBOTH			// 6
             //	| JBOX_IGNORELOCKCLICK	// 7
             //	| JBOX_HILITE			// 8
                | c74::max::JBOX_BACKGROUND		// 9
             //	| JBOX_NOFLOATINSPECTOR	// 10
             // | c74::max::JBOX_TEXTFIELD		// 11
             //   | c74::max::JBOX_MOUSEDRAGDELTA	// 12
             //	| JBOX_COLOR			// 13
             //	| JBOX_BINBUF			// 14
             //	| JBOX_DRAWIOLOCKED		// 15
             //	| JBOX_DRAWBACKGROUND	// 16
             //	| JBOX_NOINSPECTFIRSTIN	// 17
             //	| JBOX_DEFAULTNAMES		// 18
             //	| JBOX_FIXWIDTH			// 19
            ;

            strings tags = instance->tags();
            auto tag_iter = std::find(tags.begin(), tags.end(), "multitouch");
            if (tag_iter != tags.end()) {
                flags |= c74::max::JBOX_MULTITOUCH;
            }
            if (m_instance->has_mousedragdelta()) {
                flags |= c74::max::JBOX_MOUSEDRAGDELTA;
            }
            if (m_instance->is_focusable()) {
                flags |= c74::max::JBOX_HILITE;
            }

            const c74::max::t_atom* argv = args.empty() ? nullptr : &args[0];
            c74::max::jbox_new(reinterpret_cast<c74::max::t_jbox*>(m_instance->maxobj()), flags, static_cast<long>(args.size()), argv);
            reinterpret_cast<c74::max::t_jbox*>(m_instance->maxobj())->b_firstin = m_instance->maxobj();
        }

        virtual ~ui_operator() {
            if (m_instance->maxobj())  // box will be a nullptr when being dummy-constructed
                jbox_free(reinterpret_cast<c74::max::t_jbox*>(m_instance->maxobj()));
        }

        void redraw() {
            if (m_instance->initialized())
                jbox_redraw(reinterpret_cast<c74::max::t_jbox*>(m_instance->maxobj()));
        }

        int default_width() const {
            return default_width_type;
        }

        int default_height() const {
            return default_height_type;
        }

        void add_color_attribute(const tagged_attribute a_color_attr) override {
            m_color_attributes.push_back(a_color_attr);
        }


        // update all style-aware attrs
        // must be done at the beginning of the Max object's "paint" method

        void update_colors() override {
            auto& self = *dynamic_cast<object_base*>(this);

            for (const auto& color_attr : m_color_attributes) {
                long              ac { 4};
                c74::max::t_atom avs[4];
                c74::max::t_atom* av = &avs[0];
                attribute_base&	attr { *color_attr.second };
                auto err = c74::max::object_attr_getvalueof(self, color_attr.first, &ac, &av);
                if (!err) {
                    atoms a {av[0], av[1], av[2], av[3]};
                    attr.set(a, false); // notify must be false to prevent feedback loops
                }
            }
        }

    private:
        object_base* m_instance;
		vector<tagged_attribute> m_color_attributes;
    };


    template<class min_class_type>
    typename enable_if< is_base_of<ui_operator_base, min_class_type>::value >::type
    wrap_as_max_external_ui(max::t_class* c, min_class_type& instance) {
        long flags {};

        strings tags = instance.tags();
        auto tag_iter = std::find(tags.begin(), tags.end(), "multitouch");
        if (tag_iter != tags.end()) {
            flags |= c74::max::JBOX_MULTITOUCH;
        }

        // flags |= c74::max::JBOX_TEXTFIELD;
        jbox_initclass(c, flags);
        c->c_flags |= c74::max::CLASS_FLAG_NEWDICTIONARY; // to specify dictionary constructor

        string default_patching_rect {"0. 0. "};
        default_patching_rect += std::to_string(instance.default_width());
        default_patching_rect += " ";
        default_patching_rect += std::to_string(instance.default_height());

        auto attr = (c74::max::t_object*)c74::max::class_attr_get(c, c74::max::gensym("patching_rect"));
        auto attr_type = (c74::max::t_symbol*)object_method(attr, c74::max::gensym("gettype"));
        c74::max::class_attr_addattr_parse(c, "patching_rect", "default", attr_type, 0, default_patching_rect.c_str());
    }

} // namespace c74::min
