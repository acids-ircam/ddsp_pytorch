/// @file
///	@ingroup 	minapi
///	@copyright	Copyright 2018 The Min-API Authors. All rights reserved.
///	@license	Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

namespace c74::min {

    class texteditor {
    public:
        using textfunction = std::function<void(const char*)>;

        texteditor(object_base* an_owner, const textfunction fn, const symbol a_name = "Editor")
        : m_owner { an_owner }
        , m_callback { fn }
        , m_name { a_name }
        {}


        void open(const char* contents) {
            if (!m_jed) {
                m_jed.instantiate("jed", m_owner->maxobj());
                m_jed.set("title", m_name);
                m_jed.set("scratch", 1);
                m_jed("settext", const_cast<char*>(contents), max::gensym("utf-8"));
            }
            else
                m_jed.set("visible", 1);
        }

        void open(const std::string& contents) {
            open(contents.c_str());
        }


    private:
        object_base*    m_owner {};
        instance        m_jed;
        textfunction    m_callback;
        symbol          m_name;

        message<> edclose_meth = {m_owner, "edclose",
            MIN_FUNCTION {
                m_jed = nullptr;
                return {};
            }
        };


        message<> okclose_meth = {m_owner, "okclose",
            MIN_FUNCTION {
                char* text = nullptr;

                m_jed("gettext", &text);
                if (text != nullptr) {
                    m_callback(text);
                    max::sysmem_freeptr(text);
                }
                return {};
            }
        };
    };

}    // namespace c74::min
