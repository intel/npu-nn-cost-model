// File: VPUNN_0.cpp
#include <exception> // std::exception
#include <sstream> // __str__

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// std::exception file:bits/exception.h line:60
struct PyCallBack_std_exception : public std::exception {
	using std::exception::exception;

	const char * what() const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::exception *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return exception::what();
	}
};

void bind_VPUNN_0(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::exception file:bits/exception.h line:60
		pybind11::class_<std::exception, std::shared_ptr<std::exception>, PyCallBack_std_exception> cl(M("std"), "exception", "");
		cl.def( pybind11::init( [](){ return new std::exception(); }, [](){ return new PyCallBack_std_exception(); } ) );
		cl.def( pybind11::init( [](PyCallBack_std_exception const &o){ return new PyCallBack_std_exception(o); } ) );
		cl.def( pybind11::init( [](std::exception const &o){ return new std::exception(o); } ) );
		cl.def("assign", (class std::exception & (std::exception::*)(const class std::exception &)) &std::exception::operator=, "C++: std::exception::operator=(const class std::exception &) --> class std::exception &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("what", (const char * (std::exception::*)() const) &std::exception::what, "C++: std::exception::what() const --> const char *", pybind11::return_value_policy::automatic);
	}
}


// File: VPUNN_1.cpp
#include <ios> // std::_Ios_Fmtflags
#include <ios> // std::_Ios_Iostate
#include <ios> // std::_Ios_Openmode
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::ios_base
#include <ios> // std::ios_base::Init
#include <ios> // std::ios_base::failure
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <sstream> // __str__
#include <stdexcept> // std::runtime_error
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <system_error> // std::error_code
#include <system_error> // std::error_condition
#include <system_error> // std::system_error

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// std::runtime_error file:stdexcept line:219
struct PyCallBack_std_runtime_error : public std::runtime_error {
	using std::runtime_error::runtime_error;

	const char * what() const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::runtime_error *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return runtime_error::what();
	}
};

// std::system_error file:system_error line:341
struct PyCallBack_std_system_error : public std::system_error {
	using std::system_error::system_error;

	const char * what() const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::system_error *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return runtime_error::what();
	}
};

// std::ios_base::failure file:bits/ios_base.h line:255
struct PyCallBack_std_ios_base_failure : public std::ios_base::failure {
	using std::ios_base::failure::failure;

	const char * what() const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::ios_base::failure *>(this), "what");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<const char *>::value) {
				static pybind11::detail::override_caster_t<const char *> caster;
				return pybind11::detail::cast_ref<const char *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const char *>(std::move(o));
		}
		return failure::what();
	}
};

// std::basic_streambuf file:bits/streambuf.tcc line:149
struct PyCallBack_std_streambuf : public std::streambuf {
	using std::streambuf::basic_streambuf;

	void imbue(const class std::locale & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "imbue");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return basic_streambuf::imbue(a0);
	}
	class std::basic_streambuf<char> * setbuf(char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "setbuf");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class std::basic_streambuf<char> *>::value) {
				static pybind11::detail::override_caster_t<class std::basic_streambuf<char> *> caster;
				return pybind11::detail::cast_ref<class std::basic_streambuf<char> *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::basic_streambuf<char> *>(std::move(o));
		}
		return basic_streambuf::setbuf(a0, a1);
	}
	int sync() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "sync");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::sync();
	}
	long showmanyc() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "showmanyc");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::showmanyc();
	}
	long xsgetn(char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "xsgetn");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::xsgetn(a0, a1);
	}
	int underflow() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "underflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::underflow();
	}
	int uflow() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "uflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::uflow();
	}
	int pbackfail(int a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "pbackfail");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::pbackfail(a0);
	}
	long xsputn(const char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "xsputn");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::xsputn(a0, a1);
	}
	int overflow(int a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::streambuf *>(this), "overflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::overflow(a0);
	}
};

void bind_VPUNN_1(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::locale file:bits/locale_classes.h line:62
		pybind11::class_<std::locale, std::shared_ptr<std::locale>> cl(M("std"), "locale", "");
		cl.def( pybind11::init( [](){ return new std::locale(); } ) );
		cl.def( pybind11::init( [](std::locale const &o){ return new std::locale(o); } ) );
		cl.def( pybind11::init<const char *>(), pybind11::arg("__s") );

		cl.def( pybind11::init<const class std::locale &, const char *, int>(), pybind11::arg("__base"), pybind11::arg("__s"), pybind11::arg("__cat") );

		cl.def( pybind11::init<const std::string &>(), pybind11::arg("__s") );

		cl.def( pybind11::init<const class std::locale &, const std::string &, int>(), pybind11::arg("__base"), pybind11::arg("__s"), pybind11::arg("__cat") );

		cl.def( pybind11::init<const class std::locale &, const class std::locale &, int>(), pybind11::arg("__base"), pybind11::arg("__add"), pybind11::arg("__cat") );

		cl.def("assign", (const class std::locale & (std::locale::*)(const class std::locale &)) &std::locale::operator=, "C++: std::locale::operator=(const class std::locale &) --> const class std::locale &", pybind11::return_value_policy::automatic, pybind11::arg("__other"));
		cl.def("name", (std::string (std::locale::*)() const) &std::locale::name, "C++: std::locale::name() const --> std::string");
		cl.def("__eq__", (bool (std::locale::*)(const class std::locale &) const) &std::locale::operator==, "C++: std::locale::operator==(const class std::locale &) const --> bool", pybind11::arg("__other"));
		cl.def("__ne__", (bool (std::locale::*)(const class std::locale &) const) &std::locale::operator!=, "C++: std::locale::operator!=(const class std::locale &) const --> bool", pybind11::arg("__other"));
		cl.def_static("global", (class std::locale (*)(const class std::locale &)) &std::locale::global, "C++: std::locale::global(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def_static("classic", (const class std::locale & (*)()) &std::locale::classic, "C++: std::locale::classic() --> const class std::locale &", pybind11::return_value_policy::automatic);

		{ // std::locale::id file:bits/locale_classes.h line:483
			auto & enclosing_class = cl;
			pybind11::class_<std::locale::id, std::shared_ptr<std::locale::id>> cl(enclosing_class, "id", "");
			cl.def( pybind11::init( [](){ return new std::locale::id(); } ) );
			cl.def("_M_id", (unsigned long (std::locale::id::*)() const) &std::locale::id::_M_id, "C++: std::locale::id::_M_id() const --> unsigned long");
		}

		{ // std::locale::_Impl file:bits/locale_classes.h line:522
			auto & enclosing_class = cl;
			pybind11::class_<std::locale::_Impl, std::locale::_Impl*> cl(enclosing_class, "_Impl", "");
		}

	}
	{ // std::runtime_error file:stdexcept line:219
		pybind11::class_<std::runtime_error, std::shared_ptr<std::runtime_error>, PyCallBack_std_runtime_error, std::exception> cl(M("std"), "runtime_error", "");
		cl.def( pybind11::init<const std::string &>(), pybind11::arg("__arg") );

		cl.def( pybind11::init<const char *>(), pybind11::arg("") );

		cl.def( pybind11::init( [](PyCallBack_std_runtime_error const &o){ return new PyCallBack_std_runtime_error(o); } ) );
		cl.def( pybind11::init( [](std::runtime_error const &o){ return new std::runtime_error(o); } ) );
		cl.def("assign", (class std::runtime_error & (std::runtime_error::*)(const class std::runtime_error &)) &std::runtime_error::operator=, "C++: std::runtime_error::operator=(const class std::runtime_error &) --> class std::runtime_error &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("what", (const char * (std::runtime_error::*)() const) &std::runtime_error::what, "C++: std::runtime_error::what() const --> const char *", pybind11::return_value_policy::automatic);
	}
	{ // std::error_code file:system_error line:146
		pybind11::class_<std::error_code, std::shared_ptr<std::error_code>> cl(M("std"), "error_code", "");
		cl.def( pybind11::init( [](){ return new std::error_code(); } ) );
		cl.def( pybind11::init( [](std::error_code const &o){ return new std::error_code(o); } ) );
		cl.def("clear", (void (std::error_code::*)()) &std::error_code::clear, "C++: std::error_code::clear() --> void");
		cl.def("value", (int (std::error_code::*)() const) &std::error_code::value, "C++: std::error_code::value() const --> int");
		cl.def("default_error_condition", (struct std::error_condition (std::error_code::*)() const) &std::error_code::default_error_condition, "C++: std::error_code::default_error_condition() const --> struct std::error_condition");
		cl.def("message", (std::string (std::error_code::*)() const) &std::error_code::message, "C++: std::error_code::message() const --> std::string");
		cl.def("assign", (struct std::error_code & (std::error_code::*)(const struct std::error_code &)) &std::error_code::operator=, "C++: std::error_code::operator=(const struct std::error_code &) --> struct std::error_code &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // std::error_condition file:system_error line:224
		pybind11::class_<std::error_condition, std::shared_ptr<std::error_condition>> cl(M("std"), "error_condition", "");
		cl.def( pybind11::init( [](){ return new std::error_condition(); } ) );
		cl.def( pybind11::init( [](std::error_condition const &o){ return new std::error_condition(o); } ) );
		cl.def("clear", (void (std::error_condition::*)()) &std::error_condition::clear, "C++: std::error_condition::clear() --> void");
		cl.def("value", (int (std::error_condition::*)() const) &std::error_condition::value, "C++: std::error_condition::value() const --> int");
		cl.def("message", (std::string (std::error_condition::*)() const) &std::error_condition::message, "C++: std::error_condition::message() const --> std::string");
	}
	{ // std::system_error file:system_error line:341
		pybind11::class_<std::system_error, std::shared_ptr<std::system_error>, PyCallBack_std_system_error, std::runtime_error> cl(M("std"), "system_error", "");
		cl.def( pybind11::init( [](){ return new std::system_error(); }, [](){ return new PyCallBack_std_system_error(); } ), "doc");
		cl.def( pybind11::init<struct std::error_code>(), pybind11::arg("__ec") );

		cl.def( pybind11::init<struct std::error_code, const std::string &>(), pybind11::arg("__ec"), pybind11::arg("__what") );

		cl.def( pybind11::init<struct std::error_code, const char *>(), pybind11::arg("__ec"), pybind11::arg("__what") );

		cl.def( pybind11::init( [](PyCallBack_std_system_error const &o){ return new PyCallBack_std_system_error(o); } ) );
		cl.def( pybind11::init( [](std::system_error const &o){ return new std::system_error(o); } ) );
		cl.def("assign", (class std::system_error & (std::system_error::*)(const class std::system_error &)) &std::system_error::operator=, "C++: std::system_error::operator=(const class std::system_error &) --> class std::system_error &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("code", (const struct std::error_code & (std::system_error::*)() const) &std::system_error::code, "C++: std::system_error::code() const --> const struct std::error_code &", pybind11::return_value_policy::automatic);
	}
	// std::_Ios_Fmtflags file:bits/ios_base.h line:57
	pybind11::enum_<std::_Ios_Fmtflags>(M("std"), "_Ios_Fmtflags", pybind11::arithmetic(), "")
		.value("_S_boolalpha", std::_S_boolalpha)
		.value("_S_dec", std::_S_dec)
		.value("_S_fixed", std::_S_fixed)
		.value("_S_hex", std::_S_hex)
		.value("_S_internal", std::_S_internal)
		.value("_S_left", std::_S_left)
		.value("_S_oct", std::_S_oct)
		.value("_S_right", std::_S_right)
		.value("_S_scientific", std::_S_scientific)
		.value("_S_showbase", std::_S_showbase)
		.value("_S_showpoint", std::_S_showpoint)
		.value("_S_showpos", std::_S_showpos)
		.value("_S_skipws", std::_S_skipws)
		.value("_S_unitbuf", std::_S_unitbuf)
		.value("_S_uppercase", std::_S_uppercase)
		.value("_S_adjustfield", std::_S_adjustfield)
		.value("_S_basefield", std::_S_basefield)
		.value("_S_floatfield", std::_S_floatfield)
		.value("_S_ios_fmtflags_end", std::_S_ios_fmtflags_end)
		.value("_S_ios_fmtflags_max", std::_S_ios_fmtflags_max)
		.value("_S_ios_fmtflags_min", std::_S_ios_fmtflags_min)
		.export_values();

;

	// std::_Ios_Openmode file:bits/ios_base.h line:111
	pybind11::enum_<std::_Ios_Openmode>(M("std"), "_Ios_Openmode", pybind11::arithmetic(), "")
		.value("_S_app", std::_S_app)
		.value("_S_ate", std::_S_ate)
		.value("_S_bin", std::_S_bin)
		.value("_S_in", std::_S_in)
		.value("_S_out", std::_S_out)
		.value("_S_trunc", std::_S_trunc)
		.value("_S_ios_openmode_end", std::_S_ios_openmode_end)
		.value("_S_ios_openmode_max", std::_S_ios_openmode_max)
		.value("_S_ios_openmode_min", std::_S_ios_openmode_min)
		.export_values();

;

	// std::_Ios_Iostate file:bits/ios_base.h line:153
	pybind11::enum_<std::_Ios_Iostate>(M("std"), "_Ios_Iostate", pybind11::arithmetic(), "")
		.value("_S_goodbit", std::_S_goodbit)
		.value("_S_badbit", std::_S_badbit)
		.value("_S_eofbit", std::_S_eofbit)
		.value("_S_failbit", std::_S_failbit)
		.value("_S_ios_iostate_end", std::_S_ios_iostate_end)
		.value("_S_ios_iostate_max", std::_S_ios_iostate_max)
		.value("_S_ios_iostate_min", std::_S_ios_iostate_min)
		.export_values();

;

	// std::_Ios_Seekdir file:bits/ios_base.h line:193
	pybind11::enum_<std::_Ios_Seekdir>(M("std"), "_Ios_Seekdir", pybind11::arithmetic(), "")
		.value("_S_beg", std::_S_beg)
		.value("_S_cur", std::_S_cur)
		.value("_S_end", std::_S_end)
		.value("_S_ios_seekdir_end", std::_S_ios_seekdir_end)
		.export_values();

;

	{ // std::ios_base file:bits/ios_base.h line:228
		pybind11::class_<std::ios_base, std::shared_ptr<std::ios_base>> cl(M("std"), "ios_base", "");

		pybind11::enum_<std::ios_base::event>(cl, "event", pybind11::arithmetic(), "")
			.value("erase_event", std::ios_base::erase_event)
			.value("imbue_event", std::ios_base::imbue_event)
			.value("copyfmt_event", std::ios_base::copyfmt_event)
			.export_values();

		cl.def("flags", (enum std::_Ios_Fmtflags (std::ios_base::*)() const) &std::ios_base::flags, "C++: std::ios_base::flags() const --> enum std::_Ios_Fmtflags");
		cl.def("flags", (enum std::_Ios_Fmtflags (std::ios_base::*)(enum std::_Ios_Fmtflags)) &std::ios_base::flags, "C++: std::ios_base::flags(enum std::_Ios_Fmtflags) --> enum std::_Ios_Fmtflags", pybind11::arg("__fmtfl"));
		cl.def("setf", (enum std::_Ios_Fmtflags (std::ios_base::*)(enum std::_Ios_Fmtflags)) &std::ios_base::setf, "C++: std::ios_base::setf(enum std::_Ios_Fmtflags) --> enum std::_Ios_Fmtflags", pybind11::arg("__fmtfl"));
		cl.def("setf", (enum std::_Ios_Fmtflags (std::ios_base::*)(enum std::_Ios_Fmtflags, enum std::_Ios_Fmtflags)) &std::ios_base::setf, "C++: std::ios_base::setf(enum std::_Ios_Fmtflags, enum std::_Ios_Fmtflags) --> enum std::_Ios_Fmtflags", pybind11::arg("__fmtfl"), pybind11::arg("__mask"));
		cl.def("unsetf", (void (std::ios_base::*)(enum std::_Ios_Fmtflags)) &std::ios_base::unsetf, "C++: std::ios_base::unsetf(enum std::_Ios_Fmtflags) --> void", pybind11::arg("__mask"));
		cl.def("precision", (long (std::ios_base::*)() const) &std::ios_base::precision, "C++: std::ios_base::precision() const --> long");
		cl.def("precision", (long (std::ios_base::*)(long)) &std::ios_base::precision, "C++: std::ios_base::precision(long) --> long", pybind11::arg("__prec"));
		cl.def("width", (long (std::ios_base::*)() const) &std::ios_base::width, "C++: std::ios_base::width() const --> long");
		cl.def("width", (long (std::ios_base::*)(long)) &std::ios_base::width, "C++: std::ios_base::width(long) --> long", pybind11::arg("__wide"));
		cl.def_static("sync_with_stdio", []() -> bool { return std::ios_base::sync_with_stdio(); }, "");
		cl.def_static("sync_with_stdio", (bool (*)(bool)) &std::ios_base::sync_with_stdio, "C++: std::ios_base::sync_with_stdio(bool) --> bool", pybind11::arg("__sync"));
		cl.def("imbue", (class std::locale (std::ios_base::*)(const class std::locale &)) &std::ios_base::imbue, "C++: std::ios_base::imbue(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def("getloc", (class std::locale (std::ios_base::*)() const) &std::ios_base::getloc, "C++: std::ios_base::getloc() const --> class std::locale");
		cl.def("_M_getloc", (const class std::locale & (std::ios_base::*)() const) &std::ios_base::_M_getloc, "C++: std::ios_base::_M_getloc() const --> const class std::locale &", pybind11::return_value_policy::automatic);
		cl.def_static("xalloc", (int (*)()) &std::ios_base::xalloc, "C++: std::ios_base::xalloc() --> int");
		cl.def("iword", (long & (std::ios_base::*)(int)) &std::ios_base::iword, "C++: std::ios_base::iword(int) --> long &", pybind11::return_value_policy::automatic, pybind11::arg("__ix"));
		cl.def("pword", (void *& (std::ios_base::*)(int)) &std::ios_base::pword, "C++: std::ios_base::pword(int) --> void *&", pybind11::return_value_policy::automatic, pybind11::arg("__ix"));

		{ // std::ios_base::failure file:bits/ios_base.h line:255
			auto & enclosing_class = cl;
			pybind11::class_<std::ios_base::failure, std::shared_ptr<std::ios_base::failure>, PyCallBack_std_ios_base_failure, std::system_error> cl(enclosing_class, "failure", "");
			cl.def( pybind11::init<const std::string &>(), pybind11::arg("__str") );

			cl.def( pybind11::init<const std::string &, const struct std::error_code &>(), pybind11::arg(""), pybind11::arg("") );

			cl.def( pybind11::init( [](const char * a0){ return new std::ios_base::failure(a0); }, [](const char * a0){ return new PyCallBack_std_ios_base_failure(a0); } ), "doc");
			cl.def( pybind11::init<const char *, const struct std::error_code &>(), pybind11::arg(""), pybind11::arg("") );

			cl.def( pybind11::init( [](PyCallBack_std_ios_base_failure const &o){ return new PyCallBack_std_ios_base_failure(o); } ) );
			cl.def( pybind11::init( [](std::ios_base::failure const &o){ return new std::ios_base::failure(o); } ) );
			cl.def("what", (const char * (std::ios_base::failure::*)() const) &std::ios_base::failure::what, "C++: std::ios_base::failure::what() const --> const char *", pybind11::return_value_policy::automatic);
			cl.def("assign", (class std::ios_base::failure & (std::ios_base::failure::*)(const class std::ios_base::failure &)) &std::ios_base::failure::operator=, "C++: std::ios_base::failure::operator=(const class std::ios_base::failure &) --> class std::ios_base::failure &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

		{ // std::ios_base::Init file:bits/ios_base.h line:608
			auto & enclosing_class = cl;
			pybind11::class_<std::ios_base::Init, std::shared_ptr<std::ios_base::Init>> cl(enclosing_class, "Init", "");
			cl.def( pybind11::init( [](){ return new std::ios_base::Init(); } ) );
			cl.def( pybind11::init( [](std::ios_base::Init const &o){ return new std::ios_base::Init(o); } ) );
			cl.def("assign", (class std::ios_base::Init & (std::ios_base::Init::*)(const class std::ios_base::Init &)) &std::ios_base::Init::operator=, "C++: std::ios_base::Init::operator=(const class std::ios_base::Init &) --> class std::ios_base::Init &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		}

	}
	{ // std::basic_streambuf file:bits/streambuf.tcc line:149
		pybind11::class_<std::streambuf, std::shared_ptr<std::streambuf>, PyCallBack_std_streambuf> cl(M("std"), "streambuf", "");
		cl.def("pubimbue", (class std::locale (std::streambuf::*)(const class std::locale &)) &std::basic_streambuf<char, std::char_traits<char> >::pubimbue, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubimbue(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def("getloc", (class std::locale (std::streambuf::*)() const) &std::basic_streambuf<char, std::char_traits<char> >::getloc, "C++: std::basic_streambuf<char, std::char_traits<char> >::getloc() const --> class std::locale");
		cl.def("pubsetbuf", (class std::basic_streambuf<char> * (std::streambuf::*)(char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::pubsetbuf, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubsetbuf(char *, long) --> class std::basic_streambuf<char> *", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("pubsync", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::pubsync, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubsync() --> int");
		cl.def("in_avail", (long (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::in_avail, "C++: std::basic_streambuf<char, std::char_traits<char> >::in_avail() --> long");
		cl.def("snextc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::snextc, "C++: std::basic_streambuf<char, std::char_traits<char> >::snextc() --> int");
		cl.def("sbumpc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sbumpc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sbumpc() --> int");
		cl.def("sgetc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sgetc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sgetc() --> int");
		cl.def("sgetn", (long (std::streambuf::*)(char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::sgetn, "C++: std::basic_streambuf<char, std::char_traits<char> >::sgetn(char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("sputbackc", (int (std::streambuf::*)(char)) &std::basic_streambuf<char, std::char_traits<char> >::sputbackc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputbackc(char) --> int", pybind11::arg("__c"));
		cl.def("sungetc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sungetc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sungetc() --> int");
		cl.def("sputc", (int (std::streambuf::*)(char)) &std::basic_streambuf<char, std::char_traits<char> >::sputc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputc(char) --> int", pybind11::arg("__c"));
		cl.def("sputn", (long (std::streambuf::*)(const char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::sputn, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputn(const char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("__safe_gbump", (void (std::streambuf::*)(long)) &std::basic_streambuf<char, std::char_traits<char> >::__safe_gbump, "C++: std::basic_streambuf<char, std::char_traits<char> >::__safe_gbump(long) --> void", pybind11::arg("__n"));
		cl.def("__safe_pbump", (void (std::streambuf::*)(long)) &std::basic_streambuf<char, std::char_traits<char> >::__safe_pbump, "C++: std::basic_streambuf<char, std::char_traits<char> >::__safe_pbump(long) --> void", pybind11::arg("__n"));
	}
}


// File: VPUNN_2.cpp
#include <ios> // std::_Ios_Iostate
#include <ios> // std::_Ios_Seekdir
#include <ios> // std::basic_ios
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <ostream> // std::basic_ostream<char, std::char_traits<char> >::sentry
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_2(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::basic_ios file:bits/basic_ios.tcc line:178
		pybind11::class_<std::basic_ios<char,std::char_traits<char>>, std::shared_ptr<std::basic_ios<char,std::char_traits<char>>>, std::ios_base> cl(M("std"), "basic_ios_char_std_char_traits_char_t", "");
		cl.def( pybind11::init<class std::basic_streambuf<char> *>(), pybind11::arg("__sb") );

		cl.def("rdstate", (enum std::_Ios_Iostate (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::rdstate, "C++: std::basic_ios<char, std::char_traits<char> >::rdstate() const --> enum std::_Ios_Iostate");
		cl.def("clear", [](std::basic_ios<char,std::char_traits<char>> &o) -> void { return o.clear(); }, "");
		cl.def("clear", (void (std::basic_ios<char,std::char_traits<char>>::*)(enum std::_Ios_Iostate)) &std::basic_ios<char, std::char_traits<char> >::clear, "C++: std::basic_ios<char, std::char_traits<char> >::clear(enum std::_Ios_Iostate) --> void", pybind11::arg("__state"));
		cl.def("setstate", (void (std::basic_ios<char,std::char_traits<char>>::*)(enum std::_Ios_Iostate)) &std::basic_ios<char, std::char_traits<char> >::setstate, "C++: std::basic_ios<char, std::char_traits<char> >::setstate(enum std::_Ios_Iostate) --> void", pybind11::arg("__state"));
		cl.def("_M_setstate", (void (std::basic_ios<char,std::char_traits<char>>::*)(enum std::_Ios_Iostate)) &std::basic_ios<char, std::char_traits<char> >::_M_setstate, "C++: std::basic_ios<char, std::char_traits<char> >::_M_setstate(enum std::_Ios_Iostate) --> void", pybind11::arg("__state"));
		cl.def("good", (bool (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::good, "C++: std::basic_ios<char, std::char_traits<char> >::good() const --> bool");
		cl.def("eof", (bool (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::eof, "C++: std::basic_ios<char, std::char_traits<char> >::eof() const --> bool");
		cl.def("fail", (bool (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::fail, "C++: std::basic_ios<char, std::char_traits<char> >::fail() const --> bool");
		cl.def("bad", (bool (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::bad, "C++: std::basic_ios<char, std::char_traits<char> >::bad() const --> bool");
		cl.def("exceptions", (enum std::_Ios_Iostate (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::exceptions, "C++: std::basic_ios<char, std::char_traits<char> >::exceptions() const --> enum std::_Ios_Iostate");
		cl.def("exceptions", (void (std::basic_ios<char,std::char_traits<char>>::*)(enum std::_Ios_Iostate)) &std::basic_ios<char, std::char_traits<char> >::exceptions, "C++: std::basic_ios<char, std::char_traits<char> >::exceptions(enum std::_Ios_Iostate) --> void", pybind11::arg("__except"));
		cl.def("tie", (std::ostream * (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::tie, "C++: std::basic_ios<char, std::char_traits<char> >::tie() const --> std::ostream *", pybind11::return_value_policy::automatic);
		cl.def("tie", (std::ostream * (std::basic_ios<char,std::char_traits<char>>::*)(std::ostream *)) &std::basic_ios<char, std::char_traits<char> >::tie, "C++: std::basic_ios<char, std::char_traits<char> >::tie(std::ostream *) --> std::ostream *", pybind11::return_value_policy::automatic, pybind11::arg("__tiestr"));
		cl.def("rdbuf", (class std::basic_streambuf<char> * (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::rdbuf, "C++: std::basic_ios<char, std::char_traits<char> >::rdbuf() const --> class std::basic_streambuf<char> *", pybind11::return_value_policy::automatic);
		cl.def("rdbuf", (class std::basic_streambuf<char> * (std::basic_ios<char,std::char_traits<char>>::*)(class std::basic_streambuf<char> *)) &std::basic_ios<char, std::char_traits<char> >::rdbuf, "C++: std::basic_ios<char, std::char_traits<char> >::rdbuf(class std::basic_streambuf<char> *) --> class std::basic_streambuf<char> *", pybind11::return_value_policy::automatic, pybind11::arg("__sb"));
		cl.def("copyfmt", (class std::basic_ios<char> & (std::basic_ios<char,std::char_traits<char>>::*)(const class std::basic_ios<char> &)) &std::basic_ios<char, std::char_traits<char> >::copyfmt, "C++: std::basic_ios<char, std::char_traits<char> >::copyfmt(const class std::basic_ios<char> &) --> class std::basic_ios<char> &", pybind11::return_value_policy::automatic, pybind11::arg("__rhs"));
		cl.def("fill", (char (std::basic_ios<char,std::char_traits<char>>::*)() const) &std::basic_ios<char, std::char_traits<char> >::fill, "C++: std::basic_ios<char, std::char_traits<char> >::fill() const --> char");
		cl.def("fill", (char (std::basic_ios<char,std::char_traits<char>>::*)(char)) &std::basic_ios<char, std::char_traits<char> >::fill, "C++: std::basic_ios<char, std::char_traits<char> >::fill(char) --> char", pybind11::arg("__ch"));
		cl.def("imbue", (class std::locale (std::basic_ios<char,std::char_traits<char>>::*)(const class std::locale &)) &std::basic_ios<char, std::char_traits<char> >::imbue, "C++: std::basic_ios<char, std::char_traits<char> >::imbue(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def("narrow", (char (std::basic_ios<char,std::char_traits<char>>::*)(char, char) const) &std::basic_ios<char, std::char_traits<char> >::narrow, "C++: std::basic_ios<char, std::char_traits<char> >::narrow(char, char) const --> char", pybind11::arg("__c"), pybind11::arg("__dfault"));
		cl.def("widen", (char (std::basic_ios<char,std::char_traits<char>>::*)(char) const) &std::basic_ios<char, std::char_traits<char> >::widen, "C++: std::basic_ios<char, std::char_traits<char> >::widen(char) const --> char", pybind11::arg("__c"));
		cl.def("flags", (enum std::_Ios_Fmtflags (std::ios_base::*)() const) &std::ios_base::flags, "C++: std::ios_base::flags() const --> enum std::_Ios_Fmtflags");
		cl.def("flags", (enum std::_Ios_Fmtflags (std::ios_base::*)(enum std::_Ios_Fmtflags)) &std::ios_base::flags, "C++: std::ios_base::flags(enum std::_Ios_Fmtflags) --> enum std::_Ios_Fmtflags", pybind11::arg("__fmtfl"));
		cl.def("setf", (enum std::_Ios_Fmtflags (std::ios_base::*)(enum std::_Ios_Fmtflags)) &std::ios_base::setf, "C++: std::ios_base::setf(enum std::_Ios_Fmtflags) --> enum std::_Ios_Fmtflags", pybind11::arg("__fmtfl"));
		cl.def("setf", (enum std::_Ios_Fmtflags (std::ios_base::*)(enum std::_Ios_Fmtflags, enum std::_Ios_Fmtflags)) &std::ios_base::setf, "C++: std::ios_base::setf(enum std::_Ios_Fmtflags, enum std::_Ios_Fmtflags) --> enum std::_Ios_Fmtflags", pybind11::arg("__fmtfl"), pybind11::arg("__mask"));
		cl.def("unsetf", (void (std::ios_base::*)(enum std::_Ios_Fmtflags)) &std::ios_base::unsetf, "C++: std::ios_base::unsetf(enum std::_Ios_Fmtflags) --> void", pybind11::arg("__mask"));
		cl.def("precision", (long (std::ios_base::*)() const) &std::ios_base::precision, "C++: std::ios_base::precision() const --> long");
		cl.def("precision", (long (std::ios_base::*)(long)) &std::ios_base::precision, "C++: std::ios_base::precision(long) --> long", pybind11::arg("__prec"));
		cl.def("width", (long (std::ios_base::*)() const) &std::ios_base::width, "C++: std::ios_base::width() const --> long");
		cl.def("width", (long (std::ios_base::*)(long)) &std::ios_base::width, "C++: std::ios_base::width(long) --> long", pybind11::arg("__wide"));
		cl.def_static("sync_with_stdio", []() -> bool { return std::ios_base::sync_with_stdio(); }, "");
		cl.def_static("sync_with_stdio", (bool (*)(bool)) &std::ios_base::sync_with_stdio, "C++: std::ios_base::sync_with_stdio(bool) --> bool", pybind11::arg("__sync"));
		cl.def("imbue", (class std::locale (std::ios_base::*)(const class std::locale &)) &std::ios_base::imbue, "C++: std::ios_base::imbue(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def("getloc", (class std::locale (std::ios_base::*)() const) &std::ios_base::getloc, "C++: std::ios_base::getloc() const --> class std::locale");
		cl.def("_M_getloc", (const class std::locale & (std::ios_base::*)() const) &std::ios_base::_M_getloc, "C++: std::ios_base::_M_getloc() const --> const class std::locale &", pybind11::return_value_policy::automatic);
		cl.def_static("xalloc", (int (*)()) &std::ios_base::xalloc, "C++: std::ios_base::xalloc() --> int");
		cl.def("iword", (long & (std::ios_base::*)(int)) &std::ios_base::iword, "C++: std::ios_base::iword(int) --> long &", pybind11::return_value_policy::automatic, pybind11::arg("__ix"));
		cl.def("pword", (void *& (std::ios_base::*)(int)) &std::ios_base::pword, "C++: std::ios_base::pword(int) --> void *&", pybind11::return_value_policy::automatic, pybind11::arg("__ix"));
	}
	{ // std::basic_ostream file:bits/ostream.tcc line:359
		pybind11::class_<std::ostream, std::shared_ptr<std::ostream>, std::basic_ios<char,std::char_traits<char>>> cl(M("std"), "ostream", "");
		cl.def( pybind11::init<class std::basic_streambuf<char> *>(), pybind11::arg("__sb") );

		cl.def("__lshift__", (std::ostream & (std::ostream::*)(long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(bool)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(bool) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(short)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(short) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned short)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned short) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(int)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(int) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned int)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned int) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(long long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(long long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned long long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned long long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(double)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(double) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__f"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(float)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(float) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__f"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(long double)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(long double) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__f"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(const void *)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(const void *) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__p"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(class std::basic_streambuf<char> *)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(class std::basic_streambuf<char> *) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__sb"));
		cl.def("put", (std::ostream & (std::ostream::*)(char)) &std::basic_ostream<char, std::char_traits<char> >::put, "C++: std::basic_ostream<char, std::char_traits<char> >::put(char) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__c"));
		cl.def("_M_write", (void (std::ostream::*)(const char *, long)) &std::basic_ostream<char, std::char_traits<char> >::_M_write, "C++: std::basic_ostream<char, std::char_traits<char> >::_M_write(const char *, long) --> void", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("write", (std::ostream & (std::ostream::*)(const char *, long)) &std::basic_ostream<char, std::char_traits<char> >::write, "C++: std::basic_ostream<char, std::char_traits<char> >::write(const char *, long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("flush", (std::ostream & (std::ostream::*)()) &std::basic_ostream<char, std::char_traits<char> >::flush, "C++: std::basic_ostream<char, std::char_traits<char> >::flush() --> std::ostream &", pybind11::return_value_policy::automatic);
		cl.def("seekp", (std::ostream & (std::ostream::*)(long, enum std::_Ios_Seekdir)) &std::basic_ostream<char, std::char_traits<char> >::seekp, "C++: std::basic_ostream<char, std::char_traits<char> >::seekp(long, enum std::_Ios_Seekdir) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg(""), pybind11::arg(""));

		{ // std::basic_ostream<char, std::char_traits<char> >::sentry file:ostream line:96
			auto & enclosing_class = cl;
			pybind11::class_<std::basic_ostream<char, std::char_traits<char> >::sentry, std::shared_ptr<std::basic_ostream<char, std::char_traits<char> >::sentry>> cl(enclosing_class, "sentry", "");
			cl.def( pybind11::init<std::ostream &>(), pybind11::arg("__os") );

		}

	}
}


// File: VPUNN_3.cpp
#include <ios> // std::_Ios_Openmode
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <sstream> // __str__
#include <sstream> // std::basic_ostringstream
#include <sstream> // std::basic_stringbuf
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// std::basic_stringbuf file:bits/sstream.tcc line:291
struct PyCallBack_std_stringbuf : public std::stringbuf {
	using std::stringbuf::basic_stringbuf;

	long showmanyc() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "showmanyc");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_stringbuf::showmanyc();
	}
	int underflow() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "underflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_stringbuf::underflow();
	}
	int pbackfail(int a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "pbackfail");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_stringbuf::pbackfail(a0);
	}
	int overflow(int a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "overflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_stringbuf::overflow(a0);
	}
	class std::basic_streambuf<char> * setbuf(char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "setbuf");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<class std::basic_streambuf<char> *>::value) {
				static pybind11::detail::override_caster_t<class std::basic_streambuf<char> *> caster;
				return pybind11::detail::cast_ref<class std::basic_streambuf<char> *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::basic_streambuf<char> *>(std::move(o));
		}
		return basic_stringbuf::setbuf(a0, a1);
	}
	void imbue(const class std::locale & a0) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "imbue");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return basic_streambuf::imbue(a0);
	}
	int sync() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "sync");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::sync();
	}
	long xsgetn(char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "xsgetn");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::xsgetn(a0, a1);
	}
	int uflow() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "uflow");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return basic_streambuf::uflow();
	}
	long xsputn(const char * a0, long a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const std::stringbuf *>(this), "xsputn");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long>::value) {
				static pybind11::detail::override_caster_t<long> caster;
				return pybind11::detail::cast_ref<long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long>(std::move(o));
		}
		return basic_streambuf::xsputn(a0, a1);
	}
};

void bind_VPUNN_3(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::basic_stringbuf file:bits/sstream.tcc line:291
		pybind11::class_<std::stringbuf, std::shared_ptr<std::stringbuf>, PyCallBack_std_stringbuf, std::streambuf> cl(M("std"), "stringbuf", "");
		cl.def( pybind11::init( [](){ return new std::stringbuf(); }, [](){ return new PyCallBack_std_stringbuf(); } ) );
		cl.def( pybind11::init<enum std::_Ios_Openmode>(), pybind11::arg("__mode") );

		cl.def( pybind11::init( [](const std::string & a0){ return new std::stringbuf(a0); }, [](const std::string & a0){ return new PyCallBack_std_stringbuf(a0); } ), "doc");
		cl.def( pybind11::init<const std::string &, enum std::_Ios_Openmode>(), pybind11::arg("__str"), pybind11::arg("__mode") );

		cl.def("swap", (void (std::stringbuf::*)(class std::basic_stringbuf<char> &)) &std::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::swap, "C++: std::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::swap(class std::basic_stringbuf<char> &) --> void", pybind11::arg("__rhs"));
		cl.def("str", (std::string (std::stringbuf::*)() const) &std::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::str, "C++: std::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::str() const --> std::string");
		cl.def("str", (void (std::stringbuf::*)(const std::string &)) &std::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::str, "C++: std::basic_stringbuf<char, std::char_traits<char>, std::allocator<char> >::str(const std::string &) --> void", pybind11::arg("__s"));
		cl.def("pubimbue", (class std::locale (std::streambuf::*)(const class std::locale &)) &std::basic_streambuf<char, std::char_traits<char> >::pubimbue, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubimbue(const class std::locale &) --> class std::locale", pybind11::arg("__loc"));
		cl.def("getloc", (class std::locale (std::streambuf::*)() const) &std::basic_streambuf<char, std::char_traits<char> >::getloc, "C++: std::basic_streambuf<char, std::char_traits<char> >::getloc() const --> class std::locale");
		cl.def("pubsetbuf", (class std::basic_streambuf<char> * (std::streambuf::*)(char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::pubsetbuf, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubsetbuf(char *, long) --> class std::basic_streambuf<char> *", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("pubsync", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::pubsync, "C++: std::basic_streambuf<char, std::char_traits<char> >::pubsync() --> int");
		cl.def("in_avail", (long (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::in_avail, "C++: std::basic_streambuf<char, std::char_traits<char> >::in_avail() --> long");
		cl.def("snextc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::snextc, "C++: std::basic_streambuf<char, std::char_traits<char> >::snextc() --> int");
		cl.def("sbumpc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sbumpc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sbumpc() --> int");
		cl.def("sgetc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sgetc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sgetc() --> int");
		cl.def("sgetn", (long (std::streambuf::*)(char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::sgetn, "C++: std::basic_streambuf<char, std::char_traits<char> >::sgetn(char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("sputbackc", (int (std::streambuf::*)(char)) &std::basic_streambuf<char, std::char_traits<char> >::sputbackc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputbackc(char) --> int", pybind11::arg("__c"));
		cl.def("sungetc", (int (std::streambuf::*)()) &std::basic_streambuf<char, std::char_traits<char> >::sungetc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sungetc() --> int");
		cl.def("sputc", (int (std::streambuf::*)(char)) &std::basic_streambuf<char, std::char_traits<char> >::sputc, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputc(char) --> int", pybind11::arg("__c"));
		cl.def("sputn", (long (std::streambuf::*)(const char *, long)) &std::basic_streambuf<char, std::char_traits<char> >::sputn, "C++: std::basic_streambuf<char, std::char_traits<char> >::sputn(const char *, long) --> long", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("__safe_gbump", (void (std::streambuf::*)(long)) &std::basic_streambuf<char, std::char_traits<char> >::__safe_gbump, "C++: std::basic_streambuf<char, std::char_traits<char> >::__safe_gbump(long) --> void", pybind11::arg("__n"));
		cl.def("__safe_pbump", (void (std::streambuf::*)(long)) &std::basic_streambuf<char, std::char_traits<char> >::__safe_pbump, "C++: std::basic_streambuf<char, std::char_traits<char> >::__safe_pbump(long) --> void", pybind11::arg("__n"));
	}
	{ // std::basic_ostringstream file:bits/sstream.tcc line:293
		pybind11::class_<std::ostringstream, std::shared_ptr<std::ostringstream>, std::ostream> cl(M("std"), "ostringstream", "");
		cl.def( pybind11::init( [](){ return new std::ostringstream(); } ) );
		cl.def( pybind11::init<enum std::_Ios_Openmode>(), pybind11::arg("__mode") );

		cl.def( pybind11::init( [](const std::string & a0){ return new std::ostringstream(a0); } ), "doc" , pybind11::arg("__str"));
		cl.def( pybind11::init<const std::string &, enum std::_Ios_Openmode>(), pybind11::arg("__str"), pybind11::arg("__mode") );

		cl.def("swap", (void (std::ostringstream::*)(class std::basic_ostringstream<char> &)) &std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::swap, "C++: std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::swap(class std::basic_ostringstream<char> &) --> void", pybind11::arg("__rhs"));
		cl.def("rdbuf", (class std::basic_stringbuf<char> * (std::ostringstream::*)() const) &std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::rdbuf, "C++: std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::rdbuf() const --> class std::basic_stringbuf<char> *", pybind11::return_value_policy::automatic);
		cl.def("str", (std::string (std::ostringstream::*)() const) &std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::str, "C++: std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::str() const --> std::string");
		cl.def("str", (void (std::ostringstream::*)(const std::string &)) &std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::str, "C++: std::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::str(const std::string &) --> void", pybind11::arg("__s"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(bool)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(bool) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(short)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(short) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned short)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned short) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(int)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(int) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned int)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned int) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(long long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(long long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(unsigned long long)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(unsigned long long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__n"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(double)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(double) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__f"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(float)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(float) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__f"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(long double)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(long double) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__f"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(const void *)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(const void *) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__p"));
		cl.def("__lshift__", (std::ostream & (std::ostream::*)(class std::basic_streambuf<char> *)) &std::basic_ostream<char, std::char_traits<char> >::operator<<, "C++: std::basic_ostream<char, std::char_traits<char> >::operator<<(class std::basic_streambuf<char> *) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__sb"));
		cl.def("put", (std::ostream & (std::ostream::*)(char)) &std::basic_ostream<char, std::char_traits<char> >::put, "C++: std::basic_ostream<char, std::char_traits<char> >::put(char) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__c"));
		cl.def("_M_write", (void (std::ostream::*)(const char *, long)) &std::basic_ostream<char, std::char_traits<char> >::_M_write, "C++: std::basic_ostream<char, std::char_traits<char> >::_M_write(const char *, long) --> void", pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("write", (std::ostream & (std::ostream::*)(const char *, long)) &std::basic_ostream<char, std::char_traits<char> >::write, "C++: std::basic_ostream<char, std::char_traits<char> >::write(const char *, long) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg("__s"), pybind11::arg("__n"));
		cl.def("flush", (std::ostream & (std::ostream::*)()) &std::basic_ostream<char, std::char_traits<char> >::flush, "C++: std::basic_ostream<char, std::char_traits<char> >::flush() --> std::ostream &", pybind11::return_value_policy::automatic);
		cl.def("seekp", (std::ostream & (std::ostream::*)(long, enum std::_Ios_Seekdir)) &std::basic_ostream<char, std::char_traits<char> >::seekp, "C++: std::basic_ostream<char, std::char_traits<char> >::seekp(long, enum std::_Ios_Seekdir) --> std::ostream &", pybind11::return_value_policy::automatic, pybind11::arg(""), pybind11::arg(""));
	}
}


// File: VPUNN_4.cpp
#include <sstream> // __str__
#include <vpu/cycles_interface_types.h> // VPUNN::Cycles

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_4(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::Cycles file:vpu/cycles_interface_types.h line:48
		pybind11::class_<VPUNN::Cycles, std::shared_ptr<VPUNN::Cycles>> cl(M("VPUNN"), "Cycles", "helper class for CyclesInterfaceType");
		cl.def( pybind11::init( [](){ return new VPUNN::Cycles(); } ) );
		cl.def_static("toCycleInterfaceType", (unsigned int (*)(float)) &VPUNN::Cycles::toCycleInterfaceType<float,void>, "C++: VPUNN::Cycles::toCycleInterfaceType(float) --> unsigned int", pybind11::arg("conversion_number"));
		cl.def_static("toCycleInterfaceType", (unsigned int (*)(double)) &VPUNN::Cycles::toCycleInterfaceType<double,void>, "C++: VPUNN::Cycles::toCycleInterfaceType(double) --> unsigned int", pybind11::arg("conversion_number"));
		cl.def_static("isErrorCode", (bool (*)(const unsigned int &)) &VPUNN::Cycles::isErrorCode, "true if v has a value that can be an error code\n\n \n the value to be interpreted.\n \n\n true if the value is large enough to be mapped to an error code (error code might exist or not)\n\nC++: VPUNN::Cycles::isErrorCode(const unsigned int &) --> bool", pybind11::arg("v"));
		cl.def_static("toErrorText", (const char * (*)(const unsigned int &)) &VPUNN::Cycles::toErrorText, "provides a text if the value is an error or zero\n\n \n the value to be interpreted. normally 0-reasonable values means cycles, and values close to max limit\n are error codes\n \n\n a plain text with error name or \"UNKNOWN\"\n\nC++: VPUNN::Cycles::toErrorText(const unsigned int &) --> const char *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("cost_adder", (unsigned int (*)(const unsigned int, const unsigned int)) &VPUNN::Cycles::cost_adder, "safe sum of cycles considering also the error handling situations and overflow\n If the sum of the valid numbers gets in the error area , it will result Cycles::EROOR_SUM_TOO_LARGE\n If one of the terms is already an error, the error is kept as result (first term has priority of both are errors)\n\n \n left term\n \n\n right term\n\n \n the sum of lhs with rhs or the specific error in case that we have a sum\n too large or with error in terms\n\nC++: VPUNN::Cycles::cost_adder(const unsigned int, const unsigned int) --> unsigned int", pybind11::arg("lhs"), pybind11::arg("rhs"));
		cl.def_static("toCycleInterfaceType", (unsigned int (*)(unsigned int)) &VPUNN::Cycles::toCycleInterfaceType, "This method is an overload because if we got a CycleInterfaceType in this conversion method we want to\n propagate the error that already was assigned and not to change it to ERROR_INVALID_CONVERSION_TO_CYCLES\n\n \n number we are trying to convert\n \n\n the same number because it is already in CyclesInterfaceType\n\nC++: VPUNN::Cycles::toCycleInterfaceType(unsigned int) --> unsigned int", pybind11::arg("conversion_number"));
	}
}


// File: VPUNN_5.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_5(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::dpu_schedule(const unsigned int, const class std::vector<unsigned int, class std::allocator<unsigned int> > &, const unsigned int) file: line:53
	M("VPUNN").def("dpu_schedule", [](const unsigned int & a0, const class std::vector<unsigned int, class std::allocator<unsigned int> > & a1) -> unsigned int { return VPUNN::dpu_schedule(a0, a1); }, "", pybind11::arg("n_procesors"), pybind11::arg("tasks_cost"));
	M("VPUNN").def("dpu_schedule", (unsigned int (*)(const unsigned int, const class std::vector<unsigned int, class std::allocator<unsigned int> > &, const unsigned int)) &VPUNN::dpu_schedule<unsigned int>, "C++: VPUNN::dpu_schedule(const unsigned int, const class std::vector<unsigned int, class std::allocator<unsigned int> > &, const unsigned int) --> unsigned int", pybind11::arg("n_procesors"), pybind11::arg("tasks_cost"), pybind11::arg("runtime_overhead"));

	// VPUNN::multiply_vector(const struct std::array<unsigned int, 4> &) file: line:85
	M("VPUNN").def("multiply_vector", (unsigned int (*)(const struct std::array<unsigned int, 4> &)) &VPUNN::multiply_vector<unsigned int,4>, "C++: VPUNN::multiply_vector(const struct std::array<unsigned int, 4> &) --> unsigned int", pybind11::arg("vec"));

	// VPUNN::multiply_vector(const struct std::array<unsigned int, 2> &) file: line:85
	M("VPUNN").def("multiply_vector", (unsigned int (*)(const struct std::array<unsigned int, 2> &)) &VPUNN::multiply_vector<unsigned int,2>, "C++: VPUNN::multiply_vector(const struct std::array<unsigned int, 2> &) --> unsigned int", pybind11::arg("vec"));

	// VPUNN::ceil_division(unsigned int, unsigned int) file: line:110
	M("VPUNN").def("ceil_division", (unsigned int (*)(unsigned int, unsigned int)) &VPUNN::ceil_division<unsigned int>, "C++: VPUNN::ceil_division(unsigned int, unsigned int) --> unsigned int", pybind11::arg("a"), pybind11::arg("b"));

	// VPUNN::ceil_division(unsigned long, unsigned long) file: line:110
	M("VPUNN").def("ceil_division", (unsigned long (*)(unsigned long, unsigned long)) &VPUNN::ceil_division<unsigned long>, "C++: VPUNN::ceil_division(unsigned long, unsigned long) --> unsigned long", pybind11::arg("a"), pybind11::arg("b"));

	// VPUNN::round_up(unsigned int, unsigned int) file: line:123
	M("VPUNN").def("round_up", (unsigned int (*)(unsigned int, unsigned int)) &VPUNN::round_up<unsigned int>, "C++: VPUNN::round_up(unsigned int, unsigned int) --> unsigned int", pybind11::arg("a"), pybind11::arg("b"));

	// VPUNN::divide_and_multiply_vectors(const class std::vector<unsigned int, class std::allocator<unsigned int> > &, const class std::vector<unsigned int, class std::allocator<unsigned int> > &) file: line:134
	M("VPUNN").def("divide_and_multiply_vectors", (unsigned int (*)(const class std::vector<unsigned int, class std::allocator<unsigned int> > &, const class std::vector<unsigned int, class std::allocator<unsigned int> > &)) &VPUNN::divide_and_multiply_vectors, "Perform an elementwise ceil division and then multiply the results together\n\n \n\n \n\n\n \n\n unsigned int\n\nC++: VPUNN::divide_and_multiply_vectors(const class std::vector<unsigned int, class std::allocator<unsigned int> > &, const class std::vector<unsigned int, class std::allocator<unsigned int> > &) --> unsigned int", pybind11::arg("v1"), pybind11::arg("v2"));

	// VPUNN::helper_input_dim(unsigned int, unsigned int, unsigned int, unsigned int) file: line:153
	M("VPUNN").def("helper_input_dim", (unsigned int (*)(unsigned int, unsigned int, unsigned int, unsigned int)) &VPUNN::helper_input_dim, "Compute an input tensor dimension from the output dimension and the operation parameters\n\n \n the output dimension\n \n\n the kernel size\n \n\n the total padding\n \n\n the kernel stride\n \n\n unsigned int the input dimension\n\nC++: VPUNN::helper_input_dim(unsigned int, unsigned int, unsigned int, unsigned int) --> unsigned int", pybind11::arg("output"), pybind11::arg("kernel"), pybind11::arg("total_padding"), pybind11::arg("stride"));

}


// File: VPUNN_6.cpp
#include <functional> // std::less
#include <iterator> // __gnu_cxx::__normal_iterator
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::ActivationFunction
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::DataType
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::ExecutionMode
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::Layout
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::MemoryLocation
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::Operation
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::Swizzling
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::VPUDevice
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::VPUSubsystem
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ActivationFunction
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::DataType
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ExecutionMode
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ISIStrategy
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Layout
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::MemoryLocation
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Operation
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Swizzling
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::VPUDevice
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::VPUSubsystem

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_6(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::link(const enum VPUNN::VPUDevice &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::VPUDevice &, const char *)) &VPUNN::link<VPUNN::VPUDevice>, "C++: VPUNN::link(const enum VPUNN::VPUDevice &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::DataType &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::DataType &, const char *)) &VPUNN::link<VPUNN::DataType>, "C++: VPUNN::link(const enum VPUNN::DataType &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::Operation &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::Operation &, const char *)) &VPUNN::link<VPUNN::Operation>, "C++: VPUNN::link(const enum VPUNN::Operation &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::ActivationFunction &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::ActivationFunction &, const char *)) &VPUNN::link<VPUNN::ActivationFunction>, "C++: VPUNN::link(const enum VPUNN::ActivationFunction &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::Swizzling &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::Swizzling &, const char *)) &VPUNN::link<VPUNN::Swizzling>, "C++: VPUNN::link(const enum VPUNN::Swizzling &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::ExecutionMode &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::ExecutionMode &, const char *)) &VPUNN::link<VPUNN::ExecutionMode>, "C++: VPUNN::link(const enum VPUNN::ExecutionMode &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::Layout &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::Layout &, const char *)) &VPUNN::link<VPUNN::Layout>, "C++: VPUNN::link(const enum VPUNN::Layout &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::ISIStrategy &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::ISIStrategy &, const char *)) &VPUNN::link<VPUNN::ISIStrategy>, "C++: VPUNN::link(const enum VPUNN::ISIStrategy &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::MemoryLocation &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::MemoryLocation &, const char *)) &VPUNN::link<VPUNN::MemoryLocation>, "C++: VPUNN::link(const enum VPUNN::MemoryLocation &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::VPUSubsystem &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::VPUSubsystem &, const char *)) &VPUNN::link<VPUNN::VPUSubsystem>, "C++: VPUNN::link(const enum VPUNN::VPUSubsystem &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::VPUTilingStrategy &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::VPUTilingStrategy &, const char *)) &VPUNN::link<VPUNN::VPUTilingStrategy>, "C++: VPUNN::link(const enum VPUNN::VPUTilingStrategy &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::VPUDevice &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::VPUDevice &, const char *)) &VPUNN::link<VPUNN::intf_01::VPUDevice>, "C++: VPUNN::link(const enum VPUNN::intf_01::VPUDevice &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::DataType &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::DataType &, const char *)) &VPUNN::link<VPUNN::intf_01::DataType>, "C++: VPUNN::link(const enum VPUNN::intf_01::DataType &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::Operation &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::Operation &, const char *)) &VPUNN::link<VPUNN::intf_01::Operation>, "C++: VPUNN::link(const enum VPUNN::intf_01::Operation &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::ActivationFunction &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::ActivationFunction &, const char *)) &VPUNN::link<VPUNN::intf_01::ActivationFunction>, "C++: VPUNN::link(const enum VPUNN::intf_01::ActivationFunction &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::Swizzling &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::Swizzling &, const char *)) &VPUNN::link<VPUNN::intf_01::Swizzling>, "C++: VPUNN::link(const enum VPUNN::intf_01::Swizzling &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::ExecutionMode &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::ExecutionMode &, const char *)) &VPUNN::link<VPUNN::intf_01::ExecutionMode>, "C++: VPUNN::link(const enum VPUNN::intf_01::ExecutionMode &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::Layout &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::Layout &, const char *)) &VPUNN::link<VPUNN::intf_01::Layout>, "C++: VPUNN::link(const enum VPUNN::intf_01::Layout &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::MemoryLocation &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::MemoryLocation &, const char *)) &VPUNN::link<VPUNN::intf_01::MemoryLocation>, "C++: VPUNN::link(const enum VPUNN::intf_01::MemoryLocation &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_01::VPUSubsystem &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_01::VPUSubsystem &, const char *)) &VPUNN::link<VPUNN::intf_01::VPUSubsystem>, "C++: VPUNN::link(const enum VPUNN::intf_01::VPUSubsystem &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::VPUDevice &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::VPUDevice &, const char *)) &VPUNN::link<VPUNN::intf_11::VPUDevice>, "C++: VPUNN::link(const enum VPUNN::intf_11::VPUDevice &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::DataType &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::DataType &, const char *)) &VPUNN::link<VPUNN::intf_11::DataType>, "C++: VPUNN::link(const enum VPUNN::intf_11::DataType &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::Operation &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::Operation &, const char *)) &VPUNN::link<VPUNN::intf_11::Operation>, "C++: VPUNN::link(const enum VPUNN::intf_11::Operation &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::ActivationFunction &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::ActivationFunction &, const char *)) &VPUNN::link<VPUNN::intf_11::ActivationFunction>, "C++: VPUNN::link(const enum VPUNN::intf_11::ActivationFunction &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::Swizzling &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::Swizzling &, const char *)) &VPUNN::link<VPUNN::intf_11::Swizzling>, "C++: VPUNN::link(const enum VPUNN::intf_11::Swizzling &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::ExecutionMode &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::ExecutionMode &, const char *)) &VPUNN::link<VPUNN::intf_11::ExecutionMode>, "C++: VPUNN::link(const enum VPUNN::intf_11::ExecutionMode &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::Layout &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::Layout &, const char *)) &VPUNN::link<VPUNN::intf_11::Layout>, "C++: VPUNN::link(const enum VPUNN::intf_11::Layout &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::ISIStrategy &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::ISIStrategy &, const char *)) &VPUNN::link<VPUNN::intf_11::ISIStrategy>, "C++: VPUNN::link(const enum VPUNN::intf_11::ISIStrategy &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::MemoryLocation &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::MemoryLocation &, const char *)) &VPUNN::link<VPUNN::intf_11::MemoryLocation>, "C++: VPUNN::link(const enum VPUNN::intf_11::MemoryLocation &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::link(const enum VPUNN::intf_11::VPUSubsystem &, const char *) file: line:35
	M("VPUNN").def("link", (struct std::pair<const int, const std::string > (*)(const enum VPUNN::intf_11::VPUSubsystem &, const char *)) &VPUNN::link<VPUNN::intf_11::VPUSubsystem>, "C++: VPUNN::link(const enum VPUNN::intf_11::VPUSubsystem &, const char *) --> struct std::pair<const int, const std::string >", pybind11::arg("enum_val"), pybind11::arg("name"));

	// VPUNN::createInverseMap(const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &) file: line:43
	M("VPUNN").def("createInverseMap", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > (*)(const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &)) &VPUNN::createInverseMap, "creates and inverse map given a direct map (EnumMap)\n\nC++: VPUNN::createInverseMap(const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &) --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > >", pybind11::arg("direct_map"));

	// VPUNN::VPUDevice file: line:89
	pybind11::enum_<VPUNN::VPUDevice>(M("VPUNN"), "VPUDevice", "VPU IP generations\n\n ")
		.value("VPU_2_0", VPUNN::VPUDevice::VPU_2_0)
		.value("VPU_2_1", VPUNN::VPUDevice::VPU_2_1)
		.value("VPU_2_7", VPUNN::VPUDevice::VPU_2_7)
		.value("VPU_4_0", VPUNN::VPUDevice::VPU_4_0)
		.value("__size", VPUNN::VPUDevice::__size);

;

	// VPUNN::mapToText() file: line:93
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::VPUDevice>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::DataType file: line:101
	pybind11::enum_<VPUNN::DataType>(M("VPUNN"), "DataType", "Supported Datatypes\n\n ")
		.value("UINT8", VPUNN::DataType::UINT8)
		.value("INT8", VPUNN::DataType::INT8)
		.value("FLOAT16", VPUNN::DataType::FLOAT16)
		.value("BFLOAT16", VPUNN::DataType::BFLOAT16)
		.value("__size", VPUNN::DataType::__size);

;

	// VPUNN::mapToText() file: line:109
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::DataType>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::Operation file: line:117
	pybind11::enum_<VPUNN::Operation>(M("VPUNN"), "Operation", "HW operations\n\n ")
		.value("CONVOLUTION", VPUNN::Operation::CONVOLUTION)
		.value("DW_CONVOLUTION", VPUNN::Operation::DW_CONVOLUTION)
		.value("ELTWISE", VPUNN::Operation::ELTWISE)
		.value("MAXPOOL", VPUNN::Operation::MAXPOOL)
		.value("AVEPOOL", VPUNN::Operation::AVEPOOL)
		.value("CM_CONVOLUTION", VPUNN::Operation::CM_CONVOLUTION)
		.value("__size", VPUNN::Operation::__size);

;

	// VPUNN::mapToText() file: line:124
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::Operation>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

}


// File: VPUNN_7.cpp
#include <functional> // std::less
#include <iterator> // __gnu_cxx::__normal_iterator
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_7(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::ActivationFunction file: line:131
	pybind11::enum_<VPUNN::ActivationFunction>(M("VPUNN"), "ActivationFunction", "Supported activation functions\n\n ")
		.value("NONE", VPUNN::ActivationFunction::NONE)
		.value("RELU", VPUNN::ActivationFunction::RELU)
		.value("LRELU", VPUNN::ActivationFunction::LRELU)
		.value("ADD", VPUNN::ActivationFunction::ADD)
		.value("SUB", VPUNN::ActivationFunction::SUB)
		.value("MULT", VPUNN::ActivationFunction::MULT)
		.value("__size", VPUNN::ActivationFunction::__size);

;

	// VPUNN::mapToText() file: line:138
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::ActivationFunction>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::Swizzling file: line:145
	pybind11::enum_<VPUNN::Swizzling>(M("VPUNN"), "Swizzling", "Swizzling keys\n\n ")
		.value("KEY_0", VPUNN::Swizzling::KEY_0)
		.value("KEY_1", VPUNN::Swizzling::KEY_1)
		.value("KEY_2", VPUNN::Swizzling::KEY_2)
		.value("KEY_3", VPUNN::Swizzling::KEY_3)
		.value("KEY_4", VPUNN::Swizzling::KEY_4)
		.value("KEY_5", VPUNN::Swizzling::KEY_5)
		.value("__size", VPUNN::Swizzling::__size);

;

	// VPUNN::mapToText() file: line:151
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::Swizzling>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::ExecutionMode file: line:158
	pybind11::enum_<VPUNN::ExecutionMode>(M("VPUNN"), "ExecutionMode", "DPU execution modes\n\n ")
		.value("VECTOR", VPUNN::ExecutionMode::VECTOR)
		.value("MATRIX", VPUNN::ExecutionMode::MATRIX)
		.value("VECTOR_FP16", VPUNN::ExecutionMode::VECTOR_FP16)
		.value("CUBOID_16x16", VPUNN::ExecutionMode::CUBOID_16x16)
		.value("CUBOID_8x16", VPUNN::ExecutionMode::CUBOID_8x16)
		.value("CUBOID_4x16", VPUNN::ExecutionMode::CUBOID_4x16)
		.value("__size", VPUNN::ExecutionMode::__size);

;

	// VPUNN::mapToText() file: line:165
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::ExecutionMode>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::Layout file: line:186
	pybind11::enum_<VPUNN::Layout>(M("VPUNN"), "Layout", "Data layout\n\n ZMAJOR and CMAJOR are coming from VPU2.0, legacy layouts\n\n  XYZ, XZY, YXZ, YZX, ZXY, ZYX  were introduced for 2.7\n They are to interpreted as from  innermost to outermost dimension of the tensor\n eg: XYZ  is NCHW;   N=Batch is always outermost,  then channels (Z), height (Y), width (X)\n\n INVALID is first usage is exposure to VPUNN in some cases where Layout does not matter, is neither good Like (for\n input_1 when MAXPOOL).\n\n Equivalence legacy to xyz permutations:\n ZMAJOR is Z,X,Y\n CMAJOR is X,Y,Z\n\n ")
		.value("ZMAJOR", VPUNN::Layout::ZMAJOR)
		.value("CMAJOR", VPUNN::Layout::CMAJOR)
		.value("XYZ", VPUNN::Layout::XYZ)
		.value("XZY", VPUNN::Layout::XZY)
		.value("YXZ", VPUNN::Layout::YXZ)
		.value("YZX", VPUNN::Layout::YZX)
		.value("ZXY", VPUNN::Layout::ZXY)
		.value("ZYX", VPUNN::Layout::ZYX)
		.value("INVALID", VPUNN::Layout::INVALID)
		.value("__size", VPUNN::Layout::__size);

;

	// VPUNN::mapToText() file: line:193
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::Layout>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::ISIStrategy file: line:198
	pybind11::enum_<VPUNN::ISIStrategy>(M("VPUNN"), "ISIStrategy", "ISI_Strategy")
		.value("CLUSTERING", VPUNN::ISIStrategy::CLUSTERING)
		.value("SPLIT_OVER_H", VPUNN::ISIStrategy::SPLIT_OVER_H)
		.value("SPLIT_OVER_K", VPUNN::ISIStrategy::SPLIT_OVER_K)
		.value("__size", VPUNN::ISIStrategy::__size);

;

	// VPUNN::mapToText() file: line:205
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::ISIStrategy>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::MemoryLocation file: line:213
	pybind11::enum_<VPUNN::MemoryLocation>(M("VPUNN"), "MemoryLocation", "Memory locations\n\n ")
		.value("DRAM", VPUNN::MemoryLocation::DRAM)
		.value("CMX", VPUNN::MemoryLocation::CMX)
		.value("CSRAM", VPUNN::MemoryLocation::CSRAM)
		.value("UPA", VPUNN::MemoryLocation::UPA)
		.value("__size", VPUNN::MemoryLocation::__size);

;

	// VPUNN::mapToText() file: line:221
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::MemoryLocation>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::VPUSubsystem file: line:229
	pybind11::enum_<VPUNN::VPUSubsystem>(M("VPUNN"), "VPUSubsystem", "VPU Hw subsystem\n\n ")
		.value("VPU_DPU", VPUNN::VPUSubsystem::VPU_DPU)
		.value("VPU_SHV", VPUNN::VPUSubsystem::VPU_SHV)
		.value("VPU_DMA", VPUNN::VPUSubsystem::VPU_DMA)
		.value("VPU_CPU", VPUNN::VPUSubsystem::VPU_CPU)
		.value("VPU_CMX", VPUNN::VPUSubsystem::VPU_CMX)
		.value("__size", VPUNN::VPUSubsystem::__size);

;

	// VPUNN::mapToText() file: line:236
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::VPUSubsystem>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

}


// File: VPUNN_8.cpp

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_8(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::Dim::Grid file: line:244
	pybind11::enum_<VPUNN::Dim::Grid>(M("VPUNN::Dim"), "Grid", pybind11::arithmetic(), "")
		.value("W", VPUNN::Dim::W)
		.value("H", VPUNN::Dim::H)
		.export_values();

;

	// VPUNN::Dim::Act file: line:245
	pybind11::enum_<VPUNN::Dim::Act>(M("VPUNN::Dim"), "Act", pybind11::arithmetic(), "")
		.value("X", VPUNN::Dim::X)
		.value("Y", VPUNN::Dim::Y)
		.value("Z", VPUNN::Dim::Z)
		.value("B", VPUNN::Dim::B)
		.export_values();

;

	// VPUNN::Dim::Wt file: line:246
	pybind11::enum_<VPUNN::Dim::Wt>(M("VPUNN::Dim"), "Wt", pybind11::arithmetic(), "")
		.value("K", VPUNN::Dim::K)
		.value("C", VPUNN::Dim::C)
		.value("Ky", VPUNN::Dim::Ky)
		.value("Kx", VPUNN::Dim::Kx)
		.export_values();

;

	// VPUNN::Dim::Padding file: line:247
	pybind11::enum_<VPUNN::Dim::Padding>(M("VPUNN::Dim"), "Padding", pybind11::arithmetic(), "")
		.value("TOP", VPUNN::Dim::TOP)
		.value("BOTTOM", VPUNN::Dim::BOTTOM)
		.value("LEFT", VPUNN::Dim::LEFT)
		.value("RIGHT", VPUNN::Dim::RIGHT)
		.export_values();

;

}


// File: VPUNN_9.cpp
#include <array> // std::array
#include <ios> // std::_Ios_Seekdir
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SWOperation file: line:767
struct PyCallBack_VPUNN_SWOperation : public VPUNN::SWOperation {
	using VPUNN::SWOperation::SWOperation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SWOperation *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"SWOperation::cycles\"");
	}
};

void bind_VPUNN_9(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::dtype_to_bytes(enum VPUNN::DataType) file: line:255
	M("VPUNN").def("dtype_to_bytes", (unsigned int (*)(enum VPUNN::DataType)) &VPUNN::dtype_to_bytes, "Get the size of the dtype\n\n \n a DataType object\n \n\n size in bytes\n\nC++: VPUNN::dtype_to_bytes(enum VPUNN::DataType) --> unsigned int", pybind11::arg("dtype"));

	// VPUNN::getDefaultLayout() file: line:266
	M("VPUNN").def("getDefaultLayout", (enum VPUNN::Layout (*)()) &VPUNN::getDefaultLayout, "default Layout that is equivalent with legacy ZMAJOR\n\nC++: VPUNN::getDefaultLayout() --> enum VPUNN::Layout");

	// VPUNN::layout_to_order(enum VPUNN::Layout) file: line:278
	M("VPUNN").def("layout_to_order", (struct std::array<unsigned int, 4> (*)(enum VPUNN::Layout)) &VPUNN::layout_to_order, "Get the tensor serial order given a layout\n\n \n a Tensor Layout\n \n\n std::array<unsigned int, 4>, order of dimensions from innermost to outermost. values represent Dim::Act\n\n Invalid will be mapped to the default one : ZMAJOR/ZXY\n\nC++: VPUNN::layout_to_order(enum VPUNN::Layout) --> struct std::array<unsigned int, 4>", pybind11::arg("layout"));

	// VPUNN::memoryLocation(enum VPUNN::VPUDevice) file: line:312
	M("VPUNN").def("memoryLocation", (class std::vector<enum VPUNN::MemoryLocation, class std::allocator<enum VPUNN::MemoryLocation> > (*)(enum VPUNN::VPUDevice)) &VPUNN::memoryLocation, "Return the available memory locations per VPU IP generation\n\n \n a VPUDevice representing the VPU IP generation\n \n\n std::vector<MemoryLocation>\n\nC++: VPUNN::memoryLocation(enum VPUNN::VPUDevice) --> class std::vector<enum VPUNN::MemoryLocation, class std::allocator<enum VPUNN::MemoryLocation> >", pybind11::arg("device"));

	// VPUNN::isMemoryLocationAvailable(enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation) file: line:331
	M("VPUNN").def("isMemoryLocationAvailable", (bool (*)(enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation)) &VPUNN::isMemoryLocationAvailable, "Check if a memory location is available for a specific VPU IP generation\n\n \n a VPUDevice representing the VPU IP generation\n \n\n a memory location\n \n\n true\n \n\n false\n\nC++: VPUNN::isMemoryLocationAvailable(enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation) --> bool", pybind11::arg("device"), pybind11::arg("location"));

	// VPUNN::mpe_mode_to_grid(enum VPUNN::ExecutionMode) file: line:342
	M("VPUNN").def("mpe_mode_to_grid", (class std::vector<unsigned int, class std::allocator<unsigned int> > (*)(enum VPUNN::ExecutionMode)) &VPUNN::mpe_mode_to_grid, "Return grid in X, Y, Z, B format\n\n \n a DPUWorkload ExecutionMode\n \n\n std::vector<unsigned int>\n\nC++: VPUNN::mpe_mode_to_grid(enum VPUNN::ExecutionMode) --> class std::vector<unsigned int, class std::allocator<unsigned int> >", pybind11::arg("mode"));

	// VPUNN::mpe_mode_to_nthw_ntk_grid(enum VPUNN::ExecutionMode) file: line:359
	M("VPUNN").def("mpe_mode_to_nthw_ntk_grid", (class std::vector<unsigned int, class std::allocator<unsigned int> > (*)(enum VPUNN::ExecutionMode)) &VPUNN::mpe_mode_to_nthw_ntk_grid, "Return the NTHW/NTK grid in X, Y, Z, B format\n\n \n a DPUWorkload ExecutionMode\n \n\n std::vector<unsigned int>\n\nC++: VPUNN::mpe_mode_to_nthw_ntk_grid(enum VPUNN::ExecutionMode) --> class std::vector<unsigned int, class std::allocator<unsigned int> >", pybind11::arg("mode"));

	{ // VPUNN::VPUTensor file: line:376
		pybind11::class_<VPUNN::VPUTensor, std::shared_ptr<VPUNN::VPUTensor>> cl(M("VPUNN"), "VPUTensor", "Cost model tensor class\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUTensor(); } ), "doc" );
		cl.def( pybind11::init( [](const struct std::array<unsigned int, 4> & a0){ return new VPUNN::VPUTensor(a0); } ), "doc" , pybind11::arg("shape"));
		cl.def( pybind11::init( [](const struct std::array<unsigned int, 4> & a0, enum VPUNN::DataType const & a1){ return new VPUNN::VPUTensor(a0, a1); } ), "doc" , pybind11::arg("shape"), pybind11::arg("dtype"));
		cl.def( pybind11::init( [](const struct std::array<unsigned int, 4> & a0, enum VPUNN::DataType const & a1, enum VPUNN::Layout const & a2){ return new VPUNN::VPUTensor(a0, a1, a2); } ), "doc" , pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("layout"));
		cl.def( pybind11::init<const struct std::array<unsigned int, 4> &, enum VPUNN::DataType, enum VPUNN::Layout, bool>(), pybind11::arg("shape"), pybind11::arg("dtype"), pybind11::arg("layout"), pybind11::arg("sparsity") );

		cl.def( pybind11::init( [](unsigned int const & a0, unsigned int const & a1, unsigned int const & a2, unsigned int const & a3, enum VPUNN::DataType const & a4){ return new VPUNN::VPUTensor(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("width"), pybind11::arg("height"), pybind11::arg("channels"), pybind11::arg("batch"), pybind11::arg("dtype"));
		cl.def( pybind11::init( [](unsigned int const & a0, unsigned int const & a1, unsigned int const & a2, unsigned int const & a3, enum VPUNN::DataType const & a4, enum VPUNN::Layout const & a5){ return new VPUNN::VPUTensor(a0, a1, a2, a3, a4, a5); } ), "doc" , pybind11::arg("width"), pybind11::arg("height"), pybind11::arg("channels"), pybind11::arg("batch"), pybind11::arg("dtype"), pybind11::arg("layout"));
		cl.def( pybind11::init<unsigned int, unsigned int, unsigned int, unsigned int, enum VPUNN::DataType, enum VPUNN::Layout, bool>(), pybind11::arg("width"), pybind11::arg("height"), pybind11::arg("channels"), pybind11::arg("batch"), pybind11::arg("dtype"), pybind11::arg("layout"), pybind11::arg("sparsity") );

		cl.def( pybind11::init<const struct std::array<unsigned int, 4> &, const class VPUNN::VPUTensor &>(), pybind11::arg("shape_"), pybind11::arg("rest") );

		cl.def( pybind11::init( [](VPUNN::VPUTensor const &o){ return new VPUNN::VPUTensor(o); } ) );
		cl.def("x", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::x, "Get the x dimension\n\nC++: VPUNN::VPUTensor::x() const --> unsigned int");
		cl.def("y", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::y, "Get the y dimension\n\nC++: VPUNN::VPUTensor::y() const --> unsigned int");
		cl.def("z", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::z, "Get the z dimension\n\nC++: VPUNN::VPUTensor::z() const --> unsigned int");
		cl.def("b", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::b, "Get the batch dimension\n\nC++: VPUNN::VPUTensor::b() const --> unsigned int");
		cl.def("sx", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::sx, "Get the x dimension stride\n\nC++: VPUNN::VPUTensor::sx() const --> unsigned int");
		cl.def("sy", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::sy, "Get the y dimension stride\n\nC++: VPUNN::VPUTensor::sy() const --> unsigned int");
		cl.def("sz", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::sz, "Get the z dimension stride\n\nC++: VPUNN::VPUTensor::sz() const --> unsigned int");
		cl.def("sb", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::sb, "Get the batch dimension stride\n\nC++: VPUNN::VPUTensor::sb() const --> unsigned int");
		cl.def("height", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::height, "Get the height\n\nC++: VPUNN::VPUTensor::height() const --> unsigned int");
		cl.def("width", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::width, "Get the width\n\nC++: VPUNN::VPUTensor::width() const --> unsigned int");
		cl.def("channels", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::channels, "Get the channels\n\nC++: VPUNN::VPUTensor::channels() const --> unsigned int");
		cl.def("batches", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::batches, "Get the batches dimension\n\nC++: VPUNN::VPUTensor::batches() const --> unsigned int");
		cl.def("size", (unsigned int (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::size, "Get the size in bytes\n \n\n size in bytes\n\nC++: VPUNN::VPUTensor::size() const --> unsigned int");
		cl.def("is_float", (bool (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::is_float, "Check if the tensor is floating point\n \n\n true if floating point type\n\nC++: VPUNN::VPUTensor::is_float() const --> bool");
		cl.def("is_int", (bool (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::is_int, "Check if the tensor is integer\n \n\n true if integer type\n\nC++: VPUNN::VPUTensor::is_int() const --> bool");
		cl.def("get_shape", (const struct std::array<unsigned int, 4> & (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::get_shape, "Get the shape\n \n\n a 4 vector containing the shape in convention XYZB\n\nC++: VPUNN::VPUTensor::get_shape() const --> const struct std::array<unsigned int, 4> &", pybind11::return_value_policy::automatic);
		cl.def("set_shape", (void (VPUNN::VPUTensor::*)(struct std::array<unsigned int, 4>)) &VPUNN::VPUTensor::set_shape, "Set the VPUTensor shape\n \n\n in convention XYZB\n\nC++: VPUNN::VPUTensor::set_shape(struct std::array<unsigned int, 4>) --> void", pybind11::arg("in_shape"));
		cl.def("get_dtype", (enum VPUNN::DataType (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::get_dtype, "Get the datatype\n\nC++: VPUNN::VPUTensor::get_dtype() const --> enum VPUNN::DataType");
		cl.def("change_datatype_superficial", (enum VPUNN::DataType (VPUNN::VPUTensor::*)(enum VPUNN::DataType)) &VPUNN::VPUTensor::change_datatype_superficial, "changes the underlying data type only if same size new vs old\n \n\n newly set type.\n\nC++: VPUNN::VPUTensor::change_datatype_superficial(enum VPUNN::DataType) --> enum VPUNN::DataType", pybind11::arg("new_datatype"));
		cl.def("get_layout", (enum VPUNN::Layout (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::get_layout, "Get the layout\n\nC++: VPUNN::VPUTensor::get_layout() const --> enum VPUNN::Layout");
		cl.def("set_if_same_layout", (bool (VPUNN::VPUTensor::*)(enum VPUNN::Layout)) &VPUNN::VPUTensor::set_if_same_layout, "changes the layout type if new one has the same structure as old\n this change must not affect the shape or strides\n \n\n the desired layout\n \n\n true if new layout set, false otherwise\n\nC++: VPUNN::VPUTensor::set_if_same_layout(enum VPUNN::Layout) --> bool", pybind11::arg("new_layout"));
		cl.def("get_sparsity", (bool (VPUNN::VPUTensor::*)() const) &VPUNN::VPUTensor::get_sparsity, "Get the sparsity flag\n\nC++: VPUNN::VPUTensor::get_sparsity() const --> bool");
		cl.def("__eq__", (bool (VPUNN::VPUTensor::*)(const class VPUNN::VPUTensor &) const) &VPUNN::VPUTensor::operator==, "equality test operator\n\nC++: VPUNN::VPUTensor::operator==(const class VPUNN::VPUTensor &) const --> bool", pybind11::arg("b"));
		cl.def("assign", (class VPUNN::VPUTensor & (VPUNN::VPUTensor::*)(const class VPUNN::VPUTensor &)) &VPUNN::VPUTensor::operator=, "C++: VPUNN::VPUTensor::operator=(const class VPUNN::VPUTensor &) --> class VPUNN::VPUTensor &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		cl.def("__str__", [](VPUNN::VPUTensor const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	// VPUNN::default_init_swizzling() file: line:581
	M("VPUNN").def("default_init_swizzling", (enum VPUNN::Swizzling (*)()) &VPUNN::default_init_swizzling, "C++: VPUNN::default_init_swizzling() --> enum VPUNN::Swizzling");

	{ // VPUNN::DPUWorkload file: line:586
		pybind11::class_<VPUNN::DPUWorkload, std::shared_ptr<VPUNN::DPUWorkload>> cl(M("VPUNN"), "DPUWorkload", "The base structure that encodes a DPU workloads");
		cl.def( pybind11::init( [](){ return new VPUNN::DPUWorkload(); } ) );
		cl.def( pybind11::init( [](VPUNN::DPUWorkload const &o){ return new VPUNN::DPUWorkload(o); } ) );
		cl.def_readwrite("device", &VPUNN::DPUWorkload::device);
		cl.def_readwrite("op", &VPUNN::DPUWorkload::op);
		cl.def_readwrite("inputs", &VPUNN::DPUWorkload::inputs);
		cl.def_readwrite("outputs", &VPUNN::DPUWorkload::outputs);
		cl.def_readwrite("kernels", &VPUNN::DPUWorkload::kernels);
		cl.def_readwrite("strides", &VPUNN::DPUWorkload::strides);
		cl.def_readwrite("padding", &VPUNN::DPUWorkload::padding);
		cl.def_readwrite("execution_order", &VPUNN::DPUWorkload::execution_order);
		cl.def_readwrite("activation_function", &VPUNN::DPUWorkload::activation_function);
		cl.def_readwrite("act_sparsity", &VPUNN::DPUWorkload::act_sparsity);
		cl.def_readwrite("weight_sparsity", &VPUNN::DPUWorkload::weight_sparsity);
		cl.def_readwrite("input_swizzling", &VPUNN::DPUWorkload::input_swizzling);
		cl.def_readwrite("output_swizzling", &VPUNN::DPUWorkload::output_swizzling);
		cl.def_readwrite("output_write_tiles", &VPUNN::DPUWorkload::output_write_tiles);
		cl.def_readwrite("offsets", &VPUNN::DPUWorkload::offsets);
		cl.def_readwrite("isi_strategy", &VPUNN::DPUWorkload::isi_strategy);
		cl.def_readwrite("weight_sparsity_enabled", &VPUNN::DPUWorkload::weight_sparsity_enabled);
		cl.def("__eq__", (bool (VPUNN::DPUWorkload::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::DPUWorkload::operator==, "equality test operator\n\nC++: VPUNN::DPUWorkload::operator==(const struct VPUNN::DPUWorkload &) const --> bool", pybind11::arg("b"));
		cl.def("assign", (struct VPUNN::DPUWorkload & (VPUNN::DPUWorkload::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::DPUWorkload::operator=, "C++: VPUNN::DPUWorkload::operator=(const struct VPUNN::DPUWorkload &) --> struct VPUNN::DPUWorkload &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		cl.def("__str__", [](VPUNN::DPUWorkload const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::DMAWorkload file: line:716
		pybind11::class_<VPUNN::DMAWorkload, std::shared_ptr<VPUNN::DMAWorkload>> cl(M("VPUNN"), "DMAWorkload", "The base structure that encodes a DMA workloads\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::DMAWorkload(); } ) );
		cl.def( pybind11::init( [](VPUNN::DMAWorkload const &o){ return new VPUNN::DMAWorkload(o); } ) );
		cl.def_readwrite("device", &VPUNN::DMAWorkload::device);
		cl.def_readwrite("input", &VPUNN::DMAWorkload::input);
		cl.def_readwrite("output", &VPUNN::DMAWorkload::output);
		cl.def_readwrite("input_location", &VPUNN::DMAWorkload::input_location);
		cl.def_readwrite("output_location", &VPUNN::DMAWorkload::output_location);
		cl.def_readwrite("output_write_tiles", &VPUNN::DMAWorkload::output_write_tiles);
		cl.def_static("sizeTODELETEME", (unsigned int (*)()) &VPUNN::DMAWorkload::sizeTODELETEME, "This function computes the size of the DMAWorkload features to feed to the NN\n\n \n unsigned int\n\nC++: VPUNN::DMAWorkload::sizeTODELETEME() --> unsigned int");

		cl.def("__str__", [](VPUNN::DMAWorkload const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::SWOperation file: line:767
		pybind11::class_<VPUNN::SWOperation, std::shared_ptr<VPUNN::SWOperation>, PyCallBack_VPUNN_SWOperation> cl(M("VPUNN"), "SWOperation", "The base structure that encodes a Software layer\n\n ");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("outputs") );

		cl.def(pybind11::init<PyCallBack_VPUNN_SWOperation const &>());
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHAVEWorkload file: line:794
		pybind11::class_<VPUNN::SHAVEWorkload, std::shared_ptr<VPUNN::SHAVEWorkload>> cl(M("VPUNN"), "SHAVEWorkload", "describes a Software layer (SHAVE) request");
		cl.def( pybind11::init<const std::string &, const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &>(), pybind11::arg("operation_name"), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("outputs") );

		cl.def( pybind11::init( [](VPUNN::SHAVEWorkload const &o){ return new VPUNN::SHAVEWorkload(o); } ) );
		cl.def("assign", (class VPUNN::SHAVEWorkload & (VPUNN::SHAVEWorkload::*)(const class VPUNN::SHAVEWorkload &)) &VPUNN::SHAVEWorkload::operator=, "C++: VPUNN::SHAVEWorkload::operator=(const class VPUNN::SHAVEWorkload &) --> class VPUNN::SHAVEWorkload &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("get_name", (std::string (VPUNN::SHAVEWorkload::*)() const) &VPUNN::SHAVEWorkload::get_name, "C++: VPUNN::SHAVEWorkload::get_name() const --> std::string");
		cl.def("get_device", (enum VPUNN::VPUDevice (VPUNN::SHAVEWorkload::*)() const) &VPUNN::SHAVEWorkload::get_device, "C++: VPUNN::SHAVEWorkload::get_device() const --> enum VPUNN::VPUDevice");
		cl.def("get_inputs", (const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > & (VPUNN::SHAVEWorkload::*)() const) &VPUNN::SHAVEWorkload::get_inputs, "C++: VPUNN::SHAVEWorkload::get_inputs() const --> const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &", pybind11::return_value_policy::automatic);
		cl.def("get_outputs", (const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > & (VPUNN::SHAVEWorkload::*)() const) &VPUNN::SHAVEWorkload::get_outputs, "C++: VPUNN::SHAVEWorkload::get_outputs() const --> const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &", pybind11::return_value_policy::automatic);
		cl.def("toString", (std::string (VPUNN::SHAVEWorkload::*)() const) &VPUNN::SHAVEWorkload::toString, "C++: VPUNN::SHAVEWorkload::toString() const --> std::string");

		cl.def("__str__", [](VPUNN::SHAVEWorkload const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
}


// File: VPUNN_10.cpp
#include <array> // std::array
#include <ios> // std::_Ios_Openmode
#include <ios> // std::_Ios_Seekdir
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <sstream> // std::basic_ostringstream
#include <sstream> // std::basic_stringbuf
#include <stdexcept> // std::runtime_error
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::IOperationDynamicConstraints file: line:23
struct PyCallBack_VPUNN_IOperationDynamicConstraints : public VPUNN::IOperationDynamicConstraints {
	using VPUNN::IOperationDynamicConstraints::IOperationDynamicConstraints;

	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::input_1_volume\"");
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::input_1_aligned_size_bytes\"");
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::input_1_contiguous_size_bytes\"");
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::deduce_input_1\"");
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOperationDynamicConstraints::limit_sparsity(a0, a1);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::check_input_output_tensor_corelation\"");
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicConstraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::check_sparsity_rules\"");
	}
};

void bind_VPUNN_10(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::LogLevel file: line:25
	pybind11::enum_<VPUNN::LogLevel>(M("VPUNN"), "LogLevel", "Logger verbosity levels (same as VPUX)\n\n ")
		.value("None", VPUNN::LogLevel::None)
		.value("Fatal", VPUNN::LogLevel::Fatal)
		.value("Error", VPUNN::LogLevel::Error)
		.value("Warning", VPUNN::LogLevel::Warning)
		.value("Info", VPUNN::LogLevel::Info)
		.value("Debug", VPUNN::LogLevel::Debug)
		.value("Trace", VPUNN::LogLevel::Trace);

;

	// VPUNN::toString(enum VPUNN::LogLevel) file: line:46
	M("VPUNN").def("toString", (const std::string (*)(enum VPUNN::LogLevel)) &VPUNN::toString, "Convert a LegLevel to an uppercase string\n\n \n\n \n\n const std::string\n\nC++: VPUNN::toString(enum VPUNN::LogLevel) --> const std::string", pybind11::arg("level"));

	{ // VPUNN::LoggerStream file: line:53
		pybind11::class_<VPUNN::LoggerStream, std::shared_ptr<VPUNN::LoggerStream>> cl(M("VPUNN"), "LoggerStream", "A class that implements a cout-enabled interface for logging\n\n ");
		cl.def( pybind11::init( [](enum VPUNN::LogLevel const & a0, bool const & a1){ return new VPUNN::LoggerStream(a0, a1); } ), "doc" , pybind11::arg("level"), pybind11::arg("enabled"));
		cl.def( pybind11::init<enum VPUNN::LogLevel, bool, class std::basic_ostringstream<char> *>(), pybind11::arg("level"), pybind11::arg("enabled"), pybind11::arg("buff") );

		cl.def( pybind11::init( [](VPUNN::LoggerStream const &o){ return new VPUNN::LoggerStream(o); } ) );
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const std::string &)) &VPUNN::LoggerStream::operator<<<std::string>, "C++: VPUNN::LoggerStream::operator<<(const std::string &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const double &)) &VPUNN::LoggerStream::operator<<<double>, "C++: VPUNN::LoggerStream::operator<<(const double &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const unsigned int &)) &VPUNN::LoggerStream::operator<<<unsigned int>, "C++: VPUNN::LoggerStream::operator<<(const unsigned int &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const struct VPUNN::DPULayer &)) &VPUNN::LoggerStream::operator<<<VPUNN::DPULayer>, "C++: VPUNN::LoggerStream::operator<<(const struct VPUNN::DPULayer &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const int &)) &VPUNN::LoggerStream::operator<<<int>, "C++: VPUNN::LoggerStream::operator<<(const int &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const char *const &)) &VPUNN::LoggerStream::operator<<<const char *>, "C++: VPUNN::LoggerStream::operator<<(const char *const &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("__lshift__", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const unsigned long &)) &VPUNN::LoggerStream::operator<<<unsigned long>, "C++: VPUNN::LoggerStream::operator<<(const unsigned long &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg("msg"));
		cl.def("assign", (class VPUNN::LoggerStream & (VPUNN::LoggerStream::*)(const class VPUNN::LoggerStream &)) &VPUNN::LoggerStream::operator=, "C++: VPUNN::LoggerStream::operator=(const class VPUNN::LoggerStream &) --> class VPUNN::LoggerStream &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::Logger file: line:115
		pybind11::class_<VPUNN::Logger, std::shared_ptr<VPUNN::Logger>> cl(M("VPUNN"), "Logger", "Logger class\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::Logger(); } ) );
		cl.def_static("clear2ndlog", (void (*)()) &VPUNN::Logger::clear2ndlog, "C++: VPUNN::Logger::clear2ndlog() --> void");
		cl.def_static("get2ndlog", (std::string (*)()) &VPUNN::Logger::get2ndlog, "C++: VPUNN::Logger::get2ndlog() --> std::string");
		cl.def_static("activate2ndlog", (void (*)()) &VPUNN::Logger::activate2ndlog, "C++: VPUNN::Logger::activate2ndlog() --> void");
		cl.def_static("deactivate2ndlog", (void (*)()) &VPUNN::Logger::deactivate2ndlog, "C++: VPUNN::Logger::deactivate2ndlog() --> void");
		cl.def_static("initialize", []() -> void { return VPUNN::Logger::initialize(); }, "");
		cl.def_static("initialize", (void (*)(enum VPUNN::LogLevel)) &VPUNN::Logger::initialize, "Initialize the Logger\n\n \n verbosity\n\nC++: VPUNN::Logger::initialize(enum VPUNN::LogLevel) --> void", pybind11::arg("level"));
		cl.def_static("level", (enum VPUNN::LogLevel (*)()) &VPUNN::Logger::level, "Return the verbosity level\n\n \n auto\n\nC++: VPUNN::Logger::level() --> enum VPUNN::LogLevel");
		cl.def_static("setLevel", (void (*)(enum VPUNN::LogLevel)) &VPUNN::Logger::setLevel, "Set the verbosity level\n\n \n\n     \n\nC++: VPUNN::Logger::setLevel(enum VPUNN::LogLevel) --> void", pybind11::arg("level"));
		cl.def_static("enabled", (bool (*)()) &VPUNN::Logger::enabled, "get if the Logger is enabled or not\n\n \n true\n \n\n false\n\nC++: VPUNN::Logger::enabled() --> bool");
		cl.def_static("fatal", (class VPUNN::LoggerStream (*)()) &VPUNN::Logger::fatal, "Fatal error\n\n \n auto\n\nC++: VPUNN::Logger::fatal() --> class VPUNN::LoggerStream");
		cl.def_static("error", (class VPUNN::LoggerStream (*)()) &VPUNN::Logger::error, "Error message\n\n \n auto\n\nC++: VPUNN::Logger::error() --> class VPUNN::LoggerStream");
		cl.def_static("warning", (class VPUNN::LoggerStream (*)()) &VPUNN::Logger::warning, "Warning message\n\n \n auto\n\nC++: VPUNN::Logger::warning() --> class VPUNN::LoggerStream");
		cl.def_static("info", (class VPUNN::LoggerStream (*)()) &VPUNN::Logger::info, "Info message\n\n \n auto\n\nC++: VPUNN::Logger::info() --> class VPUNN::LoggerStream");
		cl.def_static("debug", (class VPUNN::LoggerStream (*)()) &VPUNN::Logger::debug, "Debug message\n\n \n auto\n\nC++: VPUNN::Logger::debug() --> class VPUNN::LoggerStream");
		cl.def_static("trace", (class VPUNN::LoggerStream (*)()) &VPUNN::Logger::trace, "Trace message\n\n \n auto\n\nC++: VPUNN::Logger::trace() --> class VPUNN::LoggerStream");
	}
	// VPUNN::throw_error(std::string) file: line:247
	M("VPUNN").def("throw_error", (void (*)(std::string)) &VPUNN::throw_error<std::runtime_error>, "C++: VPUNN::throw_error(std::string) --> void", pybind11::arg("msg"));

	{ // VPUNN::TensorInfo file: line:22
		pybind11::class_<VPUNN::TensorInfo, std::shared_ptr<VPUNN::TensorInfo>> cl(M("VPUNN"), "TensorInfo", "holds info for a tensor.");
		cl.def( pybind11::init<const class VPUNN::VPUTensor &>(), pybind11::arg("t") );

		cl.def( pybind11::init( [](){ return new VPUNN::TensorInfo(); } ) );
		cl.def( pybind11::init( [](VPUNN::TensorInfo const &o){ return new VPUNN::TensorInfo(o); } ) );
		cl.def_readwrite("height", &VPUNN::TensorInfo::height);
		cl.def_readwrite("width", &VPUNN::TensorInfo::width);
		cl.def_readwrite("channels", &VPUNN::TensorInfo::channels);
		cl.def_readwrite("batch", &VPUNN::TensorInfo::batch);
		cl.def_readwrite("datatype", &VPUNN::TensorInfo::datatype);
		cl.def_readwrite("layout", &VPUNN::TensorInfo::layout);
		cl.def_readwrite("sparsity", &VPUNN::TensorInfo::sparsity);
		cl.def_readwrite("sparsity_enabled", &VPUNN::TensorInfo::sparsity_enabled);
		cl.def_readwrite("swizzling", &VPUNN::TensorInfo::swizzling);

		cl.def("__str__", [](VPUNN::TensorInfo const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::KernelInfo file: line:48
		pybind11::class_<VPUNN::KernelInfo, std::shared_ptr<VPUNN::KernelInfo>> cl(M("VPUNN"), "KernelInfo", "kernel related informations, including stride and padding");
		cl.def( pybind11::init<const struct VPUNN::DPUWorkload &>(), pybind11::arg("w") );

		cl.def( pybind11::init( [](){ return new VPUNN::KernelInfo(); } ) );
		cl.def( pybind11::init( [](VPUNN::KernelInfo const &o){ return new VPUNN::KernelInfo(o); } ) );
		cl.def_readwrite("height", &VPUNN::KernelInfo::height);
		cl.def_readwrite("width", &VPUNN::KernelInfo::width);
		cl.def_readwrite("pad_bottom", &VPUNN::KernelInfo::pad_bottom);
		cl.def_readwrite("pad_left", &VPUNN::KernelInfo::pad_left);
		cl.def_readwrite("pad_right", &VPUNN::KernelInfo::pad_right);
		cl.def_readwrite("pad_top", &VPUNN::KernelInfo::pad_top);
		cl.def_readwrite("stride_height", &VPUNN::KernelInfo::stride_height);
		cl.def_readwrite("stride_width", &VPUNN::KernelInfo::stride_width);

		cl.def("__str__", [](VPUNN::KernelInfo const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::DPUOperation file: line:76
		pybind11::class_<VPUNN::DPUOperation, std::shared_ptr<VPUNN::DPUOperation>> cl(M("VPUNN"), "DPUOperation", "local type describing a workload\n easy to change and adapt without touching the DPUWorkload interface");
		cl.def( pybind11::init<const struct VPUNN::DPUWorkload &>(), pybind11::arg("w") );

		cl.def( pybind11::init( [](){ return new VPUNN::DPUOperation(); } ) );
		cl.def( pybind11::init( [](VPUNN::DPUOperation const &o){ return new VPUNN::DPUOperation(o); } ) );
		cl.def_readwrite("device", &VPUNN::DPUOperation::device);
		cl.def_readwrite("operation", &VPUNN::DPUOperation::operation);
		cl.def_readwrite("input_0", &VPUNN::DPUOperation::input_0);
		cl.def_readwrite("input_1", &VPUNN::DPUOperation::input_1);
		cl.def_readwrite("output_0", &VPUNN::DPUOperation::output_0);
		cl.def_readwrite("execution_order", &VPUNN::DPUOperation::execution_order);
		cl.def_readwrite("kernel", &VPUNN::DPUOperation::kernel);
		cl.def_readwrite("output_write_tiles", &VPUNN::DPUOperation::output_write_tiles);
		cl.def_readwrite("isi_strategy", &VPUNN::DPUOperation::isi_strategy);
		cl.def("set_intended_split", (void (VPUNN::DPUOperation::*)(enum VPUNN::ISIStrategy, unsigned int)) &VPUNN::DPUOperation::set_intended_split, "C++: VPUNN::DPUOperation::set_intended_split(enum VPUNN::ISIStrategy, unsigned int) --> void", pybind11::arg("strategy"), pybind11::arg("nTiles"));
		cl.def("clone_as_DPUWorkload", (struct VPUNN::DPUWorkload (VPUNN::DPUOperation::*)() const) &VPUNN::DPUOperation::clone_as_DPUWorkload, "C++: VPUNN::DPUOperation::clone_as_DPUWorkload() const --> struct VPUNN::DPUWorkload");

		cl.def("__str__", [](VPUNN::DPUOperation const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::IOperationDynamicConstraints file: line:23
		pybind11::class_<VPUNN::IOperationDynamicConstraints, VPUNN::IOperationDynamicConstraints*, PyCallBack_VPUNN_IOperationDynamicConstraints> cl(M("VPUNN"), "IOperationDynamicConstraints", "Interface class for constraints/behaviors that are specific to operations\n It enforces dynamically the workload setup. Derived classes will implement specific rules based on the operation");
		cl.def(pybind11::init<PyCallBack_VPUNN_IOperationDynamicConstraints const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_IOperationDynamicConstraints(); } ) );
		cl.def("input_1_volume", (long long (VPUNN::IOperationDynamicConstraints::*)(const struct VPUNN::TensorInfo &) const) &VPUNN::IOperationDynamicConstraints::input_1_volume, "computes size of weights (input_1) in elements not bytes\n\nC++: VPUNN::IOperationDynamicConstraints::input_1_volume(const struct VPUNN::TensorInfo &) const --> long long", pybind11::arg("w"));
		cl.def("input_1_aligned_size_bytes", (long long (VPUNN::IOperationDynamicConstraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const) &VPUNN::IOperationDynamicConstraints::input_1_aligned_size_bytes, "computes the aligned size in bytes for weights of a workload\n\nC++: VPUNN::IOperationDynamicConstraints::input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const --> long long", pybind11::arg("config"), pybind11::arg("dpu"));
		cl.def("input_1_contiguous_size_bytes", (long long (VPUNN::IOperationDynamicConstraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const) &VPUNN::IOperationDynamicConstraints::input_1_contiguous_size_bytes, "computes the non CMX aligned/contiguous  size in bytes for the weights\n\nC++: VPUNN::IOperationDynamicConstraints::input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const --> long long", pybind11::arg("config"), pybind11::arg("dpu"));
		cl.def("input_0_volume", (long long (VPUNN::IOperationDynamicConstraints::*)(const struct VPUNN::TensorInfo &) const) &VPUNN::IOperationDynamicConstraints::input_0_volume, "computes size of activators (input_0)\n\nC++: VPUNN::IOperationDynamicConstraints::input_0_volume(const struct VPUNN::TensorInfo &) const --> long long", pybind11::arg("w"));
		cl.def("output_0_volume", (long long (VPUNN::IOperationDynamicConstraints::*)(const struct VPUNN::TensorInfo &) const) &VPUNN::IOperationDynamicConstraints::output_0_volume, "computes size of activators (input_0)\n\nC++: VPUNN::IOperationDynamicConstraints::output_0_volume(const struct VPUNN::TensorInfo &) const --> long long", pybind11::arg("w"));
		cl.def("deduce_input_1", (void (VPUNN::IOperationDynamicConstraints::*)(const struct VPUNN::TensorInfo &, const struct VPUNN::TensorInfo &, const class VPUNN::IDeviceValidValues &, const struct VPUNN::KernelInfo &, struct VPUNN::TensorInfo &) const) &VPUNN::IOperationDynamicConstraints::deduce_input_1, "deduce input_1 based on input_0 and output_0,\n deduce the weights\n\nC++: VPUNN::IOperationDynamicConstraints::deduce_input_1(const struct VPUNN::TensorInfo &, const struct VPUNN::TensorInfo &, const class VPUNN::IDeviceValidValues &, const struct VPUNN::KernelInfo &, struct VPUNN::TensorInfo &) const --> void", pybind11::arg("in_0"), pybind11::arg("out_0"), pybind11::arg("config"), pybind11::arg("kernel"), pybind11::arg("w"));
		cl.def("filter_ISI_Strategy_Options", (class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > (VPUNN::IOperationDynamicConstraints::*)(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > &) const) &VPUNN::IOperationDynamicConstraints::filter_ISI_Strategy_Options, "a filtered strategy container that has the invalid ones eliminated. Operation dependent.\n\nC++: VPUNN::IOperationDynamicConstraints::filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > &) const --> class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >", pybind11::arg("strategies"));
		cl.def("filter_output_write_tile_Options", (class std::vector<int, class std::allocator<int> > (VPUNN::IOperationDynamicConstraints::*)(const class std::vector<int, class std::allocator<int> > &) const) &VPUNN::IOperationDynamicConstraints::filter_output_write_tile_Options, "a output_write_tile container that has the invalid ones eliminated. Operation dependent.\n\nC++: VPUNN::IOperationDynamicConstraints::filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("output_write_tile_variants"));
		cl.def("normalize_kernel_dimension", (bool (VPUNN::IOperationDynamicConstraints::*)(const enum VPUNN::ISIStrategy &, struct VPUNN::KernelInfo &) const) &VPUNN::IOperationDynamicConstraints::normalize_kernel_dimension, "changes kernels in case a stricter constraint must be used\n \n\n true if normalization was done (kernel changed)\n\nC++: VPUNN::IOperationDynamicConstraints::normalize_kernel_dimension(const enum VPUNN::ISIStrategy &, struct VPUNN::KernelInfo &) const --> bool", pybind11::arg(""), pybind11::arg(""));
		cl.def("limit_sparsity", (void (VPUNN::IOperationDynamicConstraints::*)(const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const) &VPUNN::IOperationDynamicConstraints::limit_sparsity, "@ reduces/adjusts sparsity  according to context\n\nC++: VPUNN::IOperationDynamicConstraints::limit_sparsity(const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const --> void", pybind11::arg(""), pybind11::arg(""));
		cl.def("check_input_output_tensor_corelation", (bool (VPUNN::IOperationDynamicConstraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const) &VPUNN::IOperationDynamicConstraints::check_input_output_tensor_corelation, "checks that the sizes of inputs and output tensors are good given the operation.\n\nC++: VPUNN::IOperationDynamicConstraints::check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const --> bool", pybind11::arg("config"), pybind11::arg("dpu"), pybind11::arg("info"));
		cl.def("check_sparsity_rules", (bool (VPUNN::IOperationDynamicConstraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const) &VPUNN::IOperationDynamicConstraints::check_sparsity_rules, "checks that the sparsity respects operation constraints\n\nC++: VPUNN::IOperationDynamicConstraints::check_sparsity_rules(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const --> bool", pybind11::arg("config"), pybind11::arg("dpu"), pybind11::arg("info"));
		cl.def("assign", (class VPUNN::IOperationDynamicConstraints & (VPUNN::IOperationDynamicConstraints::*)(const class VPUNN::IOperationDynamicConstraints &)) &VPUNN::IOperationDynamicConstraints::operator=, "C++: VPUNN::IOperationDynamicConstraints::operator=(const class VPUNN::IOperationDynamicConstraints &) --> class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: VPUNN_11.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::IContainer_OperationsDynamicBehavior file: line:82
struct PyCallBack_VPUNN_IContainer_OperationsDynamicBehavior : public VPUNN::IContainer_OperationsDynamicBehavior {
	using VPUNN::IContainer_OperationsDynamicBehavior::IContainer_OperationsDynamicBehavior;

	const class VPUNN::IOperationDynamicConstraints & get_operation_specific_behaviour(const enum VPUNN::Operation a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IContainer_OperationsDynamicBehavior *>(this), "get_operation_specific_behaviour");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class VPUNN::IOperationDynamicConstraints &>::value) {
				static pybind11::detail::override_caster_t<const class VPUNN::IOperationDynamicConstraints &> caster;
				return pybind11::detail::cast_ref<const class VPUNN::IOperationDynamicConstraints &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class VPUNN::IOperationDynamicConstraints &>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour\"");
	}
};

// VPUNN::IDeviceValidValues file: line:68
struct PyCallBack_VPUNN_IDeviceValidValues : public VPUNN::IDeviceValidValues {
	using VPUNN::IDeviceValidValues::IDeviceValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IDeviceValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IDeviceValidValues::get_output_channels_range\"");
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IDeviceValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IDeviceValidValues::get_input_channels_range\"");
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IDeviceValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IDeviceValidValues::adapt_device_comaptible_tensor_layout\"");
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IDeviceValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IDeviceValidValues::adapt_device_comaptible_swizzling\"");
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IDeviceValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

void bind_VPUNN_11(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::IContainer_OperationsDynamicBehavior file: line:82
		pybind11::class_<VPUNN::IContainer_OperationsDynamicBehavior, VPUNN::IContainer_OperationsDynamicBehavior*, PyCallBack_VPUNN_IContainer_OperationsDynamicBehavior> cl(M("VPUNN"), "IContainer_OperationsDynamicBehavior", "interface to a container of IOperationDynamicConstraints associated 1-1 to operations");
		cl.def(pybind11::init<PyCallBack_VPUNN_IContainer_OperationsDynamicBehavior const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_IContainer_OperationsDynamicBehavior(); } ) );
		cl.def("get_operation_specific_behaviour", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::IContainer_OperationsDynamicBehavior::*)(const enum VPUNN::Operation) const) &VPUNN::IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour, "C++: VPUNN::IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("assign", (class VPUNN::IContainer_OperationsDynamicBehavior & (VPUNN::IContainer_OperationsDynamicBehavior::*)(const class VPUNN::IContainer_OperationsDynamicBehavior &)) &VPUNN::IContainer_OperationsDynamicBehavior::operator=, "C++: VPUNN::IContainer_OperationsDynamicBehavior::operator=(const class VPUNN::IContainer_OperationsDynamicBehavior &) --> class VPUNN::IContainer_OperationsDynamicBehavior &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::ValidValuesInfrastructure file: line:34
		pybind11::class_<VPUNN::ValidValuesInfrastructure, std::shared_ptr<VPUNN::ValidValuesInfrastructure>> cl(M("VPUNN"), "ValidValuesInfrastructure", "infrastructure class for describing and creating  valid values");
		cl.def( pybind11::init( [](VPUNN::ValidValuesInfrastructure const &o){ return new VPUNN::ValidValuesInfrastructure(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::ValidValuesInfrastructure(); } ) );
		cl.def("contains_value", (bool (VPUNN::ValidValuesInfrastructure::*)(const class std::vector<enum VPUNN::DataType, class std::allocator<enum VPUNN::DataType> > &, const enum VPUNN::DataType &) const) &VPUNN::ValidValuesInfrastructure::contains_value<VPUNN::DataType>, "C++: VPUNN::ValidValuesInfrastructure::contains_value(const class std::vector<enum VPUNN::DataType, class std::allocator<enum VPUNN::DataType> > &, const enum VPUNN::DataType &) const --> bool", pybind11::arg("container"), pybind11::arg("element"));
		cl.def("makeList", [](VPUNN::ValidValuesInfrastructure const &o, int const & a0, int const & a1) -> std::vector<int, class std::allocator<int> > { return o.makeList(a0, a1); }, "", pybind11::arg("from"), pybind11::arg("to"));
		cl.def("makeList", (class std::vector<int, class std::allocator<int> > (VPUNN::ValidValuesInfrastructure::*)(int, int, int) const) &VPUNN::ValidValuesInfrastructure::makeList, "creates a list , rule: [from ... to] * multiply\n\nC++: VPUNN::ValidValuesInfrastructure::makeList(int, int, int) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("from"), pybind11::arg("to"), pybind11::arg("multiply"));
		cl.def("assign", (class VPUNN::ValidValuesInfrastructure & (VPUNN::ValidValuesInfrastructure::*)(const class VPUNN::ValidValuesInfrastructure &)) &VPUNN::ValidValuesInfrastructure::operator=, "C++: VPUNN::ValidValuesInfrastructure::operator=(const class VPUNN::ValidValuesInfrastructure &) --> class VPUNN::ValidValuesInfrastructure &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::IDeviceValidValues file: line:68
		pybind11::class_<VPUNN::IDeviceValidValues, VPUNN::IDeviceValidValues*, PyCallBack_VPUNN_IDeviceValidValues, VPUNN::ValidValuesInfrastructure> cl(M("VPUNN"), "IDeviceValidValues", "interface for finding out what are the valid values for a workload that has a particular device and operation\n for stable values (independent of operation) : holds the data values that a workload can take on its fields\n dynamic behavior is provided via methods, including pure virtual ones.\n has also a connection to the specific behavior interface that discriminated between operations");
		cl.def(pybind11::init<PyCallBack_VPUNN_IDeviceValidValues const &>());
		cl.def_readwrite("valid_execution_order", &VPUNN::IDeviceValidValues::valid_execution_order);
		cl.def_readwrite("valid_swizzlings", &VPUNN::IDeviceValidValues::valid_swizzlings);
		cl.def_readwrite("default_swizzling", &VPUNN::IDeviceValidValues::default_swizzling);
		cl.def_readwrite("valid_layouts", &VPUNN::IDeviceValidValues::valid_layouts);
		cl.def_readwrite("devices", &VPUNN::IDeviceValidValues::devices);
		cl.def_readwrite("cmx_KB_sizes", &VPUNN::IDeviceValidValues::cmx_KB_sizes);
		cl.def_readonly("quantized_datatypes", &VPUNN::IDeviceValidValues::quantized_datatypes);
		cl.def_readonly("float_datatypes", &VPUNN::IDeviceValidValues::float_datatypes);
		cl.def_readonly("valid_datatypes", &VPUNN::IDeviceValidValues::valid_datatypes);
		cl.def_readonly("valid_operations", &VPUNN::IDeviceValidValues::valid_operations);
		cl.def_readonly("boolean_datatypes", &VPUNN::IDeviceValidValues::boolean_datatypes);
		cl.def_readwrite("output_write_tile_options", &VPUNN::IDeviceValidValues::output_write_tile_options);
		cl.def_readwrite("isi_stategy_options", &VPUNN::IDeviceValidValues::isi_stategy_options);
		cl.def_readwrite("input_heigth_start_factor_SOH", &VPUNN::IDeviceValidValues::input_heigth_start_factor_SOH);
		cl.def("get_specific_behaviour", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::IDeviceValidValues::*)(const enum VPUNN::Operation) const) &VPUNN::IDeviceValidValues::get_specific_behaviour, "wrapper for accessing IContainer_OperationsDynamicBehavior\n\nC++: VPUNN::IDeviceValidValues::get_specific_behaviour(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("get_valid_operations_range", (const class std::vector<enum VPUNN::Operation, class std::allocator<enum VPUNN::Operation> > & (VPUNN::IDeviceValidValues::*)() const) &VPUNN::IDeviceValidValues::get_valid_operations_range, "C++: VPUNN::IDeviceValidValues::get_valid_operations_range() const --> const class std::vector<enum VPUNN::Operation, class std::allocator<enum VPUNN::Operation> > &", pybind11::return_value_policy::automatic);
		cl.def("get_output_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_output_channels_range, "C++: VPUNN::IDeviceValidValues::get_output_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("get_input_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_input_channels_range, "C++: VPUNN::IDeviceValidValues::get_input_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("adapt_device_comaptible_tensor_layout", (enum VPUNN::Layout (VPUNN::IDeviceValidValues::*)(enum VPUNN::Layout) const) &VPUNN::IDeviceValidValues::adapt_device_comaptible_tensor_layout, "CHanges tensor layouts to match the device convention (if possible). Useful for defaults\n\nC++: VPUNN::IDeviceValidValues::adapt_device_comaptible_tensor_layout(enum VPUNN::Layout) const --> enum VPUNN::Layout", pybind11::arg("layout"));
		cl.def("adapt_device_comaptible_swizzling", (enum VPUNN::Swizzling (VPUNN::IDeviceValidValues::*)(enum VPUNN::Swizzling) const) &VPUNN::IDeviceValidValues::adapt_device_comaptible_swizzling, "Changes tensor swizzling to match the device special restrictions or conventions. Useful for defaults\n\nC++: VPUNN::IDeviceValidValues::adapt_device_comaptible_swizzling(enum VPUNN::Swizzling) const --> enum VPUNN::Swizzling", pybind11::arg("swizz"));
		cl.def("get_input_height_interval", (struct std::pair<int, int> (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_input_height_interval, "C++: VPUNN::IDeviceValidValues::get_input_height_interval(const struct VPUNN::DPUOperation &) const --> struct std::pair<int, int>", pybind11::arg("dpu"));
		cl.def("get_input_height_range", (class std::vector<int, class std::allocator<int> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_input_height_range, "C++: VPUNN::IDeviceValidValues::get_input_height_range(const struct VPUNN::DPUOperation &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("dpu"));
		cl.def("get_input_width_interval", (struct std::pair<int, int> (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_input_width_interval, "C++: VPUNN::IDeviceValidValues::get_input_width_interval(const struct VPUNN::DPUOperation &) const --> struct std::pair<int, int>", pybind11::arg("dpu"));
		cl.def("get_input_width_range", (class std::vector<int, class std::allocator<int> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_input_width_range, "C++: VPUNN::IDeviceValidValues::get_input_width_range(const struct VPUNN::DPUOperation &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("dpu"));
		cl.def("get_pad_horz_range", (class std::vector<int, class std::allocator<int> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_pad_horz_range, "C++: VPUNN::IDeviceValidValues::get_pad_horz_range(const struct VPUNN::DPUOperation &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("dpu"));
		cl.def("get_pad_vert_range", (class std::vector<int, class std::allocator<int> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_pad_vert_range, "C++: VPUNN::IDeviceValidValues::get_pad_vert_range(const struct VPUNN::DPUOperation &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("dpu"));
		cl.def("get_ISI_Strategy_Range", (class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_ISI_Strategy_Range, "restricts ISI options based on output write tile value.\n\nC++: VPUNN::IDeviceValidValues::get_ISI_Strategy_Range(const struct VPUNN::DPUOperation &) const --> class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >", pybind11::arg("dpu"));
		cl.def("get_output_write_tile_Range", (class std::vector<int, class std::allocator<int> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_output_write_tile_Range, "restricts output_write_tile options based on operation.\n\nC++: VPUNN::IDeviceValidValues::get_output_write_tile_Range(const struct VPUNN::DPUOperation &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("dpu"));
		cl.def("get_kernel_range", (class std::vector<int, class std::allocator<int> > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_kernel_range, "C++: VPUNN::IDeviceValidValues::get_kernel_range(const struct VPUNN::DPUOperation &) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("dpu"));
		cl.def("get_strides_range", (struct std::pair<class std::vector<int, class std::allocator<int> >, class std::vector<int, class std::allocator<int> > > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_strides_range, "@ brief strides range , depends on input zero, and operation sometimes\n \n\n a pair of lists of values, one first for  stride Width, second for stride height\n\nC++: VPUNN::IDeviceValidValues::get_strides_range(const struct VPUNN::DPUOperation &) const --> struct std::pair<class std::vector<int, class std::allocator<int> >, class std::vector<int, class std::allocator<int> > >", pybind11::arg("dpu"));
		cl.def("get_dpu_strides_range", (struct std::pair<class std::vector<int, class std::allocator<int> >, class std::vector<int, class std::allocator<int> > > (VPUNN::IDeviceValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::IDeviceValidValues::get_dpu_strides_range, "@ brief this function is set to determine the range of strides for the split layers\n \n\n a pair of lists of values, one first for  stride Width, second for stride height\n\nC++: VPUNN::IDeviceValidValues::get_dpu_strides_range(const struct VPUNN::DPUOperation &) const --> struct std::pair<class std::vector<int, class std::allocator<int> >, class std::vector<int, class std::allocator<int> > >", pybind11::arg("dpu"));
		cl.def("restrict_datatype", (enum VPUNN::DataType (VPUNN::IDeviceValidValues::*)(const enum VPUNN::DataType) const) &VPUNN::IDeviceValidValues::restrict_datatype, "restrict the datatype normally to one per int range and one per float range\n\nC++: VPUNN::IDeviceValidValues::restrict_datatype(const enum VPUNN::DataType) const --> enum VPUNN::DataType", pybind11::arg("in"));
		cl.def("get_cmx_size", (int (VPUNN::IDeviceValidValues::*)(const enum VPUNN::VPUDevice &) const) &VPUNN::IDeviceValidValues::get_cmx_size, "size of CMX in bytes\n\nC++: VPUNN::IDeviceValidValues::get_cmx_size(const enum VPUNN::VPUDevice &) const --> int", pybind11::arg("device"));
		cl.def("get_padMax", (int (VPUNN::IDeviceValidValues::*)(int) const) &VPUNN::IDeviceValidValues::get_padMax, "maximum padding required if the kernel is known\n\nC++: VPUNN::IDeviceValidValues::get_padMax(int) const --> int", pybind11::arg("kernel_dim"));
		cl.def("compute_output_dim", (int (VPUNN::IDeviceValidValues::*)(int, int, int, int, int) const) &VPUNN::IDeviceValidValues::compute_output_dim, "@ computes output dimension based on input, kernel, padding and stride\n\nC++: VPUNN::IDeviceValidValues::compute_output_dim(int, int, int, int, int) const --> int", pybind11::arg("input"), pybind11::arg("pad"), pybind11::arg("pad_oppopsite"), pybind11::arg("kernel"), pybind11::arg("kernel_stride"));
		cl.def("align_to", (long long (VPUNN::IDeviceValidValues::*)(long long, int) const) &VPUNN::IDeviceValidValues::align_to, "computes the next larger value of x that is multiple of multiple\n \n\n only for positive values\n\nC++: VPUNN::IDeviceValidValues::align_to(long long, int) const --> long long", pybind11::arg("x"), pybind11::arg("multiple"));
		cl.def("sanitize_sparsity", (float (VPUNN::IDeviceValidValues::*)(long long, float) const) &VPUNN::IDeviceValidValues::sanitize_sparsity, "sparsity is applied in blocks (16 B normally) , the desired value has to be quantized taken that in\n consideration\n\nC++: VPUNN::IDeviceValidValues::sanitize_sparsity(long long, float) const --> float", pybind11::arg("tensor_size"), pybind11::arg("desired_sparsity_level"));
		cl.def("check_trailing_padding", (int (VPUNN::IDeviceValidValues::*)(int, int, int, int, int) const) &VPUNN::IDeviceValidValues::check_trailing_padding, "Adapt  padding\n Reference :\n https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html#zero-padding-non-unit-strides\n\nC++: VPUNN::IDeviceValidValues::check_trailing_padding(int, int, int, int, int) const --> int", pybind11::arg("in_dim"), pybind11::arg("out_dim"), pybind11::arg("leading_pad"), pybind11::arg("kernel_radix"), pybind11::arg("stride"));
		cl.def("compute_size_aligned", (long long (VPUNN::IDeviceValidValues::*)(const long long, const enum VPUNN::DataType &) const) &VPUNN::IDeviceValidValues::compute_size_aligned, "computes size with alignment\n\nC++: VPUNN::IDeviceValidValues::compute_size_aligned(const long long, const enum VPUNN::DataType &) const --> long long", pybind11::arg("elements_count"), pybind11::arg("datatype"));
		cl.def("compute_size_raw", (long long (VPUNN::IDeviceValidValues::*)(const long long, const enum VPUNN::DataType &) const) &VPUNN::IDeviceValidValues::compute_size_raw, "computes size without alignment\n\nC++: VPUNN::IDeviceValidValues::compute_size_raw(const long long, const enum VPUNN::DataType &) const --> long long", pybind11::arg("elements_count"), pybind11::arg("datatype"));
	}
}


// File: VPUNN_12.cpp
#include <array> // std::array
#include <functional> // std::less
#include <iterator> // __gnu_cxx::__normal_iterator
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_12(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::VPUTilingStrategy file: line:22
	pybind11::enum_<VPUNN::VPUTilingStrategy>(M("VPUNN"), "VPUTilingStrategy", "VPU tiling strategy. How to split a Layer on multiple tiles")
		.value("NONE", VPUNN::VPUTilingStrategy::NONE)
		.value("SOH", VPUNN::VPUTilingStrategy::SOH)
		.value("SOK", VPUNN::VPUTilingStrategy::SOK)
		.value("SOW", VPUNN::VPUTilingStrategy::SOW)
		.value("SOHW", VPUNN::VPUTilingStrategy::SOHW)
		.value("SOHK", VPUNN::VPUTilingStrategy::SOHK)
		.value("__size", VPUNN::VPUTilingStrategy::__size);

;

	// VPUNN::mapToText() file: line:29
	M("VPUNN").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::mapToText<VPUNN::VPUTilingStrategy>, "C++: VPUNN::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	{ // VPUNN::DPULayer file: line:34
		pybind11::class_<VPUNN::DPULayer, std::shared_ptr<VPUNN::DPULayer>, VPUNN::DPUWorkload> cl(M("VPUNN"), "DPULayer", "DPULayer class. no data  only methods on top of DPUWorkload");
		cl.def( pybind11::init<enum VPUNN::VPUDevice, enum VPUNN::Operation, struct std::array<class VPUNN::VPUTensor, 1>, struct std::array<class VPUNN::VPUTensor, 1>, struct std::array<unsigned int, 2>, struct std::array<unsigned int, 2>, struct std::array<unsigned int, 4>>(), pybind11::arg("device"), pybind11::arg("op"), pybind11::arg("inputs"), pybind11::arg("outputs"), pybind11::arg("kernels"), pybind11::arg("strides"), pybind11::arg("padding") );

		cl.def( pybind11::init<const struct VPUNN::DPUWorkload &>(), pybind11::arg("wl") );

		cl.def( pybind11::init( [](VPUNN::DPULayer const &o){ return new VPUNN::DPULayer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DPULayer(); } ) );
		cl.def("clustering", (class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > (VPUNN::DPULayer::*)(unsigned int) const) &VPUNN::DPULayer::clustering, "Implements the Clustering tiling strategy (inter tile)\n \n\n In the clustering tiling strategy, both activations and weights are fully replicated in all tiles\n isi_strategy set to clustering\n and output_write_tiles are propagated from input\n\n \n number of dpu tiles\n \n\n std::vector<DPULayer> the list of layers\n\nC++: VPUNN::DPULayer::clustering(unsigned int) const --> class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> >", pybind11::arg("nTiles"));
		cl.def("SOH", (class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > (VPUNN::DPULayer::*)(unsigned int) const) &VPUNN::DPULayer::SOH, "Implements the SplitOverH (SOH) tiling strategy (inter tile)\n \n\n In the SOH tiling strategy, activations are split across the tiles over the H dimension\n todo: Cut lines must be without padding\n todo: compute tensors have a halo where the cut is to indicate how many lines are taken from the other tile\n The weights are fully replicated in all tiles\n Populates also ISI strategy with SOH\n output_write_tiles is propagated from inputLayer  to the tiles Layers\n\nif output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted\n\n \n number of CMX tiles\n \n\n std::vector<DPULayer>  the list of split layers. can be smaller than nTiles\n\nC++: VPUNN::DPULayer::SOH(unsigned int) const --> class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> >", pybind11::arg("nTiles"));
		cl.def("SOH_overlapped", (class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > (VPUNN::DPULayer::*)(unsigned int) const) &VPUNN::DPULayer::SOH_overlapped, "Implements the SplitOverH_OVERLAPPED (SOH) tiling strategy (inter tile)\n \n\n In the SOHOVERLAPPED tiling strategy, activations are split across the tiles over the H dimension\n Compute tensors are the same as memory tensors (no halo at cut lines)\n no padding at cut lines\n Internal slices (tiles>2), have 2 borders and may produce more than outside slices.\n Populates also ISI strategy with CLUSTERING,\n output_write_tiles is propagated from inputLayer  to the tiles Layers\n\n if output tiles are less than nTiles , the ISI strategy or output_write_tiles are not adjusted\n\n \n number of CMX tiles\n \n\n std::vector<DPULayer>  the list of split layers. can be smaller than nTiles\n\nC++: VPUNN::DPULayer::SOH_overlapped(unsigned int) const --> class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> >", pybind11::arg("nTiles"));
		cl.def("SOK", [](VPUNN::DPULayer const &o, unsigned int const & a0) -> std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > { return o.SOK(a0); }, "", pybind11::arg("nTiles"));
		cl.def("SOK", (class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > (VPUNN::DPULayer::*)(unsigned int, unsigned int) const) &VPUNN::DPULayer::SOK, "Implements the SplitOverK (SOK) tiling strategy\n \n\n In the SOK tiling strategy, weights are split across the tiles over the K dimension.\n The DPU in each tile compute a K-slice of the output tensors and then broadcast the result in each\n CMX tile, implicitly concatenating the results and having then all activations completely replicated\n\n Populates also ISI strategy with SOK\n output_write_tiles is set to actual nTiles\n if output tiles are less than nTiles output_write_tiles are adjusted to actual output tiles\n\n \n number of CMX tiles\n \n\n the channel alignment\n \n\n std::vector<DPULayer>\n\nC++: VPUNN::DPULayer::SOK(unsigned int, unsigned int) const --> class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> >", pybind11::arg("nTiles"), pybind11::arg("rounding"));
		cl.def("splitAcrossTiles", [](VPUNN::DPULayer const &o, enum VPUNN::VPUTilingStrategy const & a0) -> std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > { return o.splitAcrossTiles(a0); }, "", pybind11::arg("strategy"));
		cl.def("splitAcrossTiles", (class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > (VPUNN::DPULayer::*)(enum VPUNN::VPUTilingStrategy, unsigned int) const) &VPUNN::DPULayer::splitAcrossTiles, "Split a DPULayer across N CMX tiles\n\n \n the VPUTilingStrategy to implement\n \n\n number of CMX tiles\n \n\n std::vector<DPULayer>\n\nC++: VPUNN::DPULayer::splitAcrossTiles(enum VPUNN::VPUTilingStrategy, unsigned int) const --> class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> >", pybind11::arg("strategy"), pybind11::arg("nTiles"));
		cl.def_static("mapTilingStrategiesToWorkload", (enum VPUNN::ISIStrategy (*)(enum VPUNN::VPUTilingStrategy)) &VPUNN::DPULayer::mapTilingStrategiesToWorkload, "C++: VPUNN::DPULayer::mapTilingStrategiesToWorkload(enum VPUNN::VPUTilingStrategy) --> enum VPUNN::ISIStrategy", pybind11::arg("strategy"));
		cl.def("input_footprint", (unsigned int (VPUNN::DPULayer::*)() const) &VPUNN::DPULayer::input_footprint, "The memory footprint of the input tensors\n\n \n unsigned int\n\nC++: VPUNN::DPULayer::input_footprint() const --> unsigned int");
		cl.def("output_footprint", (unsigned int (VPUNN::DPULayer::*)() const) &VPUNN::DPULayer::output_footprint, "The memory footprint of the output tensors\n\n \n unsigned int\n\nC++: VPUNN::DPULayer::output_footprint() const --> unsigned int");
		cl.def("weight_footprint", (unsigned int (VPUNN::DPULayer::*)(const class VPUNN::IDeviceValidValues &) const) &VPUNN::DPULayer::weight_footprint, "The memory footprint of the weights, without end cmx alignment(16Kb normally)\n \n\n: this might be wrong, review\n\n \n a good configuration(rules & behaviors) according to the device of the layer.\n \n\n unsigned int bytes\n\nC++: VPUNN::DPULayer::weight_footprint(const class VPUNN::IDeviceValidValues &) const --> unsigned int", pybind11::arg("config"));
		cl.def("footprint", (unsigned int (VPUNN::DPULayer::*)(const class VPUNN::IDeviceValidValues &) const) &VPUNN::DPULayer::footprint, "Layer total memory footprint\n\n \n unsigned int\n\nC++: VPUNN::DPULayer::footprint(const class VPUNN::IDeviceValidValues &) const --> unsigned int", pybind11::arg("config"));
		cl.def("recomputeInputTensorShape", (void (VPUNN::DPULayer::*)()) &VPUNN::DPULayer::recomputeInputTensorShape, "recomputes the input based on operation and a changed output size\n affects only WHC dimensions based on Outputs kernels, stride, padding\n\nC++: VPUNN::DPULayer::recomputeInputTensorShape() --> void");
		cl.def("set_weight_sparsity", (void (VPUNN::DPULayer::*)(bool, float)) &VPUNN::DPULayer::set_weight_sparsity, "enables /disables the w sparsity and sets the value. Only combinations that are allowed\n when enabling sparsity_value is limited to [0.0, 1.0]\n when disabling , sparsity will be set to zero\n\n \n true/false\n \n\n , limited to [0.0, 1.0] for enabled and to 0.0 for disabled\n\nC++: VPUNN::DPULayer::set_weight_sparsity(bool, float) --> void", pybind11::arg("enabled"), pybind11::arg("sparsity_value"));
		cl.def("assign", (struct VPUNN::DPULayer & (VPUNN::DPULayer::*)(const struct VPUNN::DPULayer &)) &VPUNN::DPULayer::operator=, "C++: VPUNN::DPULayer::operator=(const struct VPUNN::DPULayer &) --> struct VPUNN::DPULayer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::DMA_CyclesInfo file: line:497
		pybind11::class_<VPUNN::DMA_CyclesInfo, std::shared_ptr<VPUNN::DMA_CyclesInfo>> cl(M("VPUNN"), "DMA_CyclesInfo", "");
		cl.def( pybind11::init( [](VPUNN::DMA_CyclesInfo const &o){ return new VPUNN::DMA_CyclesInfo(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DMA_CyclesInfo(); } ) );
		cl.def_readwrite("cycles", &VPUNN::DMA_CyclesInfo::cycles);
		cl.def("assign", (struct VPUNN::DMA_CyclesInfo & (VPUNN::DMA_CyclesInfo::*)(const struct VPUNN::DMA_CyclesInfo &)) &VPUNN::DMA_CyclesInfo::operator=, "C++: VPUNN::DMA_CyclesInfo::operator=(const struct VPUNN::DMA_CyclesInfo &) --> struct VPUNN::DMA_CyclesInfo &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::DMALayerInfo file: line:501
		pybind11::class_<VPUNN::DMALayerInfo, std::shared_ptr<VPUNN::DMALayerInfo>> cl(M("VPUNN"), "DMALayerInfo", "");
		cl.def( pybind11::init( [](VPUNN::DMALayerInfo const &o){ return new VPUNN::DMALayerInfo(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DMALayerInfo(); } ) );
		cl.def_readwrite("w_tensor", &VPUNN::DMALayerInfo::w_tensor);
		cl.def_readwrite("input_tensor", &VPUNN::DMALayerInfo::input_tensor);
		cl.def_readwrite("output_tensor", &VPUNN::DMALayerInfo::output_tensor);
		cl.def("assign", (struct VPUNN::DMALayerInfo & (VPUNN::DMALayerInfo::*)(const struct VPUNN::DMALayerInfo &)) &VPUNN::DMALayerInfo::operator=, "C++: VPUNN::DMALayerInfo::operator=(const struct VPUNN::DMALayerInfo &) --> struct VPUNN::DMALayerInfo &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::OneTileLayerInfo file: line:516
		pybind11::class_<VPUNN::OneTileLayerInfo, std::shared_ptr<VPUNN::OneTileLayerInfo>> cl(M("VPUNN"), "OneTileLayerInfo", "details about a tile split strategy");
		cl.def( pybind11::init( [](VPUNN::OneTileLayerInfo const &o){ return new VPUNN::OneTileLayerInfo(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::OneTileLayerInfo(); } ) );
		cl.def_readwrite("inter_tile_split_layer", &VPUNN::OneTileLayerInfo::inter_tile_split_layer);
		cl.def_readwrite("best_intra_tile_split", &VPUNN::OneTileLayerInfo::best_intra_tile_split);
		cl.def_readwrite("DMA_info", &VPUNN::OneTileLayerInfo::DMA_info);
		cl.def("assign", (struct VPUNN::OneTileLayerInfo & (VPUNN::OneTileLayerInfo::*)(const struct VPUNN::OneTileLayerInfo &)) &VPUNN::OneTileLayerInfo::operator=, "C++: VPUNN::OneTileLayerInfo::operator=(const struct VPUNN::OneTileLayerInfo &) --> struct VPUNN::OneTileLayerInfo &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::DPULayerModes file: line:530
		pybind11::class_<VPUNN::DPULayerModes, std::shared_ptr<VPUNN::DPULayerModes>> cl(M("VPUNN"), "DPULayerModes", "provides differentiated information for a layer based on its content");
		cl.def_static("getValidExecutionMode", (class std::vector<enum VPUNN::ExecutionMode, class std::allocator<enum VPUNN::ExecutionMode> > (*)(const struct VPUNN::DPULayer &)) &VPUNN::DPULayerModes::getValidExecutionMode, "Get the valid ExecutionMode for the DPULayer\n\n \n the DPULayer\n \n\n std::vector<ExecutionMode>\n\nC++: VPUNN::DPULayerModes::getValidExecutionMode(const struct VPUNN::DPULayer &) --> class std::vector<enum VPUNN::ExecutionMode, class std::allocator<enum VPUNN::ExecutionMode> >", pybind11::arg("wl"));
	}
}


// File: VPUNN_13.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::Behaviours file: line:29
struct PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_t : public VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints> {
	using VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::Behaviours;

	const class VPUNN::IOperationDynamicConstraints & get_operation_specific_behaviour(const enum VPUNN::Operation a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints> *>(this), "get_operation_specific_behaviour");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class VPUNN::IOperationDynamicConstraints &>::value) {
				static pybind11::detail::override_caster_t<const class VPUNN::IOperationDynamicConstraints &> caster;
				return pybind11::detail::cast_ref<const class VPUNN::IOperationDynamicConstraints &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class VPUNN::IOperationDynamicConstraints &>(std::move(o));
		}
		return Behaviours::get_operation_specific_behaviour(a0);
	}
};

// VPUNN::Behaviours file: line:29
struct PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_t : public VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer> {
	using VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::Behaviours;

	const class VPUNN::IOperationDynamicConstraints & get_operation_specific_behaviour(const enum VPUNN::Operation a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer> *>(this), "get_operation_specific_behaviour");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class VPUNN::IOperationDynamicConstraints &>::value) {
				static pybind11::detail::override_caster_t<const class VPUNN::IOperationDynamicConstraints &> caster;
				return pybind11::detail::cast_ref<const class VPUNN::IOperationDynamicConstraints &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class VPUNN::IOperationDynamicConstraints &>(std::move(o));
		}
		return Behaviours::get_operation_specific_behaviour(a0);
	}
};

void bind_VPUNN_13(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::LRUCache file: line:26
		pybind11::class_<VPUNN::LRUCache<float>, std::shared_ptr<VPUNN::LRUCache<float>>> cl(M("VPUNN"), "LRUCache_float_t", "");
		cl.def( pybind11::init<unsigned long>(), pybind11::arg("max_size") );

		cl.def( pybind11::init( [](VPUNN::LRUCache<float> const &o){ return new VPUNN::LRUCache<float>(o); } ) );
		cl.def("add", (void (VPUNN::LRUCache<float>::*)(const class std::vector<float, class std::allocator<float> > &, const float &)) &VPUNN::LRUCache<float>::add, "C++: VPUNN::LRUCache<float>::add(const class std::vector<float, class std::allocator<float> > &, const float &) --> void", pybind11::arg("wl"), pybind11::arg("value"));
		cl.def("get", (float * (VPUNN::LRUCache<float>::*)(const class std::vector<float, class std::allocator<float> > &)) &VPUNN::LRUCache<float>::get, "C++: VPUNN::LRUCache<float>::get(const class std::vector<float, class std::allocator<float> > &) --> float *", pybind11::return_value_policy::automatic, pybind11::arg("wl"));
	}
	{ // VPUNN::Behaviours file: line:29
		pybind11::class_<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, std::shared_ptr<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>>, PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_t, VPUNN::IContainer_OperationsDynamicBehavior> cl(M("VPUNN"), "Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_t", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_t const &o){ return new PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints> const &o){ return new VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>(); }, [](){ return new PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_t(); } ) );
		cl.def("get_operation_specific_", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::*)(const enum VPUNN::Operation) const) &VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::get_operation_specific_<VPUNN::IOperationDynamicConstraints>, "C++: VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::get_operation_specific_(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("get_operation_specific_behaviour", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::*)(const enum VPUNN::Operation) const) &VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::get_operation_specific_behaviour, "C++: VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::get_operation_specific_behaviour(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("assign", (class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints, class VPUNN::DW_CONVOLUTION_Constraints, class VPUNN::CM_CONVOLUTION_Constraints, class VPUNN::ELTWISE_Constraints, class VPUNN::MAXPOOL_Constraints> & (VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::*)(const class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints, class VPUNN::DW_CONVOLUTION_Constraints, class VPUNN::CM_CONVOLUTION_Constraints, class VPUNN::ELTWISE_Constraints, class VPUNN::MAXPOOL_Constraints> &)) &VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::operator=, "C++: VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>::operator=(const class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints, class VPUNN::DW_CONVOLUTION_Constraints, class VPUNN::CM_CONVOLUTION_Constraints, class VPUNN::ELTWISE_Constraints, class VPUNN::MAXPOOL_Constraints> &) --> class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints, class VPUNN::DW_CONVOLUTION_Constraints, class VPUNN::CM_CONVOLUTION_Constraints, class VPUNN::ELTWISE_Constraints, class VPUNN::MAXPOOL_Constraints> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("get_operation_specific_behaviour", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::IContainer_OperationsDynamicBehavior::*)(const enum VPUNN::Operation) const) &VPUNN::IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour, "C++: VPUNN::IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("assign", (class VPUNN::IContainer_OperationsDynamicBehavior & (VPUNN::IContainer_OperationsDynamicBehavior::*)(const class VPUNN::IContainer_OperationsDynamicBehavior &)) &VPUNN::IContainer_OperationsDynamicBehavior::operator=, "C++: VPUNN::IContainer_OperationsDynamicBehavior::operator=(const class VPUNN::IContainer_OperationsDynamicBehavior &) --> class VPUNN::IContainer_OperationsDynamicBehavior &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::Behaviours file: line:29
		pybind11::class_<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>, std::shared_ptr<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>>, PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_t, VPUNN::IContainer_OperationsDynamicBehavior> cl(M("VPUNN"), "Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_t", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_t const &o){ return new PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer> const &o){ return new VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>(); }, [](){ return new PyCallBack_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_t(); } ) );
		cl.def("get_operation_specific_", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::*)(const enum VPUNN::Operation) const) &VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::get_operation_specific_<VPUNN::IOperationDynamicConstraints>, "C++: VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::get_operation_specific_(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("get_operation_specific_behaviour", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::*)(const enum VPUNN::Operation) const) &VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::get_operation_specific_behaviour, "C++: VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::get_operation_specific_behaviour(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("assign", (class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints_Layer, class VPUNN::DW_CONVOLUTION_Constraints_Layer, class VPUNN::CM_CONVOLUTION_Constraints_Layer, class VPUNN::ELTWISE_Constraints_Layer, class VPUNN::MAXPOOL_Constraints_Layer> & (VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::*)(const class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints_Layer, class VPUNN::DW_CONVOLUTION_Constraints_Layer, class VPUNN::CM_CONVOLUTION_Constraints_Layer, class VPUNN::ELTWISE_Constraints_Layer, class VPUNN::MAXPOOL_Constraints_Layer> &)) &VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::operator=, "C++: VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>::operator=(const class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints_Layer, class VPUNN::DW_CONVOLUTION_Constraints_Layer, class VPUNN::CM_CONVOLUTION_Constraints_Layer, class VPUNN::ELTWISE_Constraints_Layer, class VPUNN::MAXPOOL_Constraints_Layer> &) --> class VPUNN::Behaviours<class VPUNN::CONVOLUTION_Constraints_Layer, class VPUNN::DW_CONVOLUTION_Constraints_Layer, class VPUNN::CM_CONVOLUTION_Constraints_Layer, class VPUNN::ELTWISE_Constraints_Layer, class VPUNN::MAXPOOL_Constraints_Layer> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("get_operation_specific_behaviour", (const class VPUNN::IOperationDynamicConstraints & (VPUNN::IContainer_OperationsDynamicBehavior::*)(const enum VPUNN::Operation) const) &VPUNN::IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour, "C++: VPUNN::IContainer_OperationsDynamicBehavior::get_operation_specific_behaviour(const enum VPUNN::Operation) const --> const class VPUNN::IOperationDynamicConstraints &", pybind11::return_value_policy::automatic, pybind11::arg("op"));
		cl.def("assign", (class VPUNN::IContainer_OperationsDynamicBehavior & (VPUNN::IContainer_OperationsDynamicBehavior::*)(const class VPUNN::IContainer_OperationsDynamicBehavior &)) &VPUNN::IContainer_OperationsDynamicBehavior::operator=, "C++: VPUNN::IContainer_OperationsDynamicBehavior::operator=(const class VPUNN::IContainer_OperationsDynamicBehavior &) --> class VPUNN::IContainer_OperationsDynamicBehavior &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::Behavior_Device_Mapping file: line:98
		pybind11::class_<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>, std::shared_ptr<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>> cl(M("VPUNN"), "Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_VPUNN_VPU2_0_WorkloadValidValues_VPUNN_VPU2_7_WorkloadValidValues_VPUNN_VPU4_0_WorkloadValidValues_t", "");
		cl.def( pybind11::init( [](VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> const &o){ return new VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>(); } ) );
		cl.def("is_supported", (bool (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::is_supported, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::is_supported(enum VPUNN::VPUDevice) const --> bool", pybind11::arg("device"));
		cl.def("get_config", (const class VPUNN::IDeviceValidValues & (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::get_config, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::get_config(enum VPUNN::VPUDevice) const --> const class VPUNN::IDeviceValidValues &", pybind11::return_value_policy::automatic, pybind11::arg("device"));
	}
}


// File: VPUNN_14.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::VPU4_0_WorkloadValidValues file: line:36
struct PyCallBack_VPUNN_VPU4_0_WorkloadValidValues : public VPUNN::VPU4_0_WorkloadValidValues {
	using VPUNN::VPU4_0_WorkloadValidValues::VPU4_0_WorkloadValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_WorkloadValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_WorkloadValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_WorkloadValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_WorkloadValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_WorkloadValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

// VPUNN::VPU4_0_LayerValidValues file: line:110
struct PyCallBack_VPUNN_VPU4_0_LayerValidValues : public VPUNN::VPU4_0_LayerValidValues {
	using VPUNN::VPU4_0_LayerValidValues::VPU4_0_LayerValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU4_0_LayerValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU4_0_LayerValidValues::get_input_channels_range(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return VPU4_0_LayerValidValues::get_ISI_Strategy_Range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
};

// VPUNN::VPU4_0_LayerOnTileValidValues file: line:164
struct PyCallBack_VPUNN_VPU4_0_LayerOnTileValidValues : public VPUNN::VPU4_0_LayerOnTileValidValues {
	using VPUNN::VPU4_0_LayerOnTileValidValues::VPU4_0_LayerOnTileValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerOnTileValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerOnTileValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerOnTileValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerOnTileValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU4_0_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU4_0_LayerOnTileValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

// VPUNN::VPU2_0_WorkloadValidValues file: line:37
struct PyCallBack_VPUNN_VPU2_0_WorkloadValidValues : public VPUNN::VPU2_0_WorkloadValidValues {
	using VPUNN::VPU2_0_WorkloadValidValues::VPU2_0_WorkloadValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_WorkloadValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_WorkloadValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_WorkloadValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_WorkloadValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_WorkloadValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

// VPUNN::VPU2_7_WorkloadValidValues file: line:103
struct PyCallBack_VPUNN_VPU2_7_WorkloadValidValues : public VPUNN::VPU2_7_WorkloadValidValues {
	using VPUNN::VPU2_7_WorkloadValidValues::VPU2_7_WorkloadValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_WorkloadValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_WorkloadValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_WorkloadValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_WorkloadValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_WorkloadValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

void bind_VPUNN_14(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::Behavior_Device_Mapping file: line:98
		pybind11::class_<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>, std::shared_ptr<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>>> cl(M("VPUNN"), "Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_Layer_VPUNN_DW_CONVOLUTION_Constraints_Layer_VPUNN_CM_CONVOLUTION_Constraints_Layer_VPUNN_ELTWISE_Constraints_Layer_VPUNN_MAXPOOL_Constraints_Layer_VPUNN_VPU2_0_LayerValidValues_VPUNN_VPU2_7_LayerValidValues_VPUNN_VPU4_0_LayerValidValues_t", "");
		cl.def( pybind11::init( [](VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues> const &o){ return new VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>(); } ) );
		cl.def("is_supported", (bool (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>, VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>::is_supported, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>, VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>::is_supported(enum VPUNN::VPUDevice) const --> bool", pybind11::arg("device"));
		cl.def("get_config", (const class VPUNN::IDeviceValidValues & (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>, VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>::get_config, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>, VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>::get_config(enum VPUNN::VPUDevice) const --> const class VPUNN::IDeviceValidValues &", pybind11::return_value_policy::automatic, pybind11::arg("device"));
	}
	{ // VPUNN::Behavior_Device_Mapping file: line:98
		pybind11::class_<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>, std::shared_ptr<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>> cl(M("VPUNN"), "Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_VPUNN_VPU2_0_LayerOnTileValidValues_VPUNN_VPU2_7_LayerOnTileValidValues_VPUNN_VPU4_0_LayerOnTileValidValues_t", "");
		cl.def( pybind11::init( [](VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> const &o){ return new VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>(); } ) );
		cl.def("is_supported", (bool (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::is_supported, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::is_supported(enum VPUNN::VPUDevice) const --> bool", pybind11::arg("device"));
		cl.def("get_config", (const class VPUNN::IDeviceValidValues & (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::get_config, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::get_config(enum VPUNN::VPUDevice) const --> const class VPUNN::IDeviceValidValues &", pybind11::return_value_policy::automatic, pybind11::arg("device"));
	}
	{ // VPUNN::VPU4_0_WorkloadValidValues file: line:36
		pybind11::class_<VPUNN::VPU4_0_WorkloadValidValues, std::shared_ptr<VPUNN::VPU4_0_WorkloadValidValues>, PyCallBack_VPUNN_VPU4_0_WorkloadValidValues, VPUNN::IDeviceValidValues> cl(M("VPUNN"), "VPU4_0_WorkloadValidValues", "///////////////////// VPU 4.0  all\n \n\n specific VPU 4.0 configuration possibilities for workload, not layer");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_constraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU4_0_WorkloadValidValues const &o){ return new PyCallBack_VPUNN_VPU4_0_WorkloadValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU4_0_WorkloadValidValues const &o){ return new VPUNN::VPU4_0_WorkloadValidValues(o); } ) );
		cl.def("get_output_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU4_0_WorkloadValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU4_0_WorkloadValidValues::get_output_channels_range, "C++: VPUNN::VPU4_0_WorkloadValidValues::get_output_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("get_input_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU4_0_WorkloadValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU4_0_WorkloadValidValues::get_input_channels_range, "C++: VPUNN::VPU4_0_WorkloadValidValues::get_input_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("adapt_device_comaptible_tensor_layout", (enum VPUNN::Layout (VPUNN::VPU4_0_WorkloadValidValues::*)(enum VPUNN::Layout) const) &VPUNN::VPU4_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout, "C++: VPUNN::VPU4_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(enum VPUNN::Layout) const --> enum VPUNN::Layout", pybind11::arg("layout"));
		cl.def("adapt_device_comaptible_swizzling", (enum VPUNN::Swizzling (VPUNN::VPU4_0_WorkloadValidValues::*)(enum VPUNN::Swizzling) const) &VPUNN::VPU4_0_WorkloadValidValues::adapt_device_comaptible_swizzling, "C++: VPUNN::VPU4_0_WorkloadValidValues::adapt_device_comaptible_swizzling(enum VPUNN::Swizzling) const --> enum VPUNN::Swizzling", pybind11::arg("swizz"));
	}
	{ // VPUNN::VPU4_0_LayerValidValues file: line:110
		pybind11::class_<VPUNN::VPU4_0_LayerValidValues, std::shared_ptr<VPUNN::VPU4_0_LayerValidValues>, PyCallBack_VPUNN_VPU4_0_LayerValidValues, VPUNN::VPU4_0_WorkloadValidValues> cl(M("VPUNN"), "VPU4_0_LayerValidValues", "///// LAYER UNSPLIT situation\n \n\n specific VPU 4.0 configuration possibilities for  layer");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_constraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU4_0_LayerValidValues const &o){ return new PyCallBack_VPUNN_VPU4_0_LayerValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU4_0_LayerValidValues const &o){ return new VPUNN::VPU4_0_LayerValidValues(o); } ) );
		cl.def("get_output_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU4_0_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU4_0_LayerValidValues::get_output_channels_range, "C++: VPUNN::VPU4_0_LayerValidValues::get_output_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("get_input_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU4_0_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU4_0_LayerValidValues::get_input_channels_range, "C++: VPUNN::VPU4_0_LayerValidValues::get_input_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("get_ISI_Strategy_Range", (class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > (VPUNN::VPU4_0_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU4_0_LayerValidValues::get_ISI_Strategy_Range, "C++: VPUNN::VPU4_0_LayerValidValues::get_ISI_Strategy_Range(const struct VPUNN::DPUOperation &) const --> class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >", pybind11::arg("dpu"));
	}
	{ // VPUNN::VPU4_0_LayerOnTileValidValues file: line:164
		pybind11::class_<VPUNN::VPU4_0_LayerOnTileValidValues, std::shared_ptr<VPUNN::VPU4_0_LayerOnTileValidValues>, PyCallBack_VPUNN_VPU4_0_LayerOnTileValidValues, VPUNN::VPU4_0_WorkloadValidValues> cl(M("VPUNN"), "VPU4_0_LayerOnTileValidValues", "specific VPU 4.0 configuration possibilities for  layer already split on tile.\n channels restrictions are less strict vs workload, since a further split is expected");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_constraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU4_0_LayerOnTileValidValues const &o){ return new PyCallBack_VPUNN_VPU4_0_LayerOnTileValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU4_0_LayerOnTileValidValues const &o){ return new VPUNN::VPU4_0_LayerOnTileValidValues(o); } ) );
	}
	{ // VPUNN::VPU2_0_WorkloadValidValues file: line:37
		pybind11::class_<VPUNN::VPU2_0_WorkloadValidValues, std::shared_ptr<VPUNN::VPU2_0_WorkloadValidValues>, PyCallBack_VPUNN_VPU2_0_WorkloadValidValues, VPUNN::IDeviceValidValues> cl(M("VPUNN"), "VPU2_0_WorkloadValidValues", "specific VPU2.0 configuration possibilities, for workload, not layer");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_constraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU2_0_WorkloadValidValues const &o){ return new PyCallBack_VPUNN_VPU2_0_WorkloadValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU2_0_WorkloadValidValues const &o){ return new VPUNN::VPU2_0_WorkloadValidValues(o); } ) );
	}
	{ // VPUNN::VPU2_7_WorkloadValidValues file: line:103
		pybind11::class_<VPUNN::VPU2_7_WorkloadValidValues, std::shared_ptr<VPUNN::VPU2_7_WorkloadValidValues>, PyCallBack_VPUNN_VPU2_7_WorkloadValidValues, VPUNN::IDeviceValidValues> cl(M("VPUNN"), "VPU2_7_WorkloadValidValues", "specific VPU 2.7 configuration possibilities for workload, not layer");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_cosntraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU2_7_WorkloadValidValues const &o){ return new PyCallBack_VPUNN_VPU2_7_WorkloadValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU2_7_WorkloadValidValues const &o){ return new VPUNN::VPU2_7_WorkloadValidValues(o); } ) );
		cl.def("get_output_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU2_7_WorkloadValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU2_7_WorkloadValidValues::get_output_channels_range, "C++: VPUNN::VPU2_7_WorkloadValidValues::get_output_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("get_input_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU2_7_WorkloadValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU2_7_WorkloadValidValues::get_input_channels_range, "C++: VPUNN::VPU2_7_WorkloadValidValues::get_input_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("adapt_device_comaptible_tensor_layout", (enum VPUNN::Layout (VPUNN::VPU2_7_WorkloadValidValues::*)(enum VPUNN::Layout) const) &VPUNN::VPU2_7_WorkloadValidValues::adapt_device_comaptible_tensor_layout, "C++: VPUNN::VPU2_7_WorkloadValidValues::adapt_device_comaptible_tensor_layout(enum VPUNN::Layout) const --> enum VPUNN::Layout", pybind11::arg("layout"));
		cl.def("adapt_device_comaptible_swizzling", (enum VPUNN::Swizzling (VPUNN::VPU2_7_WorkloadValidValues::*)(enum VPUNN::Swizzling) const) &VPUNN::VPU2_7_WorkloadValidValues::adapt_device_comaptible_swizzling, "C++: VPUNN::VPU2_7_WorkloadValidValues::adapt_device_comaptible_swizzling(enum VPUNN::Swizzling) const --> enum VPUNN::Swizzling", pybind11::arg("swizz"));
	}
}


// File: VPUNN_15.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::VPU2_0_LayerValidValues file: line:178
struct PyCallBack_VPUNN_VPU2_0_LayerValidValues : public VPUNN::VPU2_0_LayerValidValues {
	using VPUNN::VPU2_0_LayerValidValues::VPU2_0_LayerValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU2_0_LayerValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

// VPUNN::VPU2_7_LayerValidValues file: line:197
struct PyCallBack_VPUNN_VPU2_7_LayerValidValues : public VPUNN::VPU2_7_LayerValidValues {
	using VPUNN::VPU2_7_LayerValidValues::VPU2_7_LayerValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU2_7_LayerValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU2_7_LayerValidValues::get_input_channels_range(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return VPU2_7_LayerValidValues::get_ISI_Strategy_Range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
};

// VPUNN::VPU2_0_LayerOnTileValidValues file: line:251
struct PyCallBack_VPUNN_VPU2_0_LayerOnTileValidValues : public VPUNN::VPU2_0_LayerOnTileValidValues {
	using VPUNN::VPU2_0_LayerOnTileValidValues::VPU2_0_LayerOnTileValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerOnTileValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerOnTileValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerOnTileValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerOnTileValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU2_0_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_0_LayerOnTileValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

// VPUNN::VPU2_7_LayerOnTileValidValues file: line:262
struct PyCallBack_VPUNN_VPU2_7_LayerOnTileValidValues : public VPUNN::VPU2_7_LayerOnTileValidValues {
	using VPUNN::VPU2_7_LayerOnTileValidValues::VPU2_7_LayerOnTileValidValues;

	using _binder_ret_0 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_0 get_output_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerOnTileValidValues *>(this), "get_output_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::get_output_channels_range(a0);
	}
	using _binder_ret_1 = const class std::vector<int, class std::allocator<int> > &;
	_binder_ret_1 get_input_channels_range(const struct VPUNN::DPUOperation & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerOnTileValidValues *>(this), "get_input_channels_range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::get_input_channels_range(a0);
	}
	enum VPUNN::Layout adapt_device_comaptible_tensor_layout(enum VPUNN::Layout a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerOnTileValidValues *>(this), "adapt_device_comaptible_tensor_layout");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Layout>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Layout> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Layout>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Layout>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::adapt_device_comaptible_tensor_layout(a0);
	}
	enum VPUNN::Swizzling adapt_device_comaptible_swizzling(enum VPUNN::Swizzling a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerOnTileValidValues *>(this), "adapt_device_comaptible_swizzling");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<enum VPUNN::Swizzling>::value) {
				static pybind11::detail::override_caster_t<enum VPUNN::Swizzling> caster;
				return pybind11::detail::cast_ref<enum VPUNN::Swizzling>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<enum VPUNN::Swizzling>(std::move(o));
		}
		return VPU2_7_WorkloadValidValues::adapt_device_comaptible_swizzling(a0);
	}
	using _binder_ret_2 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_2 get_ISI_Strategy_Range(const struct VPUNN::DPUOperation & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::VPU2_7_LayerOnTileValidValues *>(this), "get_ISI_Strategy_Range");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return IDeviceValidValues::get_ISI_Strategy_Range(a0);
	}
};

void bind_VPUNN_15(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::VPU2_0_LayerValidValues file: line:178
		pybind11::class_<VPUNN::VPU2_0_LayerValidValues, std::shared_ptr<VPUNN::VPU2_0_LayerValidValues>, PyCallBack_VPUNN_VPU2_0_LayerValidValues, VPUNN::VPU2_0_WorkloadValidValues> cl(M("VPUNN"), "VPU2_0_LayerValidValues", "specific VPU2.0 configuration possibilities, for  layer");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_cosntraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU2_0_LayerValidValues const &o){ return new PyCallBack_VPUNN_VPU2_0_LayerValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU2_0_LayerValidValues const &o){ return new VPUNN::VPU2_0_LayerValidValues(o); } ) );
		cl.def("get_output_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU2_0_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU2_0_LayerValidValues::get_output_channels_range, "C++: VPUNN::VPU2_0_LayerValidValues::get_output_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
	}
	{ // VPUNN::VPU2_7_LayerValidValues file: line:197
		pybind11::class_<VPUNN::VPU2_7_LayerValidValues, std::shared_ptr<VPUNN::VPU2_7_LayerValidValues>, PyCallBack_VPUNN_VPU2_7_LayerValidValues, VPUNN::VPU2_7_WorkloadValidValues> cl(M("VPUNN"), "VPU2_7_LayerValidValues", "specific VPU 2.7 configuration possibilities for  layer");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_cosntraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU2_7_LayerValidValues const &o){ return new PyCallBack_VPUNN_VPU2_7_LayerValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU2_7_LayerValidValues const &o){ return new VPUNN::VPU2_7_LayerValidValues(o); } ) );
		cl.def("get_output_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU2_7_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU2_7_LayerValidValues::get_output_channels_range, "C++: VPUNN::VPU2_7_LayerValidValues::get_output_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("get_input_channels_range", (const class std::vector<int, class std::allocator<int> > & (VPUNN::VPU2_7_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU2_7_LayerValidValues::get_input_channels_range, "C++: VPUNN::VPU2_7_LayerValidValues::get_input_channels_range(const struct VPUNN::DPUOperation &) const --> const class std::vector<int, class std::allocator<int> > &", pybind11::return_value_policy::automatic, pybind11::arg("dpu"));
		cl.def("get_ISI_Strategy_Range", (class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > (VPUNN::VPU2_7_LayerValidValues::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::VPU2_7_LayerValidValues::get_ISI_Strategy_Range, "C++: VPUNN::VPU2_7_LayerValidValues::get_ISI_Strategy_Range(const struct VPUNN::DPUOperation &) const --> class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >", pybind11::arg("dpu"));
	}
	{ // VPUNN::VPU2_0_LayerOnTileValidValues file: line:251
		pybind11::class_<VPUNN::VPU2_0_LayerOnTileValidValues, std::shared_ptr<VPUNN::VPU2_0_LayerOnTileValidValues>, PyCallBack_VPUNN_VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_0_WorkloadValidValues> cl(M("VPUNN"), "VPU2_0_LayerOnTileValidValues", "specific VPU2.0 configuration possibilities, for  layer already split on tile.\n channels restrictions are less strict vs workload, since a further split is expected");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_cosntraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU2_0_LayerOnTileValidValues const &o){ return new PyCallBack_VPUNN_VPU2_0_LayerOnTileValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU2_0_LayerOnTileValidValues const &o){ return new VPUNN::VPU2_0_LayerOnTileValidValues(o); } ) );
	}
	{ // VPUNN::VPU2_7_LayerOnTileValidValues file: line:262
		pybind11::class_<VPUNN::VPU2_7_LayerOnTileValidValues, std::shared_ptr<VPUNN::VPU2_7_LayerOnTileValidValues>, PyCallBack_VPUNN_VPU2_7_LayerOnTileValidValues, VPUNN::VPU2_7_WorkloadValidValues> cl(M("VPUNN"), "VPU2_7_LayerOnTileValidValues", "specific VPU 2.7 configuration possibilities for  layer already split on tile.\n channels restrictions are less strict vs workload, since a further split is expected");
		cl.def( pybind11::init<const class VPUNN::IContainer_OperationsDynamicBehavior &>(), pybind11::arg("op_dynamic_cosntraints") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_VPU2_7_LayerOnTileValidValues const &o){ return new PyCallBack_VPUNN_VPU2_7_LayerOnTileValidValues(o); } ) );
		cl.def( pybind11::init( [](VPUNN::VPU2_7_LayerOnTileValidValues const &o){ return new VPUNN::VPU2_7_LayerOnTileValidValues(o); } ) );
	}
	{ // VPUNN::Checker file: line:24
		pybind11::class_<VPUNN::Checker, std::shared_ptr<VPUNN::Checker>> cl(M("VPUNN"), "Checker", "simple checker mechanism that logs textual info and remembers if a error was recorded.");
		cl.def( pybind11::init( [](VPUNN::Checker const &o){ return new VPUNN::Checker(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::Checker(); } ) );
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const int &, const class std::vector<int, class std::allocator<int> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<int>, "C++: VPUNN::Checker::check_is_in_list(const int &, const class std::vector<int, class std::allocator<int> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::VPUDevice &, const class std::vector<enum VPUNN::VPUDevice, class std::allocator<enum VPUNN::VPUDevice> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::VPUDevice>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::VPUDevice &, const class std::vector<enum VPUNN::VPUDevice, class std::allocator<enum VPUNN::VPUDevice> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::Operation &, const class std::vector<enum VPUNN::Operation, class std::allocator<enum VPUNN::Operation> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::Operation>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::Operation &, const class std::vector<enum VPUNN::Operation, class std::allocator<enum VPUNN::Operation> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::ISIStrategy &, const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::ISIStrategy>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::ISIStrategy &, const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::DataType &, const class std::vector<enum VPUNN::DataType, class std::allocator<enum VPUNN::DataType> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::DataType>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::DataType &, const class std::vector<enum VPUNN::DataType, class std::allocator<enum VPUNN::DataType> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::Layout &, const class std::vector<enum VPUNN::Layout, class std::allocator<enum VPUNN::Layout> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::Layout>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::Layout &, const class std::vector<enum VPUNN::Layout, class std::allocator<enum VPUNN::Layout> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::Swizzling &, const class std::vector<enum VPUNN::Swizzling, class std::allocator<enum VPUNN::Swizzling> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::Swizzling>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::Swizzling &, const class std::vector<enum VPUNN::Swizzling, class std::allocator<enum VPUNN::Swizzling> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_list", (bool (VPUNN::Checker::*)(const enum VPUNN::ExecutionMode &, const class std::vector<enum VPUNN::ExecutionMode, class std::allocator<enum VPUNN::ExecutionMode> > &, const std::string &)) &VPUNN::Checker::check_is_in_list<VPUNN::ExecutionMode>, "C++: VPUNN::Checker::check_is_in_list(const enum VPUNN::ExecutionMode &, const class std::vector<enum VPUNN::ExecutionMode, class std::allocator<enum VPUNN::ExecutionMode> > &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("container"), pybind11::arg("what"));
		cl.def("check_is_in_interval", (bool (VPUNN::Checker::*)(const int &, const struct std::pair<int, int> &, const std::string &)) &VPUNN::Checker::check_is_in_interval<int>, "C++: VPUNN::Checker::check_is_in_interval(const int &, const struct std::pair<int, int> &, const std::string &) --> bool", pybind11::arg("item"), pybind11::arg("interval"), pybind11::arg("what"));
		cl.def("check_is_equal", (bool (VPUNN::Checker::*)(const bool &, const bool &, const std::string)) &VPUNN::Checker::check_is_equal<bool>, "C++: VPUNN::Checker::check_is_equal(const bool &, const bool &, const std::string) --> bool", pybind11::arg("item"), pybind11::arg("right_side"), pybind11::arg("what"));
		cl.def("check_is_equal", (bool (VPUNN::Checker::*)(const float &, const float &, const std::string)) &VPUNN::Checker::check_is_equal<float>, "C++: VPUNN::Checker::check_is_equal(const float &, const float &, const std::string) --> bool", pybind11::arg("item"), pybind11::arg("right_side"), pybind11::arg("what"));
		cl.def_static("set_print_tags", (bool (*)(bool)) &VPUNN::Checker::set_print_tags, "sets a new mode and returns the current mode\n\nC++: VPUNN::Checker::set_print_tags(bool) --> bool", pybind11::arg("new_mode"));
		cl.def("reset", (bool (VPUNN::Checker::*)()) &VPUNN::Checker::reset, "cleans  up the history\n \n\n the state before reset\n\nC++: VPUNN::Checker::reset() --> bool");
		cl.def("is_clean", (bool (VPUNN::Checker::*)() const) &VPUNN::Checker::is_clean, "true if the checker has no error recorded\n\nC++: VPUNN::Checker::is_clean() const --> bool");
		cl.def("add_check_failed", (void (VPUNN::Checker::*)(const std::string &)) &VPUNN::Checker::add_check_failed, "marks the checker with error and ads a textual info\n \n\n the string with information\n\nC++: VPUNN::Checker::add_check_failed(const std::string &) --> void", pybind11::arg("info"));
		cl.def("findings", (std::string (VPUNN::Checker::*)() const) &VPUNN::Checker::findings, "the string containing the textual information that was logged (since reset).\n\nC++: VPUNN::Checker::findings() const --> std::string");

		{ // VPUNN::Checker::Show file: line:72
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<bool,void>, std::shared_ptr<VPUNN::Checker::Show<bool,void>>> cl(enclosing_class, "Show_bool_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<bool,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const bool &)) &VPUNN::Checker::Show<bool, void>::show_value, "C++: VPUNN::Checker::Show<bool, void>::show_value(const bool &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:72
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<float,void>, std::shared_ptr<VPUNN::Checker::Show<float,void>>> cl(enclosing_class, "Show_float_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<float,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const float &)) &VPUNN::Checker::Show<float, void>::show_value, "C++: VPUNN::Checker::Show<float, void>::show_value(const float &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:72
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<int,void>, std::shared_ptr<VPUNN::Checker::Show<int,void>>> cl(enclosing_class, "Show_int_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<int,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const int &)) &VPUNN::Checker::Show<int, void>::show_value, "C++: VPUNN::Checker::Show<int, void>::show_value(const int &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::VPUDevice,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::VPUDevice,void>>> cl(enclosing_class, "Show_VPUNN_VPUDevice_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::VPUDevice,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::VPUDevice &)) &VPUNN::Checker::Show<VPUNN::VPUDevice, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::VPUDevice, void>::show_value(const enum VPUNN::VPUDevice &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::Operation,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::Operation,void>>> cl(enclosing_class, "Show_VPUNN_Operation_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::Operation,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::Operation &)) &VPUNN::Checker::Show<VPUNN::Operation, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::Operation, void>::show_value(const enum VPUNN::Operation &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::ISIStrategy,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::ISIStrategy,void>>> cl(enclosing_class, "Show_VPUNN_ISIStrategy_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::ISIStrategy,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::ISIStrategy &)) &VPUNN::Checker::Show<VPUNN::ISIStrategy, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::ISIStrategy, void>::show_value(const enum VPUNN::ISIStrategy &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::DataType,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::DataType,void>>> cl(enclosing_class, "Show_VPUNN_DataType_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::DataType,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::DataType &)) &VPUNN::Checker::Show<VPUNN::DataType, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::DataType, void>::show_value(const enum VPUNN::DataType &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::Layout,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::Layout,void>>> cl(enclosing_class, "Show_VPUNN_Layout_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::Layout,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::Layout &)) &VPUNN::Checker::Show<VPUNN::Layout, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::Layout, void>::show_value(const enum VPUNN::Layout &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::Swizzling,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::Swizzling,void>>> cl(enclosing_class, "Show_VPUNN_Swizzling_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::Swizzling,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::Swizzling &)) &VPUNN::Checker::Show<VPUNN::Swizzling, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::Swizzling, void>::show_value(const enum VPUNN::Swizzling &) --> std::string", pybind11::arg("item"));
		}

		{ // VPUNN::Checker::Show file: line:82
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::Checker::Show<VPUNN::ExecutionMode,void>, std::shared_ptr<VPUNN::Checker::Show<VPUNN::ExecutionMode,void>>> cl(enclosing_class, "Show_VPUNN_ExecutionMode_void_t", "");
			cl.def( pybind11::init( [](){ return new VPUNN::Checker::Show<VPUNN::ExecutionMode,void>(); } ) );
			cl.def_static("show_value", (std::string (*)(const enum VPUNN::ExecutionMode &)) &VPUNN::Checker::Show<VPUNN::ExecutionMode, void>::show_value, "C++: VPUNN::Checker::Show<VPUNN::ExecutionMode, void>::show_value(const enum VPUNN::ExecutionMode &) --> std::string", pybind11::arg("item"));
		}

	}
}


// File: VPUNN_16.cpp
#include <array> // std::array
#include <ios> // std::_Ios_Seekdir
#include <iterator> // __gnu_cxx::__normal_iterator
#include <iterator> // std::reverse_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::_Bit_const_iterator
#include <vector> // std::_Bit_iterator
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::IOperationDynamicGenerator file: line:23
struct PyCallBack_VPUNN_IOperationDynamicGenerator : public VPUNN::IOperationDynamicGenerator {
	using VPUNN::IOperationDynamicGenerator::IOperationDynamicGenerator;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicGenerator *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicGenerator::generate_operation_dependent_tensors\"");
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::IOperationDynamicGenerator *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicGenerator::generate_sparsity\"");
	}
};

// VPUNN::Base_Constraints file: line:37
struct PyCallBack_VPUNN_Base_Constraints : public VPUNN::Base_Constraints {
	using VPUNN::Base_Constraints::Base_Constraints;

	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::deduce_input_1\"");
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOperationDynamicConstraints::limit_sparsity(a0, a1);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Base_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::check_input_output_tensor_corelation\"");
	}
};

// VPUNN::GenericConvolution_Constraints file: line:94
struct PyCallBack_VPUNN_GenericConvolution_Constraints : public VPUNN::GenericConvolution_Constraints {
	using VPUNN::GenericConvolution_Constraints::GenericConvolution_Constraints;

	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::generate_sparsity(a0, a1, a2);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::deduce_input_1\"");
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicConstraints::check_input_output_tensor_corelation\"");
	}
	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::GenericConvolution_Constraints *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"IOperationDynamicGenerator::generate_operation_dependent_tensors\"");
	}
};

// VPUNN::CONVOLUTION_Constraints file: line:139
struct PyCallBack_VPUNN_CONVOLUTION_Constraints : public VPUNN::CONVOLUTION_Constraints {
	using VPUNN::CONVOLUTION_Constraints::CONVOLUTION_Constraints;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CONVOLUTION_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return CONVOLUTION_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CONVOLUTION_Constraints::generate_sparsity(a0, a1, a2);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return CONVOLUTION_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CONVOLUTION_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
};

// VPUNN::DW_CONVOLUTION_Constraints file: line:215
struct PyCallBack_VPUNN_DW_CONVOLUTION_Constraints : public VPUNN::DW_CONVOLUTION_Constraints {
	using VPUNN::DW_CONVOLUTION_Constraints::DW_CONVOLUTION_Constraints;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DW_CONVOLUTION_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DW_CONVOLUTION_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DW_CONVOLUTION_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::generate_sparsity(a0, a1, a2);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
};

// VPUNN::CM_CONVOLUTION_Constraints file: line:255
struct PyCallBack_VPUNN_CM_CONVOLUTION_Constraints : public VPUNN::CM_CONVOLUTION_Constraints {
	using VPUNN::CM_CONVOLUTION_Constraints::CM_CONVOLUTION_Constraints;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::input_0_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::generate_sparsity(a0, a1, a2);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
};

// VPUNN::ELTWISE_Constraints file: line:308
struct PyCallBack_VPUNN_ELTWISE_Constraints : public VPUNN::ELTWISE_Constraints {
	using VPUNN::ELTWISE_Constraints::ELTWISE_Constraints;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ELTWISE_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return ELTWISE_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ELTWISE_Constraints::generate_sparsity(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return ELTWISE_Constraints::input_1_volume(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return ELTWISE_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ELTWISE_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return ELTWISE_Constraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return ELTWISE_Constraints::filter_output_write_tile_Options(a0);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return ELTWISE_Constraints::get_weight_table_size(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOperationDynamicConstraints::limit_sparsity(a0, a1);
	}
};

// VPUNN::MAXPOOL_Constraints file: line:408
struct PyCallBack_VPUNN_MAXPOOL_Constraints : public VPUNN::MAXPOOL_Constraints {
	using VPUNN::MAXPOOL_Constraints::MAXPOOL_Constraints;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MAXPOOL_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return MAXPOOL_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MAXPOOL_Constraints::generate_sparsity(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return MAXPOOL_Constraints::input_1_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MAXPOOL_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOperationDynamicConstraints::limit_sparsity(a0, a1);
	}
};

void bind_VPUNN_16(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::Sampler file: line:18
		pybind11::class_<VPUNN::Sampler, std::shared_ptr<VPUNN::Sampler>> cl(M("VPUNN"), "Sampler", "Generates sample, different distributions, controlled random seed");
		cl.def( pybind11::init( [](){ return new VPUNN::Sampler(); } ) );
		cl.def( pybind11::init<unsigned int>(), pybind11::arg("forced_seed") );

		cl.def( pybind11::init( [](VPUNN::Sampler const &o){ return new VPUNN::Sampler(o); } ) );
		cl.def("sample_list", (enum VPUNN::DataType (VPUNN::Sampler::*)(const class std::vector<enum VPUNN::DataType, class std::allocator<enum VPUNN::DataType> > &) const) &VPUNN::Sampler::sample_list<std::vector<VPUNN::DataType, std::allocator<VPUNN::DataType> >>, "C++: VPUNN::Sampler::sample_list(const class std::vector<enum VPUNN::DataType, class std::allocator<enum VPUNN::DataType> > &) const --> enum VPUNN::DataType", pybind11::arg("elements"));
		cl.def("sample_list", (bool (VPUNN::Sampler::*)(const class std::vector<bool, class std::allocator<bool> > &) const) &VPUNN::Sampler::sample_list<std::vector<bool, std::allocator<bool> >>, "C++: VPUNN::Sampler::sample_list(const class std::vector<bool, class std::allocator<bool> > &) const --> bool", pybind11::arg("elements"));
		cl.def("sample_list_decrease_prob", (int (VPUNN::Sampler::*)(const class std::vector<int, class std::allocator<int> > &) const) &VPUNN::Sampler::sample_list_decrease_prob<std::vector<int, std::allocator<int> >>, "C++: VPUNN::Sampler::sample_list_decrease_prob(const class std::vector<int, class std::allocator<int> > &) const --> int", pybind11::arg("elements"));
		cl.def("get_seed", (unsigned int (VPUNN::Sampler::*)() const) &VPUNN::Sampler::get_seed, "C++: VPUNN::Sampler::get_seed() const --> unsigned int");
		cl.def("sample_continuous_uniform", [](VPUNN::Sampler &o) -> float { return o.sample_continuous_uniform(); }, "");
		cl.def("sample_continuous_uniform", [](VPUNN::Sampler &o, float const & a0) -> float { return o.sample_continuous_uniform(a0); }, "", pybind11::arg("min_interval_closed"));
		cl.def("sample_continuous_uniform", (float (VPUNN::Sampler::*)(float, float)) &VPUNN::Sampler::sample_continuous_uniform, "C++: VPUNN::Sampler::sample_continuous_uniform(float, float) --> float", pybind11::arg("min_interval_closed"), pybind11::arg("max_interval_open"));
	}
	{ // VPUNN::IOperationDynamicGenerator file: line:23
		pybind11::class_<VPUNN::IOperationDynamicGenerator, VPUNN::IOperationDynamicGenerator*, PyCallBack_VPUNN_IOperationDynamicGenerator> cl(M("VPUNN"), "IOperationDynamicGenerator", "");
		cl.def(pybind11::init<PyCallBack_VPUNN_IOperationDynamicGenerator const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_IOperationDynamicGenerator(); } ) );
		cl.def("generate_operation_dependent_tensors", (void (VPUNN::IOperationDynamicGenerator::*)(class VPUNN::Sampler &, const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const) &VPUNN::IOperationDynamicGenerator::generate_operation_dependent_tensors, "dynamic establishment of output_0 and input_1\n eg input_1 (weights) may depend dynamically on output_0 info (channels)\n\nC++: VPUNN::IOperationDynamicGenerator::generate_operation_dependent_tensors(class VPUNN::Sampler &, const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const --> void", pybind11::arg("sampler"), pybind11::arg("config"), pybind11::arg("dpu"));
		cl.def("generate_sparsity", (void (VPUNN::IOperationDynamicGenerator::*)(class VPUNN::Sampler &, const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const) &VPUNN::IOperationDynamicGenerator::generate_sparsity, "fills/generates sparsity values\n\nC++: VPUNN::IOperationDynamicGenerator::generate_sparsity(class VPUNN::Sampler &, const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const --> void", pybind11::arg("sampler"), pybind11::arg("config"), pybind11::arg("dpu"));
		cl.def("assign", (class VPUNN::IOperationDynamicGenerator & (VPUNN::IOperationDynamicGenerator::*)(const class VPUNN::IOperationDynamicGenerator &)) &VPUNN::IOperationDynamicGenerator::operator=, "C++: VPUNN::IOperationDynamicGenerator::operator=(const class VPUNN::IOperationDynamicGenerator &) --> class VPUNN::IOperationDynamicGenerator &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::Base_Constraints file: line:37
		pybind11::class_<VPUNN::Base_Constraints, std::shared_ptr<VPUNN::Base_Constraints>, PyCallBack_VPUNN_Base_Constraints, VPUNN::IOperationDynamicConstraints> cl(M("VPUNN"), "Base_Constraints", "");
		cl.def(pybind11::init<PyCallBack_VPUNN_Base_Constraints const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_Base_Constraints(); } ) );
		cl.def("check_sparsity_rules", (bool (VPUNN::Base_Constraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const) &VPUNN::Base_Constraints::check_sparsity_rules, "this specialization checks sparsity is turned off\n\nC++: VPUNN::Base_Constraints::check_sparsity_rules(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const --> bool", pybind11::arg(""), pybind11::arg("dpu"), pybind11::arg("info"));
		cl.def("input_1_volume", (long long (VPUNN::Base_Constraints::*)(const struct VPUNN::TensorInfo &) const) &VPUNN::Base_Constraints::input_1_volume, "C++: VPUNN::Base_Constraints::input_1_volume(const struct VPUNN::TensorInfo &) const --> long long", pybind11::arg("w"));
		cl.def("input_1_aligned_size_bytes", (long long (VPUNN::Base_Constraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const) &VPUNN::Base_Constraints::input_1_aligned_size_bytes, "computes the aligned size in bytes for weights\n\nC++: VPUNN::Base_Constraints::input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const --> long long", pybind11::arg("config"), pybind11::arg("dpu"));
		cl.def("input_1_contiguous_size_bytes", (long long (VPUNN::Base_Constraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const) &VPUNN::Base_Constraints::input_1_contiguous_size_bytes, "computes the non CMX aligned/contiguous  size in bytes for the weights\n\nC++: VPUNN::Base_Constraints::input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &) const --> long long", pybind11::arg("config"), pybind11::arg("dpu"));
		cl.def("assign", (class VPUNN::Base_Constraints & (VPUNN::Base_Constraints::*)(const class VPUNN::Base_Constraints &)) &VPUNN::Base_Constraints::operator=, "C++: VPUNN::Base_Constraints::operator=(const class VPUNN::Base_Constraints &) --> class VPUNN::Base_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::GenericConvolution_Constraints file: line:94
		pybind11::class_<VPUNN::GenericConvolution_Constraints, std::shared_ptr<VPUNN::GenericConvolution_Constraints>, PyCallBack_VPUNN_GenericConvolution_Constraints, VPUNN::Base_Constraints, VPUNN::IOperationDynamicGenerator> cl(M("VPUNN"), "GenericConvolution_Constraints", "");
		cl.def(pybind11::init<PyCallBack_VPUNN_GenericConvolution_Constraints const &>());
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_GenericConvolution_Constraints(); } ) );
		cl.def("generate_sparsity", (void (VPUNN::GenericConvolution_Constraints::*)(class VPUNN::Sampler &, const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const) &VPUNN::GenericConvolution_Constraints::generate_sparsity, "C++: VPUNN::GenericConvolution_Constraints::generate_sparsity(class VPUNN::Sampler &, const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg("dpu"));
		cl.def("limit_sparsity", (void (VPUNN::GenericConvolution_Constraints::*)(const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const) &VPUNN::GenericConvolution_Constraints::limit_sparsity, "@ reduces/adjusts sparsity  according to context\n\nC++: VPUNN::GenericConvolution_Constraints::limit_sparsity(const class VPUNN::IDeviceValidValues &, struct VPUNN::DPUOperation &) const --> void", pybind11::arg(""), pybind11::arg("dpu"));
		cl.def("check_sparsity_layer_SOK", (bool (VPUNN::GenericConvolution_Constraints::*)(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const) &VPUNN::GenericConvolution_Constraints::check_sparsity_layer_SOK, "if SOK => channels must be aligned to 32 channels\n rule to be applied to un-tiled layer, before split on tiles\n\nC++: VPUNN::GenericConvolution_Constraints::check_sparsity_layer_SOK(const class VPUNN::IDeviceValidValues &, const struct VPUNN::DPUOperation &, std::string &) const --> bool", pybind11::arg(""), pybind11::arg("dpu"), pybind11::arg("info"));
		cl.def("assign", (class VPUNN::GenericConvolution_Constraints & (VPUNN::GenericConvolution_Constraints::*)(const class VPUNN::GenericConvolution_Constraints &)) &VPUNN::GenericConvolution_Constraints::operator=, "C++: VPUNN::GenericConvolution_Constraints::operator=(const class VPUNN::GenericConvolution_Constraints &) --> class VPUNN::GenericConvolution_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::CONVOLUTION_Constraints file: line:139
		pybind11::class_<VPUNN::CONVOLUTION_Constraints, std::shared_ptr<VPUNN::CONVOLUTION_Constraints>, PyCallBack_VPUNN_CONVOLUTION_Constraints, VPUNN::GenericConvolution_Constraints> cl(M("VPUNN"), "CONVOLUTION_Constraints", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_CONVOLUTION_Constraints const &o){ return new PyCallBack_VPUNN_CONVOLUTION_Constraints(o); } ) );
		cl.def( pybind11::init( [](VPUNN::CONVOLUTION_Constraints const &o){ return new VPUNN::CONVOLUTION_Constraints(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::CONVOLUTION_Constraints(); }, [](){ return new PyCallBack_VPUNN_CONVOLUTION_Constraints(); } ) );
		cl.def("assign", (class VPUNN::CONVOLUTION_Constraints & (VPUNN::CONVOLUTION_Constraints::*)(const class VPUNN::CONVOLUTION_Constraints &)) &VPUNN::CONVOLUTION_Constraints::operator=, "C++: VPUNN::CONVOLUTION_Constraints::operator=(const class VPUNN::CONVOLUTION_Constraints &) --> class VPUNN::CONVOLUTION_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::DW_CONVOLUTION_Constraints file: line:215
		pybind11::class_<VPUNN::DW_CONVOLUTION_Constraints, std::shared_ptr<VPUNN::DW_CONVOLUTION_Constraints>, PyCallBack_VPUNN_DW_CONVOLUTION_Constraints, VPUNN::GenericConvolution_Constraints> cl(M("VPUNN"), "DW_CONVOLUTION_Constraints", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_DW_CONVOLUTION_Constraints const &o){ return new PyCallBack_VPUNN_DW_CONVOLUTION_Constraints(o); } ) );
		cl.def( pybind11::init( [](VPUNN::DW_CONVOLUTION_Constraints const &o){ return new VPUNN::DW_CONVOLUTION_Constraints(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DW_CONVOLUTION_Constraints(); }, [](){ return new PyCallBack_VPUNN_DW_CONVOLUTION_Constraints(); } ) );
		cl.def("assign", (class VPUNN::DW_CONVOLUTION_Constraints & (VPUNN::DW_CONVOLUTION_Constraints::*)(const class VPUNN::DW_CONVOLUTION_Constraints &)) &VPUNN::DW_CONVOLUTION_Constraints::operator=, "C++: VPUNN::DW_CONVOLUTION_Constraints::operator=(const class VPUNN::DW_CONVOLUTION_Constraints &) --> class VPUNN::DW_CONVOLUTION_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::CM_CONVOLUTION_Constraints file: line:255
		pybind11::class_<VPUNN::CM_CONVOLUTION_Constraints, std::shared_ptr<VPUNN::CM_CONVOLUTION_Constraints>, PyCallBack_VPUNN_CM_CONVOLUTION_Constraints, VPUNN::GenericConvolution_Constraints> cl(M("VPUNN"), "CM_CONVOLUTION_Constraints", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_CM_CONVOLUTION_Constraints const &o){ return new PyCallBack_VPUNN_CM_CONVOLUTION_Constraints(o); } ) );
		cl.def( pybind11::init( [](VPUNN::CM_CONVOLUTION_Constraints const &o){ return new VPUNN::CM_CONVOLUTION_Constraints(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::CM_CONVOLUTION_Constraints(); }, [](){ return new PyCallBack_VPUNN_CM_CONVOLUTION_Constraints(); } ) );
		cl.def("assign", (class VPUNN::CM_CONVOLUTION_Constraints & (VPUNN::CM_CONVOLUTION_Constraints::*)(const class VPUNN::CM_CONVOLUTION_Constraints &)) &VPUNN::CM_CONVOLUTION_Constraints::operator=, "C++: VPUNN::CM_CONVOLUTION_Constraints::operator=(const class VPUNN::CM_CONVOLUTION_Constraints &) --> class VPUNN::CM_CONVOLUTION_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::ELTWISE_Constraints file: line:308
		pybind11::class_<VPUNN::ELTWISE_Constraints, std::shared_ptr<VPUNN::ELTWISE_Constraints>, PyCallBack_VPUNN_ELTWISE_Constraints, VPUNN::Base_Constraints, VPUNN::IOperationDynamicGenerator> cl(M("VPUNN"), "ELTWISE_Constraints", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_ELTWISE_Constraints const &o){ return new PyCallBack_VPUNN_ELTWISE_Constraints(o); } ) );
		cl.def( pybind11::init( [](VPUNN::ELTWISE_Constraints const &o){ return new VPUNN::ELTWISE_Constraints(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::ELTWISE_Constraints(); }, [](){ return new PyCallBack_VPUNN_ELTWISE_Constraints(); } ) );
		cl.def("assign", (class VPUNN::ELTWISE_Constraints & (VPUNN::ELTWISE_Constraints::*)(const class VPUNN::ELTWISE_Constraints &)) &VPUNN::ELTWISE_Constraints::operator=, "C++: VPUNN::ELTWISE_Constraints::operator=(const class VPUNN::ELTWISE_Constraints &) --> class VPUNN::ELTWISE_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::MAXPOOL_Constraints file: line:408
		pybind11::class_<VPUNN::MAXPOOL_Constraints, std::shared_ptr<VPUNN::MAXPOOL_Constraints>, PyCallBack_VPUNN_MAXPOOL_Constraints, VPUNN::Base_Constraints, VPUNN::IOperationDynamicGenerator> cl(M("VPUNN"), "MAXPOOL_Constraints", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_MAXPOOL_Constraints const &o){ return new PyCallBack_VPUNN_MAXPOOL_Constraints(o); } ) );
		cl.def( pybind11::init( [](VPUNN::MAXPOOL_Constraints const &o){ return new VPUNN::MAXPOOL_Constraints(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::MAXPOOL_Constraints(); }, [](){ return new PyCallBack_VPUNN_MAXPOOL_Constraints(); } ) );
		cl.def("assign", (class VPUNN::MAXPOOL_Constraints & (VPUNN::MAXPOOL_Constraints::*)(const class VPUNN::MAXPOOL_Constraints &)) &VPUNN::MAXPOOL_Constraints::operator=, "C++: VPUNN::MAXPOOL_Constraints::operator=(const class VPUNN::MAXPOOL_Constraints &) --> class VPUNN::MAXPOOL_Constraints &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::MemorySize file: line:24
		pybind11::class_<VPUNN::MemorySize, std::shared_ptr<VPUNN::MemorySize>> cl(M("VPUNN"), "MemorySize", "size of a DPU tensors in memory");
		cl.def( pybind11::init( [](){ return new VPUNN::MemorySize(); } ) );
		cl.def( pybind11::init( [](VPUNN::MemorySize const &o){ return new VPUNN::MemorySize(o); } ) );
		cl.def_readwrite("cmx", &VPUNN::MemorySize::cmx);
		cl.def_readwrite("input_0", &VPUNN::MemorySize::input_0);
		cl.def_readwrite("input_1", &VPUNN::MemorySize::input_1);
		cl.def_readwrite("output_0", &VPUNN::MemorySize::output_0);
		cl.def_readwrite("inplace_output", &VPUNN::MemorySize::inplace_output);
		cl.def_readwrite("cmx_overhead", &VPUNN::MemorySize::cmx_overhead);
		cl.def_readwrite("ignore_overhead", &VPUNN::MemorySize::ignore_overhead);

		cl.def("__str__", [](VPUNN::MemorySize const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::WorkloadMemorySizeCalculator file: line:48
		pybind11::class_<VPUNN::WorkloadMemorySizeCalculator, std::shared_ptr<VPUNN::WorkloadMemorySizeCalculator>> cl(M("VPUNN"), "WorkloadMemorySizeCalculator", "");
		cl.def( pybind11::init( [](){ return new VPUNN::WorkloadMemorySizeCalculator(); } ) );
		cl.def( pybind11::init( [](VPUNN::WorkloadMemorySizeCalculator const &o){ return new VPUNN::WorkloadMemorySizeCalculator(o); } ) );
		cl.def("set_ignore_cmx_overhead", (void (VPUNN::WorkloadMemorySizeCalculator::*)(bool)) &VPUNN::WorkloadMemorySizeCalculator::set_ignore_cmx_overhead, "changes the state of ignore_cmx_overhead, that controls if overhead is added or not to final memory\n\nC++: VPUNN::WorkloadMemorySizeCalculator::set_ignore_cmx_overhead(bool) --> void", pybind11::arg("new_state"));
		cl.def("compute_memory", (struct VPUNN::MemorySize (VPUNN::WorkloadMemorySizeCalculator::*)(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &) const) &VPUNN::WorkloadMemorySizeCalculator::compute_memory, "cmx memory in bytes , not considering broadcasting\n\n \n is the workload for which the memory to be computed\n \n\n knows device configurations and restrictions\n \n\n memory information.\n\nC++: VPUNN::WorkloadMemorySizeCalculator::compute_memory(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &) const --> struct VPUNN::MemorySize", pybind11::arg("w"), pybind11::arg("config"));
	}
}


// File: VPUNN_17.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_17(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SanityReport file: line:20
		pybind11::class_<VPUNN::SanityReport, std::shared_ptr<VPUNN::SanityReport>> cl(M("VPUNN"), "SanityReport", "Post sanity analysis.");
		cl.def( pybind11::init( [](VPUNN::SanityReport const &o){ return new VPUNN::SanityReport(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::SanityReport(); } ) );
		cl.def_readwrite("info", &VPUNN::SanityReport::info);
		cl.def("is_usable", (bool (VPUNN::SanityReport::*)() const) &VPUNN::SanityReport::is_usable, "is the workload usable for NN run\n\nC++: VPUNN::SanityReport::is_usable() const --> bool");
		cl.def("has_error", (bool (VPUNN::SanityReport::*)() const) &VPUNN::SanityReport::has_error, "C++: VPUNN::SanityReport::has_error() const --> bool");
		cl.def("resetOK", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::resetOK, "C++: VPUNN::SanityReport::resetOK() --> void");
		cl.def("value", (unsigned int (VPUNN::SanityReport::*)() const) &VPUNN::SanityReport::value, "C++: VPUNN::SanityReport::value() const --> unsigned int");
		cl.def("mark_size_too_big", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::mark_size_too_big, "C++: VPUNN::SanityReport::mark_size_too_big() --> void");
		cl.def("mark_unknown_device", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::mark_unknown_device, "C++: VPUNN::SanityReport::mark_unknown_device() --> void");
		cl.def("mark_unknown_operation", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::mark_unknown_operation, "C++: VPUNN::SanityReport::mark_unknown_operation() --> void");
		cl.def("mark_invalid_NN_response", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::mark_invalid_NN_response, "C++: VPUNN::SanityReport::mark_invalid_NN_response() --> void");
		cl.def("mark_invalid_DPU_workload", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::mark_invalid_DPU_workload, "C++: VPUNN::SanityReport::mark_invalid_DPU_workload() --> void");
		cl.def("mark_invalid_LayerConfiguration", (void (VPUNN::SanityReport::*)()) &VPUNN::SanityReport::mark_invalid_LayerConfiguration, "C++: VPUNN::SanityReport::mark_invalid_LayerConfiguration() --> void");
		cl.def("assign", (struct VPUNN::SanityReport & (VPUNN::SanityReport::*)(const struct VPUNN::SanityReport &)) &VPUNN::SanityReport::operator=, "C++: VPUNN::SanityReport::operator=(const struct VPUNN::SanityReport &) --> struct VPUNN::SanityReport &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::ContextualMemoryCalculator file: line:38
		pybind11::class_<VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>, std::shared_ptr<VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>>> cl(M("VPUNN"), "ContextualMemoryCalculator_VPUNN_Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_VPUNN_VPU2_0_WorkloadValidValues_VPUNN_VPU2_7_WorkloadValidValues_VPUNN_VPU4_0_WorkloadValidValues_t", "");
		cl.def( pybind11::init( [](VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>> const &o){ return new VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>(o); } ) );
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory(const struct VPUNN::DPUOperation &) const --> struct VPUNN::MemorySize", pybind11::arg("w"));
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory(const struct VPUNN::DPUWorkload &) const --> struct VPUNN::MemorySize", pybind11::arg("wl"));
	}
	{ // VPUNN::ContextualMemoryCalculator file: line:38
		pybind11::class_<VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>, std::shared_ptr<VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>>> cl(M("VPUNN"), "ContextualMemoryCalculator_VPUNN_Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_VPUNN_VPU2_0_LayerOnTileValidValues_VPUNN_VPU2_7_LayerOnTileValidValues_VPUNN_VPU4_0_LayerOnTileValidValues_t", "");
		cl.def( pybind11::init( [](VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>> const &o){ return new VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>(o); } ) );
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory(const struct VPUNN::DPUOperation &) const --> struct VPUNN::MemorySize", pybind11::arg("w"));
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory(const struct VPUNN::DPUWorkload &) const --> struct VPUNN::MemorySize", pybind11::arg("wl"));
	}
	{ // VPUNN::DPU_ConfigurableOperationValidator file: line:83
		pybind11::class_<VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>, std::shared_ptr<VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>>, VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>, VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>> cl(M("VPUNN"), "DPU_ConfigurableOperationValidator_VPUNN_Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_VPUNN_VPU2_0_WorkloadValidValues_VPUNN_VPU2_7_WorkloadValidValues_VPUNN_VPU4_0_WorkloadValidValues_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>(); } ) );
		cl.def( pybind11::init( [](VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>> const &o){ return new VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>(o); } ) );
		cl.def("construct_input_1", (class VPUNN::VPUTensor (VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::construct_input_1, "C++: VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::construct_input_1(const struct VPUNN::DPUWorkload &) const --> class VPUNN::VPUTensor", pybind11::arg("wl"));
		cl.def("check_workload_consistency", (void (VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>::*)(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &, const class VPUNN::IOperationDynamicConstraints &, struct VPUNN::SanityReport &) const) &VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::check_workload_consistency, "C++: VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::check_workload_consistency(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &, const class VPUNN::IOperationDynamicConstraints &, struct VPUNN::SanityReport &) const --> void", pybind11::arg("w"), pybind11::arg("config"), pybind11::arg("operation_behaviour"), pybind11::arg("result"));
		cl.def("is_supported", (bool (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::is_supported, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::is_supported(enum VPUNN::VPUDevice) const --> bool", pybind11::arg("device"));
		cl.def("get_config", (const class VPUNN::IDeviceValidValues & (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::get_config, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>::get_config(enum VPUNN::VPUDevice) const --> const class VPUNN::IDeviceValidValues &", pybind11::return_value_policy::automatic, pybind11::arg("device"));
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory(const struct VPUNN::DPUOperation &) const --> struct VPUNN::MemorySize", pybind11::arg("w"));
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues>>::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_WorkloadValidValues, VPUNN::VPU2_7_WorkloadValidValues, VPUNN::VPU4_0_WorkloadValidValues> >::compute_wl_memory(const struct VPUNN::DPUWorkload &) const --> struct VPUNN::MemorySize", pybind11::arg("wl"));
	}
}


// File: VPUNN_18.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::Preprocessing file: line:31
struct PyCallBack_VPUNN_Preprocessing_float_t : public VPUNN::Preprocessing<float> {
	using VPUNN::Preprocessing<float>::Preprocessing;

	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing<float> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"Preprocessing::generate_descriptor\"");
	}
	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing<float> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"Preprocessing::interface_version\"");
	}
};

// VPUNN::PreprocessingInserter file: line:162
struct PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_PreprocessingLatest_float_t : public VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> {
	using VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::PreprocessingInserter;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

void bind_VPUNN_18(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::DPU_ConfigurableOperationValidator file: line:83
		pybind11::class_<VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>, std::shared_ptr<VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>>, VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>, VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>> cl(M("VPUNN"), "DPU_ConfigurableOperationValidator_VPUNN_Behavior_Device_Mapping_VPUNN_Behaviours_VPUNN_CONVOLUTION_Constraints_VPUNN_DW_CONVOLUTION_Constraints_VPUNN_CM_CONVOLUTION_Constraints_VPUNN_ELTWISE_Constraints_VPUNN_MAXPOOL_Constraints_VPUNN_VPU2_0_LayerOnTileValidValues_VPUNN_VPU2_7_LayerOnTileValidValues_VPUNN_VPU4_0_LayerOnTileValidValues_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>(); } ) );
		cl.def( pybind11::init( [](VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>> const &o){ return new VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>(o); } ) );
		cl.def("construct_input_1", (class VPUNN::VPUTensor (VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::construct_input_1, "C++: VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::construct_input_1(const struct VPUNN::DPUWorkload &) const --> class VPUNN::VPUTensor", pybind11::arg("wl"));
		cl.def("check_workload_consistency", (void (VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>::*)(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &, const class VPUNN::IOperationDynamicConstraints &, struct VPUNN::SanityReport &) const) &VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::check_workload_consistency, "C++: VPUNN::DPU_ConfigurableOperationValidator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::check_workload_consistency(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &, const class VPUNN::IOperationDynamicConstraints &, struct VPUNN::SanityReport &) const --> void", pybind11::arg("w"), pybind11::arg("config"), pybind11::arg("operation_behaviour"), pybind11::arg("result"));
		cl.def("is_supported", (bool (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::is_supported, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::is_supported(enum VPUNN::VPUDevice) const --> bool", pybind11::arg("device"));
		cl.def("get_config", (const class VPUNN::IDeviceValidValues & (VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>,VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::*)(enum VPUNN::VPUDevice) const) &VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::get_config, "C++: VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>::get_config(enum VPUNN::VPUDevice) const --> const class VPUNN::IDeviceValidValues &", pybind11::return_value_policy::automatic, pybind11::arg("device"));
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>::*)(const struct VPUNN::DPUOperation &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory(const struct VPUNN::DPUOperation &) const --> struct VPUNN::MemorySize", pybind11::arg("w"));
		cl.def("compute_wl_memory", (struct VPUNN::MemorySize (VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues>>::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory, "C++: VPUNN::ContextualMemoryCalculator<VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints, VPUNN::DW_CONVOLUTION_Constraints, VPUNN::CM_CONVOLUTION_Constraints, VPUNN::ELTWISE_Constraints, VPUNN::MAXPOOL_Constraints>, VPUNN::VPU2_0_LayerOnTileValidValues, VPUNN::VPU2_7_LayerOnTileValidValues, VPUNN::VPU4_0_LayerOnTileValidValues> >::compute_wl_memory(const struct VPUNN::DPUWorkload &) const --> struct VPUNN::MemorySize", pybind11::arg("wl"));
	}
	{ // VPUNN::Preprocessing file: line:31
		pybind11::class_<VPUNN::Preprocessing<float>, std::shared_ptr<VPUNN::Preprocessing<float>>, PyCallBack_VPUNN_Preprocessing_float_t> cl(M("VPUNN"), "Preprocessing_float_t", "");
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_Preprocessing_float_t(); } ) );
		cl.def(pybind11::init<PyCallBack_VPUNN_Preprocessing_float_t const &>());
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::PreprocessingInserter file: line:162
		pybind11::class_<VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>, std::shared_ptr<VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>>, PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_PreprocessingLatest_float_t, VPUNN::Preprocessing<float>> cl(M("VPUNN"), "PreprocessingInserter_float_VPUNN_PreprocessingLatest_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>(); }, [](){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_PreprocessingLatest_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_PreprocessingLatest_float_t const &o){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_PreprocessingLatest_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> const &o){ return new VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>(o); } ) );
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: VPUNN_19.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <vector> // std::vector
#include <vpu/compatibility/types01.h> // VPUNN::Preprocessing_Interface01
#include <vpu/compatibility/types01.h> // VPUNN::Preprocessing_Interface10
#include <vpu/compatibility/types11.h> // VPUNN::Preprocessing_Interface11

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::PreprocessingInserter file: line:162
struct PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface01_float_t : public VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> {
	using VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::PreprocessingInserter;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

// VPUNN::PreprocessingInserter file: line:162
struct PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface10_float_t : public VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> {
	using VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::PreprocessingInserter;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

// VPUNN::PreprocessingInserter file: line:162
struct PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface11_float_t : public VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> {
	using VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::PreprocessingInserter;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

void bind_VPUNN_19(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::PreprocessingInserter file: line:162
		pybind11::class_<VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>, std::shared_ptr<VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>>, PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface01_float_t, VPUNN::Preprocessing<float>> cl(M("VPUNN"), "PreprocessingInserter_float_VPUNN_Preprocessing_Interface01_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>(); }, [](){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface01_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface01_float_t const &o){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface01_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> const &o){ return new VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>(o); } ) );
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::PreprocessingInserter file: line:162
		pybind11::class_<VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>, std::shared_ptr<VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>>, PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface10_float_t, VPUNN::Preprocessing<float>> cl(M("VPUNN"), "PreprocessingInserter_float_VPUNN_Preprocessing_Interface10_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>(); }, [](){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface10_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface10_float_t const &o){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface10_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> const &o){ return new VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>(o); } ) );
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::PreprocessingInserter file: line:162
		pybind11::class_<VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>, std::shared_ptr<VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>>, PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface11_float_t, VPUNN::Preprocessing<float>> cl(M("VPUNN"), "PreprocessingInserter_float_VPUNN_Preprocessing_Interface11_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>(); }, [](){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface11_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface11_float_t const &o){ return new PyCallBack_VPUNN_PreprocessingInserter_float_VPUNN_Preprocessing_Interface11_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> const &o){ return new VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>(o); } ) );
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: VPUNN_20.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::PreprocessingLatest file: line:304
struct PyCallBack_VPUNN_PreprocessingLatest_float_t : public VPUNN::PreprocessingLatest<float> {
	using VPUNN::PreprocessingLatest<float>::PreprocessingLatest;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingLatest<float> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::PreprocessingLatest<float> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

void bind_VPUNN_20(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::NNVersions file: line:293
	pybind11::enum_<VPUNN::NNVersions>(M("VPUNN"), "NNVersions", "enum for NN descriptor versions (input versions)")
		.value("VERSION_00_LATEST_NONE", VPUNN::NNVersions::VERSION_00_LATEST_NONE)
		.value("VERSION_01_BASE", VPUNN::NNVersions::VERSION_01_BASE)
		.value("VERSION_10_ENUMS_SAME", VPUNN::NNVersions::VERSION_10_ENUMS_SAME)
		.value("VERSION_11_VPU27_BETA", VPUNN::NNVersions::VERSION_11_VPU27_BETA);

;

	{ // VPUNN::PreprocessingLatest file: line:304
		pybind11::class_<VPUNN::PreprocessingLatest<float>, std::shared_ptr<VPUNN::PreprocessingLatest<float>>, PyCallBack_VPUNN_PreprocessingLatest_float_t, VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>> cl(M("VPUNN"), "PreprocessingLatest_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::PreprocessingLatest<float>(); }, [](){ return new PyCallBack_VPUNN_PreprocessingLatest_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_PreprocessingLatest_float_t const &o){ return new PyCallBack_VPUNN_PreprocessingLatest_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::PreprocessingLatest<float> const &o){ return new VPUNN::PreprocessingLatest<float>(o); } ) );
		cl.def_static("getInterfaceVersion", (int (*)()) &VPUNN::PreprocessingLatest<float>::getInterfaceVersion, "C++: VPUNN::PreprocessingLatest<float>::getInterfaceVersion() --> int");
		cl.def_static("implements_also_interface", (int (*)()) &VPUNN::PreprocessingLatest<float>::implements_also_interface, "C++: VPUNN::PreprocessingLatest<float>::implements_also_interface() --> int");
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::PreprocessingLatest<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::PreprocessingLatest<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::PreprocessingLatest<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: VPUNN_21.cpp
#include <functional> // std::less
#include <iterator> // __gnu_cxx::__normal_iterator
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::ActivationFunction
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::DataType
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::ExecutionMode
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::Layout
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::MemoryLocation
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::Operation
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::Swizzling
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::VPUDevice
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::VPUSubsystem
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::convert
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::mapFromText
#include <vpu/compatibility/types01.h> // VPUNN::intf_01::mapToText

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_21(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::intf_01::mapFromText() file:vpu/compatibility/types01.h line:38
	M("VPUNN::intf_01").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_01::mapFromText<VPUNN::intf_01::VPUDevice>, "C++: VPUNN::intf_01::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::mapFromText() file:vpu/compatibility/types01.h line:38
	M("VPUNN::intf_01").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_01::mapFromText<VPUNN::intf_01::Operation>, "C++: VPUNN::intf_01::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::mapFromText() file:vpu/compatibility/types01.h line:38
	M("VPUNN::intf_01").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_01::mapFromText<VPUNN::intf_01::DataType>, "C++: VPUNN::intf_01::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::mapFromText() file:vpu/compatibility/types01.h line:38
	M("VPUNN::intf_01").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_01::mapFromText<VPUNN::intf_01::ExecutionMode>, "C++: VPUNN::intf_01::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::mapFromText() file:vpu/compatibility/types01.h line:38
	M("VPUNN::intf_01").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_01::mapFromText<VPUNN::intf_01::ActivationFunction>, "C++: VPUNN::intf_01::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::mapFromText() file:vpu/compatibility/types01.h line:38
	M("VPUNN::intf_01").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_01::mapFromText<VPUNN::intf_01::Swizzling>, "C++: VPUNN::intf_01::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::VPUDevice file:vpu/compatibility/types01.h line:47
	pybind11::enum_<VPUNN::intf_01::VPUDevice>(M("VPUNN::intf_01"), "VPUDevice", "VPU IP generations\n\n ")
		.value("VPU_2_0", VPUNN::intf_01::VPUDevice::VPU_2_0)
		.value("VPU_2_1", VPUNN::intf_01::VPUDevice::VPU_2_1)
		.value("VPU_2_7", VPUNN::intf_01::VPUDevice::VPU_2_7)
		.value("VPU_4_0", VPUNN::intf_01::VPUDevice::VPU_4_0)
		.value("__size", VPUNN::intf_01::VPUDevice::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:55
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::VPUDevice>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::DataType file:vpu/compatibility/types01.h line:63
	pybind11::enum_<VPUNN::intf_01::DataType>(M("VPUNN::intf_01"), "DataType", "Supported Datatypes\n\n ")
		.value("UINT8", VPUNN::intf_01::DataType::UINT8)
		.value("INT8", VPUNN::intf_01::DataType::INT8)
		.value("FLOAT16", VPUNN::intf_01::DataType::FLOAT16)
		.value("BFLOAT16", VPUNN::intf_01::DataType::BFLOAT16)
		.value("__size", VPUNN::intf_01::DataType::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:71
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::DataType>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::Operation file:vpu/compatibility/types01.h line:79
	pybind11::enum_<VPUNN::intf_01::Operation>(M("VPUNN::intf_01"), "Operation", "HW operations\n\n ")
		.value("CONVOLUTION", VPUNN::intf_01::Operation::CONVOLUTION)
		.value("DW_CONVOLUTION", VPUNN::intf_01::Operation::DW_CONVOLUTION)
		.value("ELTWISE", VPUNN::intf_01::Operation::ELTWISE)
		.value("MAXPOOL", VPUNN::intf_01::Operation::MAXPOOL)
		.value("AVEPOOL", VPUNN::intf_01::Operation::AVEPOOL)
		.value("CM_CONVOLUTION", VPUNN::intf_01::Operation::CM_CONVOLUTION)
		.value("__size", VPUNN::intf_01::Operation::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:86
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::Operation>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::ActivationFunction file:vpu/compatibility/types01.h line:94
	pybind11::enum_<VPUNN::intf_01::ActivationFunction>(M("VPUNN::intf_01"), "ActivationFunction", "Supported activation functions\n\n ")
		.value("NONE", VPUNN::intf_01::ActivationFunction::NONE)
		.value("RELU", VPUNN::intf_01::ActivationFunction::RELU)
		.value("LRELU", VPUNN::intf_01::ActivationFunction::LRELU)
		.value("ADD", VPUNN::intf_01::ActivationFunction::ADD)
		.value("SUB", VPUNN::intf_01::ActivationFunction::SUB)
		.value("MULT", VPUNN::intf_01::ActivationFunction::MULT)
		.value("__size", VPUNN::intf_01::ActivationFunction::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:101
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::ActivationFunction>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::Swizzling file:vpu/compatibility/types01.h line:109
	pybind11::enum_<VPUNN::intf_01::Swizzling>(M("VPUNN::intf_01"), "Swizzling", "Swizzling keys\n\n ")
		.value("KEY_0", VPUNN::intf_01::Swizzling::KEY_0)
		.value("KEY_1", VPUNN::intf_01::Swizzling::KEY_1)
		.value("KEY_2", VPUNN::intf_01::Swizzling::KEY_2)
		.value("KEY_3", VPUNN::intf_01::Swizzling::KEY_3)
		.value("KEY_4", VPUNN::intf_01::Swizzling::KEY_4)
		.value("KEY_5", VPUNN::intf_01::Swizzling::KEY_5)
		.value("__size", VPUNN::intf_01::Swizzling::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:115
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::Swizzling>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::ExecutionMode file:vpu/compatibility/types01.h line:122
	pybind11::enum_<VPUNN::intf_01::ExecutionMode>(M("VPUNN::intf_01"), "ExecutionMode", "DPU execution modes\n\n ")
		.value("VECTOR", VPUNN::intf_01::ExecutionMode::VECTOR)
		.value("MATRIX", VPUNN::intf_01::ExecutionMode::MATRIX)
		.value("VECTOR_FP16", VPUNN::intf_01::ExecutionMode::VECTOR_FP16)
		.value("CUBOID_16x16", VPUNN::intf_01::ExecutionMode::CUBOID_16x16)
		.value("CUBOID_8x16", VPUNN::intf_01::ExecutionMode::CUBOID_8x16)
		.value("CUBOID_4x16", VPUNN::intf_01::ExecutionMode::CUBOID_4x16)
		.value("__size", VPUNN::intf_01::ExecutionMode::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:129
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::ExecutionMode>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::Layout file:vpu/compatibility/types01.h line:137
	pybind11::enum_<VPUNN::intf_01::Layout>(M("VPUNN::intf_01"), "Layout", "Data layout\n\n ")
		.value("ZMAJOR", VPUNN::intf_01::Layout::ZMAJOR)
		.value("CMAJOR", VPUNN::intf_01::Layout::CMAJOR)
		.value("__size", VPUNN::intf_01::Layout::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:143
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::Layout>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::MemoryLocation file:vpu/compatibility/types01.h line:151
	pybind11::enum_<VPUNN::intf_01::MemoryLocation>(M("VPUNN::intf_01"), "MemoryLocation", "Memory locations\n\n ")
		.value("DRAM", VPUNN::intf_01::MemoryLocation::DRAM)
		.value("CMX", VPUNN::intf_01::MemoryLocation::CMX)
		.value("CSRAM", VPUNN::intf_01::MemoryLocation::CSRAM)
		.value("UPA", VPUNN::intf_01::MemoryLocation::UPA)
		.value("__size", VPUNN::intf_01::MemoryLocation::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:159
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::MemoryLocation>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::VPUSubsystem file:vpu/compatibility/types01.h line:167
	pybind11::enum_<VPUNN::intf_01::VPUSubsystem>(M("VPUNN::intf_01"), "VPUSubsystem", "VPU Hw subsystem\n\n ")
		.value("VPU_DPU", VPUNN::intf_01::VPUSubsystem::VPU_DPU)
		.value("VPU_SHV", VPUNN::intf_01::VPUSubsystem::VPU_SHV)
		.value("VPU_DMA", VPUNN::intf_01::VPUSubsystem::VPU_DMA)
		.value("VPU_CPU", VPUNN::intf_01::VPUSubsystem::VPU_CPU)
		.value("VPU_CMX", VPUNN::intf_01::VPUSubsystem::VPU_CMX)
		.value("__size", VPUNN::intf_01::VPUSubsystem::__size);

;

	// VPUNN::intf_01::mapToText() file:vpu/compatibility/types01.h line:174
	M("VPUNN::intf_01").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_01::mapToText<VPUNN::intf_01::VPUSubsystem>, "C++: VPUNN::intf_01::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_01::convert(enum VPUNN::DataType) file:vpu/compatibility/types01.h line:184
	M("VPUNN::intf_01").def("convert", (enum VPUNN::intf_01::DataType (*)(enum VPUNN::DataType)) &VPUNN::intf_01::convert<VPUNN::intf_01::DataType,VPUNN::DataType>, "C++: VPUNN::intf_01::convert(enum VPUNN::DataType) --> enum VPUNN::intf_01::DataType", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_01::convert(enum VPUNN::VPUDevice) file:vpu/compatibility/types01.h line:184
	M("VPUNN::intf_01").def("convert", (enum VPUNN::intf_01::VPUDevice (*)(enum VPUNN::VPUDevice)) &VPUNN::intf_01::convert<VPUNN::intf_01::VPUDevice,VPUNN::VPUDevice>, "C++: VPUNN::intf_01::convert(enum VPUNN::VPUDevice) --> enum VPUNN::intf_01::VPUDevice", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_01::convert(enum VPUNN::Operation) file:vpu/compatibility/types01.h line:184
	M("VPUNN::intf_01").def("convert", (enum VPUNN::intf_01::Operation (*)(enum VPUNN::Operation)) &VPUNN::intf_01::convert<VPUNN::intf_01::Operation,VPUNN::Operation>, "C++: VPUNN::intf_01::convert(enum VPUNN::Operation) --> enum VPUNN::intf_01::Operation", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_01::convert(enum VPUNN::ExecutionMode) file:vpu/compatibility/types01.h line:184
	M("VPUNN::intf_01").def("convert", (enum VPUNN::intf_01::ExecutionMode (*)(enum VPUNN::ExecutionMode)) &VPUNN::intf_01::convert<VPUNN::intf_01::ExecutionMode,VPUNN::ExecutionMode>, "C++: VPUNN::intf_01::convert(enum VPUNN::ExecutionMode) --> enum VPUNN::intf_01::ExecutionMode", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_01::convert(enum VPUNN::ActivationFunction) file:vpu/compatibility/types01.h line:184
	M("VPUNN::intf_01").def("convert", (enum VPUNN::intf_01::ActivationFunction (*)(enum VPUNN::ActivationFunction)) &VPUNN::intf_01::convert<VPUNN::intf_01::ActivationFunction,VPUNN::ActivationFunction>, "C++: VPUNN::intf_01::convert(enum VPUNN::ActivationFunction) --> enum VPUNN::intf_01::ActivationFunction", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_01::convert(enum VPUNN::Swizzling) file:vpu/compatibility/types01.h line:184
	M("VPUNN::intf_01").def("convert", (enum VPUNN::intf_01::Swizzling (*)(enum VPUNN::Swizzling)) &VPUNN::intf_01::convert<VPUNN::intf_01::Swizzling,VPUNN::Swizzling>, "C++: VPUNN::intf_01::convert(enum VPUNN::Swizzling) --> enum VPUNN::intf_01::Swizzling", pybind11::arg("present_day_value_type"));

}


// File: VPUNN_22.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <vector> // std::vector
#include <vpu/compatibility/types01.h> // VPUNN::Preprocessing_Interface01
#include <vpu/compatibility/types01.h> // VPUNN::Preprocessing_Interface10

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::Preprocessing_Interface01 file:vpu/compatibility/types01.h line:229
struct PyCallBack_VPUNN_Preprocessing_Interface01_float_t : public VPUNN::Preprocessing_Interface01<float> {
	using VPUNN::Preprocessing_Interface01<float>::Preprocessing_Interface01;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing_Interface01<float> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing_Interface01<float> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

// VPUNN::Preprocessing_Interface10 file:vpu/compatibility/types01.h line:327
struct PyCallBack_VPUNN_Preprocessing_Interface10_float_t : public VPUNN::Preprocessing_Interface10<float> {
	using VPUNN::Preprocessing_Interface10<float>::Preprocessing_Interface10;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing_Interface10<float> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing_Interface10<float> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

void bind_VPUNN_22(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::Preprocessing_Interface01 file:vpu/compatibility/types01.h line:229
		pybind11::class_<VPUNN::Preprocessing_Interface01<float>, std::shared_ptr<VPUNN::Preprocessing_Interface01<float>>, PyCallBack_VPUNN_Preprocessing_Interface01_float_t, VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>> cl(M("VPUNN"), "Preprocessing_Interface01_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::Preprocessing_Interface01<float>(); }, [](){ return new PyCallBack_VPUNN_Preprocessing_Interface01_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_Preprocessing_Interface01_float_t const &o){ return new PyCallBack_VPUNN_Preprocessing_Interface01_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::Preprocessing_Interface01<float> const &o){ return new VPUNN::Preprocessing_Interface01<float>(o); } ) );
		cl.def_static("getInterfaceVersion", (int (*)()) &VPUNN::Preprocessing_Interface01<float>::getInterfaceVersion, "C++: VPUNN::Preprocessing_Interface01<float>::getInterfaceVersion() --> int");
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface01<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface01<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface01<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::Preprocessing_Interface10 file:vpu/compatibility/types01.h line:327
		pybind11::class_<VPUNN::Preprocessing_Interface10<float>, std::shared_ptr<VPUNN::Preprocessing_Interface10<float>>, PyCallBack_VPUNN_Preprocessing_Interface10_float_t, VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>> cl(M("VPUNN"), "Preprocessing_Interface10_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::Preprocessing_Interface10<float>(); }, [](){ return new PyCallBack_VPUNN_Preprocessing_Interface10_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_Preprocessing_Interface10_float_t const &o){ return new PyCallBack_VPUNN_Preprocessing_Interface10_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::Preprocessing_Interface10<float> const &o){ return new VPUNN::Preprocessing_Interface10<float>(o); } ) );
		cl.def_static("getInterfaceVersion", (int (*)()) &VPUNN::Preprocessing_Interface10<float>::getInterfaceVersion, "C++: VPUNN::Preprocessing_Interface10<float>::getInterfaceVersion() --> int");
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface10<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface10<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface10<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: VPUNN_23.cpp
#include <functional> // std::less
#include <iterator> // __gnu_cxx::__normal_iterator
#include <map> // std::_Rb_tree_const_iterator
#include <map> // std::_Rb_tree_iterator
#include <map> // std::map
#include <memory> // std::allocator
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <utility> // std::pair
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ActivationFunction
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::DataType
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ExecutionMode
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ISIStrategy
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Layout
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::MemoryLocation
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Operation
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Swizzling
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::VPUDevice
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::VPUSubsystem
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::convert
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::mapFromText
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::mapToText

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_23(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::VPUDevice>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::Operation>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::DataType>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::Layout>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::ExecutionMode>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::Swizzling>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::mapFromText() file:vpu/compatibility/types11.h line:45
	M("VPUNN::intf_11").def("mapFromText", (const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > & (*)()) &VPUNN::intf_11::mapFromText<VPUNN::intf_11::ISIStrategy>, "C++: VPUNN::intf_11::mapFromText() --> const class std::map<const std::string, const int, struct std::less<const std::string >, class std::allocator<struct std::pair<const std::string, const int> > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::VPUDevice file:vpu/compatibility/types11.h line:54
	pybind11::enum_<VPUNN::intf_11::VPUDevice>(M("VPUNN::intf_11"), "VPUDevice", "VPU IP generations\n\n ")
		.value("VPU_2_0", VPUNN::intf_11::VPUDevice::VPU_2_0)
		.value("VPU_2_1", VPUNN::intf_11::VPUDevice::VPU_2_1)
		.value("VPU_2_7", VPUNN::intf_11::VPUDevice::VPU_2_7)
		.value("VPU_4_0", VPUNN::intf_11::VPUDevice::VPU_4_0)
		.value("__size", VPUNN::intf_11::VPUDevice::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:58
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::VPUDevice>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::DataType file:vpu/compatibility/types11.h line:66
	pybind11::enum_<VPUNN::intf_11::DataType>(M("VPUNN::intf_11"), "DataType", "Supported Datatypes\n\n ")
		.value("UINT8", VPUNN::intf_11::DataType::UINT8)
		.value("INT8", VPUNN::intf_11::DataType::INT8)
		.value("FLOAT16", VPUNN::intf_11::DataType::FLOAT16)
		.value("BFLOAT16", VPUNN::intf_11::DataType::BFLOAT16)
		.value("__size", VPUNN::intf_11::DataType::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:74
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::DataType>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::Operation file:vpu/compatibility/types11.h line:82
	pybind11::enum_<VPUNN::intf_11::Operation>(M("VPUNN::intf_11"), "Operation", "HW operations\n\n ")
		.value("CONVOLUTION", VPUNN::intf_11::Operation::CONVOLUTION)
		.value("DW_CONVOLUTION", VPUNN::intf_11::Operation::DW_CONVOLUTION)
		.value("ELTWISE", VPUNN::intf_11::Operation::ELTWISE)
		.value("MAXPOOL", VPUNN::intf_11::Operation::MAXPOOL)
		.value("AVEPOOL", VPUNN::intf_11::Operation::AVEPOOL)
		.value("CM_CONVOLUTION", VPUNN::intf_11::Operation::CM_CONVOLUTION)
		.value("__size", VPUNN::intf_11::Operation::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:89
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::Operation>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::ActivationFunction file:vpu/compatibility/types11.h line:96
	pybind11::enum_<VPUNN::intf_11::ActivationFunction>(M("VPUNN::intf_11"), "ActivationFunction", "Supported activation functions\n\n ")
		.value("NONE", VPUNN::intf_11::ActivationFunction::NONE)
		.value("RELU", VPUNN::intf_11::ActivationFunction::RELU)
		.value("LRELU", VPUNN::intf_11::ActivationFunction::LRELU)
		.value("ADD", VPUNN::intf_11::ActivationFunction::ADD)
		.value("SUB", VPUNN::intf_11::ActivationFunction::SUB)
		.value("MULT", VPUNN::intf_11::ActivationFunction::MULT)
		.value("__size", VPUNN::intf_11::ActivationFunction::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:103
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::ActivationFunction>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::Swizzling file:vpu/compatibility/types11.h line:110
	pybind11::enum_<VPUNN::intf_11::Swizzling>(M("VPUNN::intf_11"), "Swizzling", "Swizzling keys\n\n ")
		.value("KEY_0", VPUNN::intf_11::Swizzling::KEY_0)
		.value("KEY_1", VPUNN::intf_11::Swizzling::KEY_1)
		.value("KEY_2", VPUNN::intf_11::Swizzling::KEY_2)
		.value("KEY_3", VPUNN::intf_11::Swizzling::KEY_3)
		.value("KEY_4", VPUNN::intf_11::Swizzling::KEY_4)
		.value("KEY_5", VPUNN::intf_11::Swizzling::KEY_5)
		.value("__size", VPUNN::intf_11::Swizzling::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:116
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::Swizzling>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::ExecutionMode file:vpu/compatibility/types11.h line:124
	pybind11::enum_<VPUNN::intf_11::ExecutionMode>(M("VPUNN::intf_11"), "ExecutionMode", "DPU execution modes\n VECTOR, MATRIX, VECTOR_FP16,are DELETED for VPU2.7\n\n ")
		.value("CUBOID_16x16", VPUNN::intf_11::ExecutionMode::CUBOID_16x16)
		.value("CUBOID_8x16", VPUNN::intf_11::ExecutionMode::CUBOID_8x16)
		.value("CUBOID_4x16", VPUNN::intf_11::ExecutionMode::CUBOID_4x16)
		.value("__size", VPUNN::intf_11::ExecutionMode::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:131
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::ExecutionMode>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::Layout file:vpu/compatibility/types11.h line:145
	pybind11::enum_<VPUNN::intf_11::Layout>(M("VPUNN::intf_11"), "Layout", "Data layout\n\n ZMAJOR and CMAJOR are coming from VPU2.0, were DELETED!\n\n XYZ, XZY, YXZ, YZX, ZXY, ZYX  were introduced for 2.7\n They are to interpreted as from  innermost to outermost dimension of the tensor\n eg: XYZ is NCHW; N=Batch is always outermost, then channels (Z), height (Y), width (X)\n\n ")
		.value("XYZ", VPUNN::intf_11::Layout::XYZ)
		.value("XZY", VPUNN::intf_11::Layout::XZY)
		.value("YXZ", VPUNN::intf_11::Layout::YXZ)
		.value("YZX", VPUNN::intf_11::Layout::YZX)
		.value("ZXY", VPUNN::intf_11::Layout::ZXY)
		.value("ZYX", VPUNN::intf_11::Layout::ZYX)
		.value("INVALID", VPUNN::intf_11::Layout::INVALID)
		.value("__size", VPUNN::intf_11::Layout::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:150
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::Layout>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::ISIStrategy file:vpu/compatibility/types11.h line:155
	pybind11::enum_<VPUNN::intf_11::ISIStrategy>(M("VPUNN::intf_11"), "ISIStrategy", "ISI_Strategy")
		.value("CLUSTERING", VPUNN::intf_11::ISIStrategy::CLUSTERING)
		.value("SPLIT_OVER_H", VPUNN::intf_11::ISIStrategy::SPLIT_OVER_H)
		.value("SPLIT_OVER_K", VPUNN::intf_11::ISIStrategy::SPLIT_OVER_K)
		.value("__size", VPUNN::intf_11::ISIStrategy::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:162
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::ISIStrategy>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::MemoryLocation file:vpu/compatibility/types11.h line:170
	pybind11::enum_<VPUNN::intf_11::MemoryLocation>(M("VPUNN::intf_11"), "MemoryLocation", "Memory locations\n\n ")
		.value("DRAM", VPUNN::intf_11::MemoryLocation::DRAM)
		.value("CMX", VPUNN::intf_11::MemoryLocation::CMX)
		.value("CSRAM", VPUNN::intf_11::MemoryLocation::CSRAM)
		.value("UPA", VPUNN::intf_11::MemoryLocation::UPA)
		.value("__size", VPUNN::intf_11::MemoryLocation::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:178
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::MemoryLocation>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::VPUSubsystem file:vpu/compatibility/types11.h line:186
	pybind11::enum_<VPUNN::intf_11::VPUSubsystem>(M("VPUNN::intf_11"), "VPUSubsystem", "VPU Hw subsystem\n\n ")
		.value("VPU_DPU", VPUNN::intf_11::VPUSubsystem::VPU_DPU)
		.value("VPU_SHV", VPUNN::intf_11::VPUSubsystem::VPU_SHV)
		.value("VPU_DMA", VPUNN::intf_11::VPUSubsystem::VPU_DMA)
		.value("VPU_CPU", VPUNN::intf_11::VPUSubsystem::VPU_CPU)
		.value("VPU_CMX", VPUNN::intf_11::VPUSubsystem::VPU_CMX)
		.value("__size", VPUNN::intf_11::VPUSubsystem::__size);

;

	// VPUNN::intf_11::mapToText() file:vpu/compatibility/types11.h line:193
	M("VPUNN::intf_11").def("mapToText", (const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > & (*)()) &VPUNN::intf_11::mapToText<VPUNN::intf_11::VPUSubsystem>, "C++: VPUNN::intf_11::mapToText() --> const class std::map<int, const std::string, struct std::less<int>, class std::allocator<struct std::pair<const int, const std::string > > > &", pybind11::return_value_policy::automatic);

	// VPUNN::intf_11::convert(enum VPUNN::DataType) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::DataType (*)(enum VPUNN::DataType)) &VPUNN::intf_11::convert<VPUNN::intf_11::DataType,VPUNN::DataType>, "C++: VPUNN::intf_11::convert(enum VPUNN::DataType) --> enum VPUNN::intf_11::DataType", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_11::convert(enum VPUNN::Layout) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::Layout (*)(enum VPUNN::Layout)) &VPUNN::intf_11::convert<VPUNN::intf_11::Layout,VPUNN::Layout>, "C++: VPUNN::intf_11::convert(enum VPUNN::Layout) --> enum VPUNN::intf_11::Layout", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_11::convert(enum VPUNN::VPUDevice) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::VPUDevice (*)(enum VPUNN::VPUDevice)) &VPUNN::intf_11::convert<VPUNN::intf_11::VPUDevice,VPUNN::VPUDevice>, "C++: VPUNN::intf_11::convert(enum VPUNN::VPUDevice) --> enum VPUNN::intf_11::VPUDevice", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_11::convert(enum VPUNN::Operation) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::Operation (*)(enum VPUNN::Operation)) &VPUNN::intf_11::convert<VPUNN::intf_11::Operation,VPUNN::Operation>, "C++: VPUNN::intf_11::convert(enum VPUNN::Operation) --> enum VPUNN::intf_11::Operation", pybind11::arg("present_day_value_type"));

}


// File: VPUNN_24.cpp
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ExecutionMode
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::ISIStrategy
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::Swizzling
#include <vpu/compatibility/types11.h> // VPUNN::intf_11::convert

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_24(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::intf_11::convert(enum VPUNN::ExecutionMode) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::ExecutionMode (*)(enum VPUNN::ExecutionMode)) &VPUNN::intf_11::convert<VPUNN::intf_11::ExecutionMode,VPUNN::ExecutionMode>, "C++: VPUNN::intf_11::convert(enum VPUNN::ExecutionMode) --> enum VPUNN::intf_11::ExecutionMode", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_11::convert(enum VPUNN::Swizzling) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::Swizzling (*)(enum VPUNN::Swizzling)) &VPUNN::intf_11::convert<VPUNN::intf_11::Swizzling,VPUNN::Swizzling>, "C++: VPUNN::intf_11::convert(enum VPUNN::Swizzling) --> enum VPUNN::intf_11::Swizzling", pybind11::arg("present_day_value_type"));

	// VPUNN::intf_11::convert(enum VPUNN::ISIStrategy) file:vpu/compatibility/types11.h line:205
	M("VPUNN::intf_11").def("convert", (enum VPUNN::intf_11::ISIStrategy (*)(enum VPUNN::ISIStrategy)) &VPUNN::intf_11::convert<VPUNN::intf_11::ISIStrategy,VPUNN::ISIStrategy>, "C++: VPUNN::intf_11::convert(enum VPUNN::ISIStrategy) --> enum VPUNN::intf_11::ISIStrategy", pybind11::arg("present_day_value_type"));

}


// File: VPUNN_25.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <vector> // std::vector
#include <vpu/compatibility/types11.h> // VPUNN::Preprocessing_Interface11

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::Preprocessing_Interface11 file:vpu/compatibility/types11.h line:248
struct PyCallBack_VPUNN_Preprocessing_Interface11_float_t : public VPUNN::Preprocessing_Interface11<float> {
	using VPUNN::Preprocessing_Interface11<float>::Preprocessing_Interface11;

	int interface_version() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing_Interface11<float> *>(this), "interface_version");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::override_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return PreprocessingInserter::interface_version();
	}
	using _binder_ret_0 = const class std::vector<float, class std::allocator<float> > &;
	_binder_ret_0 generate_descriptor(const struct VPUNN::DPUWorkload & a0, unsigned long & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::Preprocessing_Interface11<float> *>(this), "generate_descriptor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return PreprocessingInserter::generate_descriptor(a0, a1);
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_4718_4715_t : public VPUNN::SHVActivation<4718,4715> {
	using VPUNN::SHVActivation<4718,4715>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<4718,4715> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_441_5067_t : public VPUNN::SHVActivation<441,5067> {
	using VPUNN::SHVActivation<441,5067>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<441,5067> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

void bind_VPUNN_25(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::Preprocessing_Interface11 file:vpu/compatibility/types11.h line:248
		pybind11::class_<VPUNN::Preprocessing_Interface11<float>, std::shared_ptr<VPUNN::Preprocessing_Interface11<float>>, PyCallBack_VPUNN_Preprocessing_Interface11_float_t, VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>> cl(M("VPUNN"), "Preprocessing_Interface11_float_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::Preprocessing_Interface11<float>(); }, [](){ return new PyCallBack_VPUNN_Preprocessing_Interface11_float_t(); } ) );
		cl.def( pybind11::init( [](PyCallBack_VPUNN_Preprocessing_Interface11_float_t const &o){ return new PyCallBack_VPUNN_Preprocessing_Interface11_float_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::Preprocessing_Interface11<float> const &o){ return new VPUNN::Preprocessing_Interface11<float>(o); } ) );
		cl.def_static("getInterfaceVersion", (int (*)()) &VPUNN::Preprocessing_Interface11<float>::getInterfaceVersion, "C++: VPUNN::Preprocessing_Interface11<float>::getInterfaceVersion() --> int");
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, unsigned int const & a1) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("transform", [](VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>> &o, const struct VPUNN::DPUWorkload & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("interface_version", (int (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::*)() const) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::interface_version, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::interface_version() const --> int");
		cl.def("generate_descriptor", (const class std::vector<float, class std::allocator<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::*)(const struct VPUNN::DPUWorkload &, unsigned long &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::generate_descriptor, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::generate_descriptor(const struct VPUNN::DPUWorkload &, unsigned long &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"), pybind11::arg("debug_offset"));
		cl.def("assign", (class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > & (VPUNN::PreprocessingInserter<float,VPUNN::Preprocessing_Interface11<float>>::*)(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > &)) &VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::operator=, "C++: VPUNN::PreprocessingInserter<float, VPUNN::Preprocessing_Interface11<float> >::operator=(const class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > &) --> class VPUNN::PreprocessingInserter<float, class VPUNN::Preprocessing_Interface11<float> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("interface_version", (int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::interface_version, "C++: VPUNN::Preprocessing<float>::interface_version() const --> int");
		cl.def("output_size", (unsigned int (VPUNN::Preprocessing<float>::*)() const) &VPUNN::Preprocessing<float>::output_size, "C++: VPUNN::Preprocessing<float>::output_size() const --> unsigned int");
		cl.def("set_size", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_size, "C++: VPUNN::Preprocessing<float>::set_size(unsigned long) --> void", pybind11::arg("size"));
		cl.def("reset", (void (VPUNN::Preprocessing<float>::*)()) &VPUNN::Preprocessing<float>::reset, "C++: VPUNN::Preprocessing<float>::reset() --> void");
		cl.def("set_probable_batch", (void (VPUNN::Preprocessing<float>::*)(unsigned long)) &VPUNN::Preprocessing<float>::set_probable_batch, "C++: VPUNN::Preprocessing<float>::set_probable_batch(unsigned long) --> void", pybind11::arg("batch_size"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const struct VPUNN::DPUWorkload &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workload"));
		cl.def("transform", [](VPUNN::Preprocessing<float> &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> const std::vector<float, class std::allocator<float> > & { return o.transform(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("transform", (const class std::vector<float, class std::allocator<float> > & (VPUNN::Preprocessing<float>::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int)) &VPUNN::Preprocessing<float>::transform, "C++: VPUNN::Preprocessing<float>::transform(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, unsigned int) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"), pybind11::arg("pad"));
		cl.def("assign", (class VPUNN::Preprocessing<float> & (VPUNN::Preprocessing<float>::*)(const class VPUNN::Preprocessing<float> &)) &VPUNN::Preprocessing<float>::operator=, "C++: VPUNN::Preprocessing<float>::operator=(const class VPUNN::Preprocessing<float> &) --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::RuntimeProcessingFactory file: line:29
		pybind11::class_<VPUNN::RuntimeProcessingFactory, std::shared_ptr<VPUNN::RuntimeProcessingFactory>> cl(M("VPUNN"), "RuntimeProcessingFactory", "Provides processing related objects based on context\n\n The provided objects may be bounded(lifespan) to this instance");
		cl.def( pybind11::init( [](VPUNN::RuntimeProcessingFactory const &o){ return new VPUNN::RuntimeProcessingFactory(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::RuntimeProcessingFactory(); } ) );
		cl.def("exists_preprocessing", (bool (VPUNN::RuntimeProcessingFactory::*)(int) const) &VPUNN::RuntimeProcessingFactory::exists_preprocessing, "True if a preprocessor exists for required/interrogated version\n\nC++: VPUNN::RuntimeProcessingFactory::exists_preprocessing(int) const --> bool", pybind11::arg("input_version"));
		cl.def("make_preprocessing", (class VPUNN::Preprocessing<float> & (VPUNN::RuntimeProcessingFactory::*)(int) const) &VPUNN::RuntimeProcessingFactory::make_preprocessing, "provides a preprocessor for the required interface\n The provided preprocessor is owned by this class.\n For NOW multiple requests for the same version will provide the same object, the factory just shares the\n preprocessors , does not create a new one for each request\n \n\n desired interface version\n \n\n the preprocessor object to be used (shared)\n \n\n out_of_range in case the version is not supported\n\nC++: VPUNN::RuntimeProcessingFactory::make_preprocessing(int) const --> class VPUNN::Preprocessing<float> &", pybind11::return_value_policy::automatic, pybind11::arg("version"));
	}
	{ // VPUNN::FirstDegreeEquation file: line:32
		pybind11::class_<VPUNN::FirstDegreeEquation, std::shared_ptr<VPUNN::FirstDegreeEquation>> cl(M("VPUNN"), "FirstDegreeEquation", "Defines the structure of the first degree equation of the line for each shave operation.\n The ecuation is slope_ * size + intercept_\n \n\n is defined by time in us divided by size of output bytes\n \n\n is defined  by time in us");
		cl.def( pybind11::init( [](){ return new VPUNN::FirstDegreeEquation(); } ) );
		cl.def( pybind11::init( [](VPUNN::FirstDegreeEquation const &o){ return new VPUNN::FirstDegreeEquation(o); } ) );
		cl.def_readwrite("slope_", &VPUNN::FirstDegreeEquation::slope_);
		cl.def_readwrite("intercept_", &VPUNN::FirstDegreeEquation::intercept_);
		cl.def("__call__", (float (VPUNN::FirstDegreeEquation::*)(const int &) const) &VPUNN::FirstDegreeEquation::operator(), "Overload for operator() which calculates the time based on the first degree equation of the Shave\n activation and the size in bytes of the output\n\n \n of output in bytes\n \n\n the time in us\n\nC++: VPUNN::FirstDegreeEquation::operator()(const int &) const --> float", pybind11::arg("size"));
	}
	{ // VPUNN::ShaveModel1to1 file: line:47
		pybind11::class_<VPUNN::ShaveModel1to1, std::shared_ptr<VPUNN::ShaveModel1to1>> cl(M("VPUNN"), "ShaveModel1to1", "");
		cl.def( pybind11::init<enum VPUNN::DataType, float, float, float, float, unsigned int, unsigned int, unsigned int, unsigned int>(), pybind11::arg("dtype"), pybind11::arg("slope"), pybind11::arg("intercept"), pybind11::arg("offset_scalar"), pybind11::arg("offset_unroll"), pybind11::arg("VectorSize"), pybind11::arg("UnrollSize"), pybind11::arg("DpuFreq"), pybind11::arg("ShvFreq") );

		cl.def( pybind11::init( [](VPUNN::ShaveModel1to1 const &o){ return new VPUNN::ShaveModel1to1(o); } ) );
		cl.def("is_in_first_block_of_operations", (bool (VPUNN::ShaveModel1to1::*)(const int &) const) &VPUNN::ShaveModel1to1::is_in_first_block_of_operations, "Checks if the op_count is in the first block or not\n\n \n numbers of ops required\n \n\n rather is in the first block or not\n\nC++: VPUNN::ShaveModel1to1::is_in_first_block_of_operations(const int &) const --> bool", pybind11::arg("op_count"));
		cl.def("is_scalar_value", (bool (VPUNN::ShaveModel1to1::*)(const int &) const) &VPUNN::ShaveModel1to1::is_scalar_value, "A rule that determines if we have to add scalar, when the given value is in the interior of the block\n based on the number of operations\n\n \n numbers of ops required\n \n\n rather is in the first block or not\n\nC++: VPUNN::ShaveModel1to1::is_scalar_value(const int &) const --> bool", pybind11::arg("op_count"));
		cl.def("is_first_value_in_block", (bool (VPUNN::ShaveModel1to1::*)(const int &) const) &VPUNN::ShaveModel1to1::is_first_value_in_block, "Determines if the given op_count is the first in the block or not\n x_value = block_size first in block for loop unroll\n\n \n numbers of ops required\n \n\n rather is in the first block or not\n\nC++: VPUNN::ShaveModel1to1::is_first_value_in_block(const int &) const --> bool", pybind11::arg("op_count"));
		cl.def("getMicroSeconds", (float (VPUNN::ShaveModel1to1::*)(const int &) const) &VPUNN::ShaveModel1to1::getMicroSeconds, "Get the time in us for the the activation based on the output size in bytes\n\n \n the time in us\n\nC++: VPUNN::ShaveModel1to1::getMicroSeconds(const int &) const --> float", pybind11::arg("output_size_bytes"));
		cl.def("getDPUCycles", (unsigned int (VPUNN::ShaveModel1to1::*)(const int) const) &VPUNN::ShaveModel1to1::getDPUCycles, "Determines the number of cycles related to the profiling DPU freq, based on the size of output\n\n \n the number of cycles required based on CyclesInterfaceType\n\nC++: VPUNN::ShaveModel1to1::getDPUCycles(const int) const --> unsigned int", pybind11::arg("output_size_bytes"));
		cl.def("getDPUCycles", (unsigned int (VPUNN::ShaveModel1to1::*)(const int, const int) const) &VPUNN::ShaveModel1to1::getDPUCycles, "Determines the number of cycles related to the input value of a DPU freq given as a parameter based on the\n size of output\n\n \n the number of cycles required based on CyclesInterfaceType\n\nC++: VPUNN::ShaveModel1to1::getDPUCycles(const int, const int) const --> unsigned int", pybind11::arg("output_size_bytes"), pybind11::arg("present_dpu_frq"));
		cl.def("getDPUCycles", (unsigned int (VPUNN::ShaveModel1to1::*)(const int, const int, const int) const) &VPUNN::ShaveModel1to1::getDPUCycles, "Determines the number of cycles related to the input value of a DPU freq and SHV freq given as a parameter\n based on the size of output. In order to get accurate numbers we use a change factor based on the freq we made\n profile and the given value for the profiling\n\n \n the number of cycles required based on CyclesInterfaceType\n\nC++: VPUNN::ShaveModel1to1::getDPUCycles(const int, const int, const int) const --> unsigned int", pybind11::arg("output_size_bytes"), pybind11::arg("present_dpu_frq"), pybind11::arg("present_shv_frq"));
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<4718,4715>, std::shared_ptr<VPUNN::SHVActivation<4718,4715>>, PyCallBack_VPUNN_SHVActivation_4718_4715_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_4718_4715_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_4718_4715_t const &o){ return new PyCallBack_VPUNN_SHVActivation_4718_4715_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<4718,4715> const &o){ return new VPUNN::SHVActivation<4718,4715>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<4718,4715>::*)() const) &VPUNN::SHVActivation<4718, 4715>::getKernelEfficiency, "C++: VPUNN::SHVActivation<4718, 4715>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<4718,4715>::*)() const) &VPUNN::SHVActivation<4718, 4715>::getLatency, "C++: VPUNN::SHVActivation<4718, 4715>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<4718,4715>::*)() const) &VPUNN::SHVActivation<4718, 4715>::cycles, "C++: VPUNN::SHVActivation<4718, 4715>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<441,5067>, std::shared_ptr<VPUNN::SHVActivation<441,5067>>, PyCallBack_VPUNN_SHVActivation_441_5067_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_441_5067_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_441_5067_t const &o){ return new PyCallBack_VPUNN_SHVActivation_441_5067_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<441,5067> const &o){ return new VPUNN::SHVActivation<441,5067>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<441,5067>::*)() const) &VPUNN::SHVActivation<441, 5067>::getKernelEfficiency, "C++: VPUNN::SHVActivation<441, 5067>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<441,5067>::*)() const) &VPUNN::SHVActivation<441, 5067>::getLatency, "C++: VPUNN::SHVActivation<441, 5067>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<441,5067>::*)() const) &VPUNN::SHVActivation<441, 5067>::cycles, "C++: VPUNN::SHVActivation<441, 5067>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
}


// File: VPUNN_26.cpp
#include <array> // std::array
#include <sstream> // __str__

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_547_4956_t : public VPUNN::SHVActivation<547,4956> {
	using VPUNN::SHVActivation<547,4956>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<547,4956> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_836_10043_t : public VPUNN::SHVActivation<836,10043> {
	using VPUNN::SHVActivation<836,10043>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<836,10043> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_1000_0_t : public VPUNN::SHVActivation<1000,0> {
	using VPUNN::SHVActivation<1000,0>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<1000,0> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_855_3319_t : public VPUNN::SHVActivation<855,3319> {
	using VPUNN::SHVActivation<855,3319>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<855,3319> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_742_4432_t : public VPUNN::SHVActivation<742,4432> {
	using VPUNN::SHVActivation<742,4432>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<742,4432> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_17_5192_t : public VPUNN::SHVActivation<17,5192> {
	using VPUNN::SHVActivation<17,5192>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<17,5192> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_742_3914_t : public VPUNN::SHVActivation<742,3914> {
	using VPUNN::SHVActivation<742,3914>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<742,3914> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_742_3824_t : public VPUNN::SHVActivation<742,3824> {
	using VPUNN::SHVActivation<742,3824>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<742,3824> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_397_5138_t : public VPUNN::SHVActivation<397,5138> {
	using VPUNN::SHVActivation<397,5138>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<397,5138> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_742_3831_t : public VPUNN::SHVActivation<742,3831> {
	using VPUNN::SHVActivation<742,3831>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<742,3831> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

void bind_VPUNN_26(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<547,4956>, std::shared_ptr<VPUNN::SHVActivation<547,4956>>, PyCallBack_VPUNN_SHVActivation_547_4956_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_547_4956_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_547_4956_t const &o){ return new PyCallBack_VPUNN_SHVActivation_547_4956_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<547,4956> const &o){ return new VPUNN::SHVActivation<547,4956>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<547,4956>::*)() const) &VPUNN::SHVActivation<547, 4956>::getKernelEfficiency, "C++: VPUNN::SHVActivation<547, 4956>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<547,4956>::*)() const) &VPUNN::SHVActivation<547, 4956>::getLatency, "C++: VPUNN::SHVActivation<547, 4956>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<547,4956>::*)() const) &VPUNN::SHVActivation<547, 4956>::cycles, "C++: VPUNN::SHVActivation<547, 4956>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<836,10043>, std::shared_ptr<VPUNN::SHVActivation<836,10043>>, PyCallBack_VPUNN_SHVActivation_836_10043_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_836_10043_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_836_10043_t const &o){ return new PyCallBack_VPUNN_SHVActivation_836_10043_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<836,10043> const &o){ return new VPUNN::SHVActivation<836,10043>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<836,10043>::*)() const) &VPUNN::SHVActivation<836, 10043>::getKernelEfficiency, "C++: VPUNN::SHVActivation<836, 10043>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<836,10043>::*)() const) &VPUNN::SHVActivation<836, 10043>::getLatency, "C++: VPUNN::SHVActivation<836, 10043>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<836,10043>::*)() const) &VPUNN::SHVActivation<836, 10043>::cycles, "C++: VPUNN::SHVActivation<836, 10043>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<1000,0>, std::shared_ptr<VPUNN::SHVActivation<1000,0>>, PyCallBack_VPUNN_SHVActivation_1000_0_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_1000_0_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_1000_0_t const &o){ return new PyCallBack_VPUNN_SHVActivation_1000_0_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<1000,0> const &o){ return new VPUNN::SHVActivation<1000,0>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<1000,0>::*)() const) &VPUNN::SHVActivation<1000, 0>::getKernelEfficiency, "C++: VPUNN::SHVActivation<1000, 0>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<1000,0>::*)() const) &VPUNN::SHVActivation<1000, 0>::getLatency, "C++: VPUNN::SHVActivation<1000, 0>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<1000,0>::*)() const) &VPUNN::SHVActivation<1000, 0>::cycles, "C++: VPUNN::SHVActivation<1000, 0>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<855,3319>, std::shared_ptr<VPUNN::SHVActivation<855,3319>>, PyCallBack_VPUNN_SHVActivation_855_3319_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_855_3319_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_855_3319_t const &o){ return new PyCallBack_VPUNN_SHVActivation_855_3319_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<855,3319> const &o){ return new VPUNN::SHVActivation<855,3319>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<855,3319>::*)() const) &VPUNN::SHVActivation<855, 3319>::getKernelEfficiency, "C++: VPUNN::SHVActivation<855, 3319>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<855,3319>::*)() const) &VPUNN::SHVActivation<855, 3319>::getLatency, "C++: VPUNN::SHVActivation<855, 3319>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<855,3319>::*)() const) &VPUNN::SHVActivation<855, 3319>::cycles, "C++: VPUNN::SHVActivation<855, 3319>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<742,4432>, std::shared_ptr<VPUNN::SHVActivation<742,4432>>, PyCallBack_VPUNN_SHVActivation_742_4432_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_742_4432_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_742_4432_t const &o){ return new PyCallBack_VPUNN_SHVActivation_742_4432_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<742,4432> const &o){ return new VPUNN::SHVActivation<742,4432>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<742,4432>::*)() const) &VPUNN::SHVActivation<742, 4432>::getKernelEfficiency, "C++: VPUNN::SHVActivation<742, 4432>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<742,4432>::*)() const) &VPUNN::SHVActivation<742, 4432>::getLatency, "C++: VPUNN::SHVActivation<742, 4432>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<742,4432>::*)() const) &VPUNN::SHVActivation<742, 4432>::cycles, "C++: VPUNN::SHVActivation<742, 4432>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<17,5192>, std::shared_ptr<VPUNN::SHVActivation<17,5192>>, PyCallBack_VPUNN_SHVActivation_17_5192_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_17_5192_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_17_5192_t const &o){ return new PyCallBack_VPUNN_SHVActivation_17_5192_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<17,5192> const &o){ return new VPUNN::SHVActivation<17,5192>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<17,5192>::*)() const) &VPUNN::SHVActivation<17, 5192>::getKernelEfficiency, "C++: VPUNN::SHVActivation<17, 5192>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<17,5192>::*)() const) &VPUNN::SHVActivation<17, 5192>::getLatency, "C++: VPUNN::SHVActivation<17, 5192>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<17,5192>::*)() const) &VPUNN::SHVActivation<17, 5192>::cycles, "C++: VPUNN::SHVActivation<17, 5192>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<742,3914>, std::shared_ptr<VPUNN::SHVActivation<742,3914>>, PyCallBack_VPUNN_SHVActivation_742_3914_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_742_3914_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_742_3914_t const &o){ return new PyCallBack_VPUNN_SHVActivation_742_3914_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<742,3914> const &o){ return new VPUNN::SHVActivation<742,3914>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<742,3914>::*)() const) &VPUNN::SHVActivation<742, 3914>::getKernelEfficiency, "C++: VPUNN::SHVActivation<742, 3914>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<742,3914>::*)() const) &VPUNN::SHVActivation<742, 3914>::getLatency, "C++: VPUNN::SHVActivation<742, 3914>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<742,3914>::*)() const) &VPUNN::SHVActivation<742, 3914>::cycles, "C++: VPUNN::SHVActivation<742, 3914>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<742,3824>, std::shared_ptr<VPUNN::SHVActivation<742,3824>>, PyCallBack_VPUNN_SHVActivation_742_3824_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_742_3824_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_742_3824_t const &o){ return new PyCallBack_VPUNN_SHVActivation_742_3824_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<742,3824> const &o){ return new VPUNN::SHVActivation<742,3824>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<742,3824>::*)() const) &VPUNN::SHVActivation<742, 3824>::getKernelEfficiency, "C++: VPUNN::SHVActivation<742, 3824>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<742,3824>::*)() const) &VPUNN::SHVActivation<742, 3824>::getLatency, "C++: VPUNN::SHVActivation<742, 3824>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<742,3824>::*)() const) &VPUNN::SHVActivation<742, 3824>::cycles, "C++: VPUNN::SHVActivation<742, 3824>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<397,5138>, std::shared_ptr<VPUNN::SHVActivation<397,5138>>, PyCallBack_VPUNN_SHVActivation_397_5138_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_397_5138_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_397_5138_t const &o){ return new PyCallBack_VPUNN_SHVActivation_397_5138_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<397,5138> const &o){ return new VPUNN::SHVActivation<397,5138>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<397,5138>::*)() const) &VPUNN::SHVActivation<397, 5138>::getKernelEfficiency, "C++: VPUNN::SHVActivation<397, 5138>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<397,5138>::*)() const) &VPUNN::SHVActivation<397, 5138>::getLatency, "C++: VPUNN::SHVActivation<397, 5138>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<397,5138>::*)() const) &VPUNN::SHVActivation<397, 5138>::cycles, "C++: VPUNN::SHVActivation<397, 5138>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<742,3831>, std::shared_ptr<VPUNN::SHVActivation<742,3831>>, PyCallBack_VPUNN_SHVActivation_742_3831_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_742_3831_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_742_3831_t const &o){ return new PyCallBack_VPUNN_SHVActivation_742_3831_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<742,3831> const &o){ return new VPUNN::SHVActivation<742,3831>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<742,3831>::*)() const) &VPUNN::SHVActivation<742, 3831>::getKernelEfficiency, "C++: VPUNN::SHVActivation<742, 3831>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<742,3831>::*)() const) &VPUNN::SHVActivation<742, 3831>::getLatency, "C++: VPUNN::SHVActivation<742, 3831>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<742,3831>::*)() const) &VPUNN::SHVActivation<742, 3831>::cycles, "C++: VPUNN::SHVActivation<742, 3831>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
}


// File: VPUNN_27.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_10_8482_t : public VPUNN::SHVActivation<10,8482> {
	using VPUNN::SHVActivation<10,8482>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<10,8482> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_291_6349_t : public VPUNN::SHVActivation<291,6349> {
	using VPUNN::SHVActivation<291,6349>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<291,6349> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_69_27428_t : public VPUNN::SHVActivation<69,27428> {
	using VPUNN::SHVActivation<69,27428>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<69,27428> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_830_2810_t : public VPUNN::SHVActivation<830,2810> {
	using VPUNN::SHVActivation<830,2810>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<830,2810> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVActivation file: line:26
struct PyCallBack_VPUNN_SHVActivation_306_8391_t : public VPUNN::SHVActivation<306,8391> {
	using VPUNN::SHVActivation<306,8391>::SHVActivation;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVActivation<306,8391> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVDataMovement file: line:26
struct PyCallBack_VPUNN_SHVDataMovement_1000_0_t : public VPUNN::SHVDataMovement<1000,0> {
	using VPUNN::SHVDataMovement<1000,0>::SHVDataMovement;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVDataMovement<1000,0> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_4_15829_t : public VPUNN::SHVElementwise<4,15829> {
	using VPUNN::SHVElementwise<4,15829>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<4,15829> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_1000_0_t : public VPUNN::SHVElementwise<1000,0> {
	using VPUNN::SHVElementwise<1000,0>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<1000,0> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_12_11587_t : public VPUNN::SHVElementwise<12,11587> {
	using VPUNN::SHVElementwise<12,11587>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<12,11587> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_8_13192_t : public VPUNN::SHVElementwise<8,13192> {
	using VPUNN::SHVElementwise<8,13192>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<8,13192> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

void bind_VPUNN_27(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<10,8482>, std::shared_ptr<VPUNN::SHVActivation<10,8482>>, PyCallBack_VPUNN_SHVActivation_10_8482_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_10_8482_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_10_8482_t const &o){ return new PyCallBack_VPUNN_SHVActivation_10_8482_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<10,8482> const &o){ return new VPUNN::SHVActivation<10,8482>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<10,8482>::*)() const) &VPUNN::SHVActivation<10, 8482>::getKernelEfficiency, "C++: VPUNN::SHVActivation<10, 8482>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<10,8482>::*)() const) &VPUNN::SHVActivation<10, 8482>::getLatency, "C++: VPUNN::SHVActivation<10, 8482>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<10,8482>::*)() const) &VPUNN::SHVActivation<10, 8482>::cycles, "C++: VPUNN::SHVActivation<10, 8482>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<291,6349>, std::shared_ptr<VPUNN::SHVActivation<291,6349>>, PyCallBack_VPUNN_SHVActivation_291_6349_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_291_6349_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_291_6349_t const &o){ return new PyCallBack_VPUNN_SHVActivation_291_6349_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<291,6349> const &o){ return new VPUNN::SHVActivation<291,6349>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<291,6349>::*)() const) &VPUNN::SHVActivation<291, 6349>::getKernelEfficiency, "C++: VPUNN::SHVActivation<291, 6349>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<291,6349>::*)() const) &VPUNN::SHVActivation<291, 6349>::getLatency, "C++: VPUNN::SHVActivation<291, 6349>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<291,6349>::*)() const) &VPUNN::SHVActivation<291, 6349>::cycles, "C++: VPUNN::SHVActivation<291, 6349>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<69,27428>, std::shared_ptr<VPUNN::SHVActivation<69,27428>>, PyCallBack_VPUNN_SHVActivation_69_27428_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_69_27428_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_69_27428_t const &o){ return new PyCallBack_VPUNN_SHVActivation_69_27428_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<69,27428> const &o){ return new VPUNN::SHVActivation<69,27428>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<69,27428>::*)() const) &VPUNN::SHVActivation<69, 27428>::getKernelEfficiency, "C++: VPUNN::SHVActivation<69, 27428>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<69,27428>::*)() const) &VPUNN::SHVActivation<69, 27428>::getLatency, "C++: VPUNN::SHVActivation<69, 27428>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<69,27428>::*)() const) &VPUNN::SHVActivation<69, 27428>::cycles, "C++: VPUNN::SHVActivation<69, 27428>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<830,2810>, std::shared_ptr<VPUNN::SHVActivation<830,2810>>, PyCallBack_VPUNN_SHVActivation_830_2810_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_830_2810_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_830_2810_t const &o){ return new PyCallBack_VPUNN_SHVActivation_830_2810_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<830,2810> const &o){ return new VPUNN::SHVActivation<830,2810>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<830,2810>::*)() const) &VPUNN::SHVActivation<830, 2810>::getKernelEfficiency, "C++: VPUNN::SHVActivation<830, 2810>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<830,2810>::*)() const) &VPUNN::SHVActivation<830, 2810>::getLatency, "C++: VPUNN::SHVActivation<830, 2810>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<830,2810>::*)() const) &VPUNN::SHVActivation<830, 2810>::cycles, "C++: VPUNN::SHVActivation<830, 2810>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVActivation file: line:26
		pybind11::class_<VPUNN::SHVActivation<306,8391>, std::shared_ptr<VPUNN::SHVActivation<306,8391>>, PyCallBack_VPUNN_SHVActivation_306_8391_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVActivation_306_8391_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVActivation_306_8391_t const &o){ return new PyCallBack_VPUNN_SHVActivation_306_8391_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVActivation<306,8391> const &o){ return new VPUNN::SHVActivation<306,8391>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVActivation<306,8391>::*)() const) &VPUNN::SHVActivation<306, 8391>::getKernelEfficiency, "C++: VPUNN::SHVActivation<306, 8391>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVActivation<306,8391>::*)() const) &VPUNN::SHVActivation<306, 8391>::getLatency, "C++: VPUNN::SHVActivation<306, 8391>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVActivation<306,8391>::*)() const) &VPUNN::SHVActivation<306, 8391>::cycles, "C++: VPUNN::SHVActivation<306, 8391>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVDataMovement file: line:26
		pybind11::class_<VPUNN::SHVDataMovement<1000,0>, std::shared_ptr<VPUNN::SHVDataMovement<1000,0>>, PyCallBack_VPUNN_SHVDataMovement_1000_0_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVDataMovement_1000_0_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVDataMovement_1000_0_t const &o){ return new PyCallBack_VPUNN_SHVDataMovement_1000_0_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVDataMovement<1000,0> const &o){ return new VPUNN::SHVDataMovement<1000,0>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVDataMovement<1000,0>::*)() const) &VPUNN::SHVDataMovement<1000, 0>::getKernelEfficiency, "C++: VPUNN::SHVDataMovement<1000, 0>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVDataMovement<1000,0>::*)() const) &VPUNN::SHVDataMovement<1000, 0>::getLatency, "C++: VPUNN::SHVDataMovement<1000, 0>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVDataMovement<1000,0>::*)() const) &VPUNN::SHVDataMovement<1000, 0>::cycles, "C++: VPUNN::SHVDataMovement<1000, 0>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<4,15829>, std::shared_ptr<VPUNN::SHVElementwise<4,15829>>, PyCallBack_VPUNN_SHVElementwise_4_15829_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_4_15829_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_4_15829_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_4_15829_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<4,15829> const &o){ return new VPUNN::SHVElementwise<4,15829>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<4,15829>::*)() const) &VPUNN::SHVElementwise<4, 15829>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<4, 15829>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<4,15829>::*)() const) &VPUNN::SHVElementwise<4, 15829>::getLatency, "C++: VPUNN::SHVElementwise<4, 15829>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<4,15829>::*)() const) &VPUNN::SHVElementwise<4, 15829>::cycles, "C++: VPUNN::SHVElementwise<4, 15829>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<1000,0>, std::shared_ptr<VPUNN::SHVElementwise<1000,0>>, PyCallBack_VPUNN_SHVElementwise_1000_0_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_1000_0_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_1000_0_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_1000_0_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<1000,0> const &o){ return new VPUNN::SHVElementwise<1000,0>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<1000,0>::*)() const) &VPUNN::SHVElementwise<1000, 0>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<1000, 0>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<1000,0>::*)() const) &VPUNN::SHVElementwise<1000, 0>::getLatency, "C++: VPUNN::SHVElementwise<1000, 0>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<1000,0>::*)() const) &VPUNN::SHVElementwise<1000, 0>::cycles, "C++: VPUNN::SHVElementwise<1000, 0>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<12,11587>, std::shared_ptr<VPUNN::SHVElementwise<12,11587>>, PyCallBack_VPUNN_SHVElementwise_12_11587_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_12_11587_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_12_11587_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_12_11587_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<12,11587> const &o){ return new VPUNN::SHVElementwise<12,11587>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<12,11587>::*)() const) &VPUNN::SHVElementwise<12, 11587>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<12, 11587>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<12,11587>::*)() const) &VPUNN::SHVElementwise<12, 11587>::getLatency, "C++: VPUNN::SHVElementwise<12, 11587>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<12,11587>::*)() const) &VPUNN::SHVElementwise<12, 11587>::cycles, "C++: VPUNN::SHVElementwise<12, 11587>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<8,13192>, std::shared_ptr<VPUNN::SHVElementwise<8,13192>>, PyCallBack_VPUNN_SHVElementwise_8_13192_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_8_13192_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_8_13192_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_8_13192_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<8,13192> const &o){ return new VPUNN::SHVElementwise<8,13192>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<8,13192>::*)() const) &VPUNN::SHVElementwise<8, 13192>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<8, 13192>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<8,13192>::*)() const) &VPUNN::SHVElementwise<8, 13192>::getLatency, "C++: VPUNN::SHVElementwise<8, 13192>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<8,13192>::*)() const) &VPUNN::SHVElementwise<8, 13192>::cycles, "C++: VPUNN::SHVElementwise<8, 13192>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
}


// File: VPUNN_28.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_8_13036_t : public VPUNN::SHVElementwise<8,13036> {
	using VPUNN::SHVElementwise<8,13036>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<8,13036> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_45_30591_t : public VPUNN::SHVElementwise<45,30591> {
	using VPUNN::SHVElementwise<45,30591>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<45,30591> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_15_11047_t : public VPUNN::SHVElementwise<15,11047> {
	using VPUNN::SHVElementwise<15,11047>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<15,11047> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_15_11000_t : public VPUNN::SHVElementwise<15,11000> {
	using VPUNN::SHVElementwise<15,11000>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<15,11000> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVElementwise file: line:26
struct PyCallBack_VPUNN_SHVElementwise_789_12946_t : public VPUNN::SHVElementwise<789,12946> {
	using VPUNN::SHVElementwise<789,12946>::SHVElementwise;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVElementwise<789,12946> *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVSigmoid file: line:19
struct PyCallBack_VPUNN_SHVSigmoid : public VPUNN::SHVSigmoid {
	using VPUNN::SHVSigmoid::SHVSigmoid;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSigmoid *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVELU file: line:19
struct PyCallBack_VPUNN_SHVELU : public VPUNN::SHVELU {
	using VPUNN::SHVELU::SHVELU;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVELU *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVHardSigmoid file: line:19
struct PyCallBack_VPUNN_SHVHardSigmoid : public VPUNN::SHVHardSigmoid {
	using VPUNN::SHVHardSigmoid::SHVHardSigmoid;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVHardSigmoid *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSoftmax file: line:19
struct PyCallBack_VPUNN_SHVSoftmax : public VPUNN::SHVSoftmax {
	using VPUNN::SHVSoftmax::SHVSoftmax;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSoftmax *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVHardSwish file: line:19
struct PyCallBack_VPUNN_SHVHardSwish : public VPUNN::SHVHardSwish {
	using VPUNN::SHVHardSwish::SHVHardSwish;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVHardSwish *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVClamp file: line:19
struct PyCallBack_VPUNN_SHVClamp : public VPUNN::SHVClamp {
	using VPUNN::SHVClamp::SHVClamp;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVClamp *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVFakeQuantize file: line:19
struct PyCallBack_VPUNN_SHVFakeQuantize : public VPUNN::SHVFakeQuantize {
	using VPUNN::SHVFakeQuantize::SHVFakeQuantize;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVFakeQuantize *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVQuantizeCast file: line:19
struct PyCallBack_VPUNN_SHVQuantizeCast : public VPUNN::SHVQuantizeCast {
	using VPUNN::SHVQuantizeCast::SHVQuantizeCast;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVQuantizeCast *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVTanh file: line:19
struct PyCallBack_VPUNN_SHVTanh : public VPUNN::SHVTanh {
	using VPUNN::SHVTanh::SHVTanh;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVTanh *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSin file: line:19
struct PyCallBack_VPUNN_SHVSin : public VPUNN::SHVSin {
	using VPUNN::SHVSin::SHVSin;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSin *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVCos file: line:19
struct PyCallBack_VPUNN_SHVCos : public VPUNN::SHVCos {
	using VPUNN::SHVCos::SHVCos;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVCos *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSqrt file: line:19
struct PyCallBack_VPUNN_SHVSqrt : public VPUNN::SHVSqrt {
	using VPUNN::SHVSqrt::SHVSqrt;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSqrt *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

void bind_VPUNN_28(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<8,13036>, std::shared_ptr<VPUNN::SHVElementwise<8,13036>>, PyCallBack_VPUNN_SHVElementwise_8_13036_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_8_13036_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_8_13036_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_8_13036_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<8,13036> const &o){ return new VPUNN::SHVElementwise<8,13036>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<8,13036>::*)() const) &VPUNN::SHVElementwise<8, 13036>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<8, 13036>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<8,13036>::*)() const) &VPUNN::SHVElementwise<8, 13036>::getLatency, "C++: VPUNN::SHVElementwise<8, 13036>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<8,13036>::*)() const) &VPUNN::SHVElementwise<8, 13036>::cycles, "C++: VPUNN::SHVElementwise<8, 13036>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<45,30591>, std::shared_ptr<VPUNN::SHVElementwise<45,30591>>, PyCallBack_VPUNN_SHVElementwise_45_30591_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_45_30591_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_45_30591_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_45_30591_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<45,30591> const &o){ return new VPUNN::SHVElementwise<45,30591>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<45,30591>::*)() const) &VPUNN::SHVElementwise<45, 30591>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<45, 30591>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<45,30591>::*)() const) &VPUNN::SHVElementwise<45, 30591>::getLatency, "C++: VPUNN::SHVElementwise<45, 30591>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<45,30591>::*)() const) &VPUNN::SHVElementwise<45, 30591>::cycles, "C++: VPUNN::SHVElementwise<45, 30591>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<15,11047>, std::shared_ptr<VPUNN::SHVElementwise<15,11047>>, PyCallBack_VPUNN_SHVElementwise_15_11047_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_15_11047_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_15_11047_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_15_11047_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<15,11047> const &o){ return new VPUNN::SHVElementwise<15,11047>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<15,11047>::*)() const) &VPUNN::SHVElementwise<15, 11047>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<15, 11047>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<15,11047>::*)() const) &VPUNN::SHVElementwise<15, 11047>::getLatency, "C++: VPUNN::SHVElementwise<15, 11047>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<15,11047>::*)() const) &VPUNN::SHVElementwise<15, 11047>::cycles, "C++: VPUNN::SHVElementwise<15, 11047>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<15,11000>, std::shared_ptr<VPUNN::SHVElementwise<15,11000>>, PyCallBack_VPUNN_SHVElementwise_15_11000_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_15_11000_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_15_11000_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_15_11000_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<15,11000> const &o){ return new VPUNN::SHVElementwise<15,11000>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<15,11000>::*)() const) &VPUNN::SHVElementwise<15, 11000>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<15, 11000>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<15,11000>::*)() const) &VPUNN::SHVElementwise<15, 11000>::getLatency, "C++: VPUNN::SHVElementwise<15, 11000>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<15,11000>::*)() const) &VPUNN::SHVElementwise<15, 11000>::cycles, "C++: VPUNN::SHVElementwise<15, 11000>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVElementwise file: line:26
		pybind11::class_<VPUNN::SHVElementwise<789,12946>, std::shared_ptr<VPUNN::SHVElementwise<789,12946>>, PyCallBack_VPUNN_SHVElementwise_789_12946_t, VPUNN::SWOperation> cl(M("VPUNN"), "SHVElementwise_789_12946_t", "");
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVElementwise_789_12946_t const &o){ return new PyCallBack_VPUNN_SHVElementwise_789_12946_t(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVElementwise<789,12946> const &o){ return new VPUNN::SHVElementwise<789,12946>(o); } ) );
		cl.def("getKernelEfficiency", (float (VPUNN::SHVElementwise<789,12946>::*)() const) &VPUNN::SHVElementwise<789, 12946>::getKernelEfficiency, "C++: VPUNN::SHVElementwise<789, 12946>::getKernelEfficiency() const --> float");
		cl.def("getLatency", (unsigned int (VPUNN::SHVElementwise<789,12946>::*)() const) &VPUNN::SHVElementwise<789, 12946>::getLatency, "C++: VPUNN::SHVElementwise<789, 12946>::getLatency() const --> unsigned int");
		cl.def("cycles", (unsigned int (VPUNN::SHVElementwise<789,12946>::*)() const) &VPUNN::SHVElementwise<789, 12946>::cycles, "C++: VPUNN::SHVElementwise<789, 12946>::cycles() const --> unsigned int");
		cl.def_readwrite("device", &VPUNN::SWOperation::device);
		cl.def_readonly("inputs", &VPUNN::SWOperation::inputs);
		cl.def_readonly("outputs", &VPUNN::SWOperation::outputs);
		cl.def("cycles", (unsigned int (VPUNN::SWOperation::*)() const) &VPUNN::SWOperation::cycles, "Return the number of cycles of the sw operation\n\n \n unsigned int\n\nC++: VPUNN::SWOperation::cycles() const --> unsigned int");
	}
	{ // VPUNN::SHVSigmoid file: line:19
		pybind11::class_<VPUNN::SHVSigmoid, std::shared_ptr<VPUNN::SHVSigmoid>, PyCallBack_VPUNN_SHVSigmoid, VPUNN::SHVActivation<4718,4715>> cl(M("VPUNN"), "SHVSigmoid", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSigmoid const &o){ return new PyCallBack_VPUNN_SHVSigmoid(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSigmoid const &o){ return new VPUNN::SHVSigmoid(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVELU file: line:19
		pybind11::class_<VPUNN::SHVELU, std::shared_ptr<VPUNN::SHVELU>, PyCallBack_VPUNN_SHVELU, VPUNN::SHVActivation<441,5067>> cl(M("VPUNN"), "SHVELU", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVELU const &o){ return new PyCallBack_VPUNN_SHVELU(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVELU const &o){ return new VPUNN::SHVELU(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVHardSigmoid file: line:19
		pybind11::class_<VPUNN::SHVHardSigmoid, std::shared_ptr<VPUNN::SHVHardSigmoid>, PyCallBack_VPUNN_SHVHardSigmoid, VPUNN::SHVActivation<547,4956>> cl(M("VPUNN"), "SHVHardSigmoid", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVHardSigmoid const &o){ return new PyCallBack_VPUNN_SHVHardSigmoid(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVHardSigmoid const &o){ return new VPUNN::SHVHardSigmoid(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSoftmax file: line:19
		pybind11::class_<VPUNN::SHVSoftmax, std::shared_ptr<VPUNN::SHVSoftmax>, PyCallBack_VPUNN_SHVSoftmax, VPUNN::SHVActivation<836,10043>> cl(M("VPUNN"), "SHVSoftmax", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSoftmax const &o){ return new PyCallBack_VPUNN_SHVSoftmax(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSoftmax const &o){ return new VPUNN::SHVSoftmax(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVHardSwish file: line:19
		pybind11::class_<VPUNN::SHVHardSwish, std::shared_ptr<VPUNN::SHVHardSwish>, PyCallBack_VPUNN_SHVHardSwish, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVHardSwish", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVHardSwish const &o){ return new PyCallBack_VPUNN_SHVHardSwish(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVHardSwish const &o){ return new VPUNN::SHVHardSwish(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVClamp file: line:19
		pybind11::class_<VPUNN::SHVClamp, std::shared_ptr<VPUNN::SHVClamp>, PyCallBack_VPUNN_SHVClamp, VPUNN::SHVActivation<855,3319>> cl(M("VPUNN"), "SHVClamp", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVClamp const &o){ return new PyCallBack_VPUNN_SHVClamp(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVClamp const &o){ return new VPUNN::SHVClamp(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVFakeQuantize file: line:19
		pybind11::class_<VPUNN::SHVFakeQuantize, std::shared_ptr<VPUNN::SHVFakeQuantize>, PyCallBack_VPUNN_SHVFakeQuantize, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVFakeQuantize", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVFakeQuantize const &o){ return new PyCallBack_VPUNN_SHVFakeQuantize(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVFakeQuantize const &o){ return new VPUNN::SHVFakeQuantize(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVQuantizeCast file: line:19
		pybind11::class_<VPUNN::SHVQuantizeCast, std::shared_ptr<VPUNN::SHVQuantizeCast>, PyCallBack_VPUNN_SHVQuantizeCast, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVQuantizeCast", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVQuantizeCast const &o){ return new PyCallBack_VPUNN_SHVQuantizeCast(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVQuantizeCast const &o){ return new VPUNN::SHVQuantizeCast(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVTanh file: line:19
		pybind11::class_<VPUNN::SHVTanh, std::shared_ptr<VPUNN::SHVTanh>, PyCallBack_VPUNN_SHVTanh, VPUNN::SHVActivation<742,4432>> cl(M("VPUNN"), "SHVTanh", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVTanh const &o){ return new PyCallBack_VPUNN_SHVTanh(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVTanh const &o){ return new VPUNN::SHVTanh(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSin file: line:19
		pybind11::class_<VPUNN::SHVSin, std::shared_ptr<VPUNN::SHVSin>, PyCallBack_VPUNN_SHVSin, VPUNN::SHVActivation<17,5192>> cl(M("VPUNN"), "SHVSin", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSin const &o){ return new PyCallBack_VPUNN_SHVSin(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSin const &o){ return new VPUNN::SHVSin(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVCos file: line:19
		pybind11::class_<VPUNN::SHVCos, std::shared_ptr<VPUNN::SHVCos>, PyCallBack_VPUNN_SHVCos, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVCos", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVCos const &o){ return new PyCallBack_VPUNN_SHVCos(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVCos const &o){ return new VPUNN::SHVCos(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSqrt file: line:19
		pybind11::class_<VPUNN::SHVSqrt, std::shared_ptr<VPUNN::SHVSqrt>, PyCallBack_VPUNN_SHVSqrt, VPUNN::SHVActivation<742,3914>> cl(M("VPUNN"), "SHVSqrt", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSqrt const &o){ return new PyCallBack_VPUNN_SHVSqrt(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSqrt const &o){ return new VPUNN::SHVSqrt(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
}


// File: VPUNN_29.cpp
#include <sstream> // __str__

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SHVSinh file: line:19
struct PyCallBack_VPUNN_SHVSinh : public VPUNN::SHVSinh {
	using VPUNN::SHVSinh::SHVSinh;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSinh *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVCosh file: line:19
struct PyCallBack_VPUNN_SHVCosh : public VPUNN::SHVCosh {
	using VPUNN::SHVCosh::SHVCosh;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVCosh *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAsinh file: line:19
struct PyCallBack_VPUNN_SHVAsinh : public VPUNN::SHVAsinh {
	using VPUNN::SHVAsinh::SHVAsinh;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAsinh *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAcosh file: line:19
struct PyCallBack_VPUNN_SHVAcosh : public VPUNN::SHVAcosh {
	using VPUNN::SHVAcosh::SHVAcosh;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAcosh *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAbs file: line:19
struct PyCallBack_VPUNN_SHVAbs : public VPUNN::SHVAbs {
	using VPUNN::SHVAbs::SHVAbs;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAbs *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAtan file: line:19
struct PyCallBack_VPUNN_SHVAtan : public VPUNN::SHVAtan {
	using VPUNN::SHVAtan::SHVAtan;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAtan *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAsin file: line:19
struct PyCallBack_VPUNN_SHVAsin : public VPUNN::SHVAsin {
	using VPUNN::SHVAsin::SHVAsin;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAsin *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAcos file: line:19
struct PyCallBack_VPUNN_SHVAcos : public VPUNN::SHVAcos {
	using VPUNN::SHVAcos::SHVAcos;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAcos *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVAtanh file: line:19
struct PyCallBack_VPUNN_SHVAtanh : public VPUNN::SHVAtanh {
	using VPUNN::SHVAtanh::SHVAtanh;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAtanh *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVLog file: line:19
struct PyCallBack_VPUNN_SHVLog : public VPUNN::SHVLog {
	using VPUNN::SHVLog::SHVLog;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVLog *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSelu file: line:19
struct PyCallBack_VPUNN_SHVSelu : public VPUNN::SHVSelu {
	using VPUNN::SHVSelu::SHVSelu;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSelu *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVGelu file: line:19
struct PyCallBack_VPUNN_SHVGelu : public VPUNN::SHVGelu {
	using VPUNN::SHVGelu::SHVGelu;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVGelu *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVExp file: line:19
struct PyCallBack_VPUNN_SHVExp : public VPUNN::SHVExp {
	using VPUNN::SHVExp::SHVExp;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVExp *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVFloor file: line:19
struct PyCallBack_VPUNN_SHVFloor : public VPUNN::SHVFloor {
	using VPUNN::SHVFloor::SHVFloor;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVFloor *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVRound file: line:19
struct PyCallBack_VPUNN_SHVRound : public VPUNN::SHVRound {
	using VPUNN::SHVRound::SHVRound;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVRound *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVMish file: line:19
struct PyCallBack_VPUNN_SHVMish : public VPUNN::SHVMish {
	using VPUNN::SHVMish::SHVMish;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVMish *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVErf file: line:19
struct PyCallBack_VPUNN_SHVErf : public VPUNN::SHVErf {
	using VPUNN::SHVErf::SHVErf;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVErf *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVNegative file: line:19
struct PyCallBack_VPUNN_SHVNegative : public VPUNN::SHVNegative {
	using VPUNN::SHVNegative::SHVNegative;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVNegative *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSign file: line:19
struct PyCallBack_VPUNN_SHVSign : public VPUNN::SHVSign {
	using VPUNN::SHVSign::SHVSign;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSign *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVScaleShift file: line:19
struct PyCallBack_VPUNN_SHVScaleShift : public VPUNN::SHVScaleShift {
	using VPUNN::SHVScaleShift::SHVScaleShift;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVScaleShift *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVYuvToRgb file: line:19
struct PyCallBack_VPUNN_SHVYuvToRgb : public VPUNN::SHVYuvToRgb {
	using VPUNN::SHVYuvToRgb::SHVYuvToRgb;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVYuvToRgb *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSoftPlus file: line:19
struct PyCallBack_VPUNN_SHVSoftPlus : public VPUNN::SHVSoftPlus {
	using VPUNN::SHVSoftPlus::SHVSoftPlus;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSoftPlus *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVSwish file: line:19
struct PyCallBack_VPUNN_SHVSwish : public VPUNN::SHVSwish {
	using VPUNN::SHVSwish::SHVSwish;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSwish *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVMVN file: line:19
struct PyCallBack_VPUNN_SHVMVN : public VPUNN::SHVMVN {
	using VPUNN::SHVMVN::SHVMVN;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVMVN *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVCeiling file: line:19
struct PyCallBack_VPUNN_SHVCeiling : public VPUNN::SHVCeiling {
	using VPUNN::SHVCeiling::SHVCeiling;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVCeiling *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVActivation::cycles();
	}
};

// VPUNN::SHVPower file: line:19
struct PyCallBack_VPUNN_SHVPower : public VPUNN::SHVPower {
	using VPUNN::SHVPower::SHVPower;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVPower *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVAdd file: line:19
struct PyCallBack_VPUNN_SHVAdd : public VPUNN::SHVAdd {
	using VPUNN::SHVAdd::SHVAdd;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAdd *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

void bind_VPUNN_29(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SHVSinh file: line:19
		pybind11::class_<VPUNN::SHVSinh, std::shared_ptr<VPUNN::SHVSinh>, PyCallBack_VPUNN_SHVSinh, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVSinh", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSinh const &o){ return new PyCallBack_VPUNN_SHVSinh(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSinh const &o){ return new VPUNN::SHVSinh(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVCosh file: line:19
		pybind11::class_<VPUNN::SHVCosh, std::shared_ptr<VPUNN::SHVCosh>, PyCallBack_VPUNN_SHVCosh, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVCosh", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVCosh const &o){ return new PyCallBack_VPUNN_SHVCosh(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVCosh const &o){ return new VPUNN::SHVCosh(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAsinh file: line:19
		pybind11::class_<VPUNN::SHVAsinh, std::shared_ptr<VPUNN::SHVAsinh>, PyCallBack_VPUNN_SHVAsinh, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAsinh", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAsinh const &o){ return new PyCallBack_VPUNN_SHVAsinh(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAsinh const &o){ return new VPUNN::SHVAsinh(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAcosh file: line:19
		pybind11::class_<VPUNN::SHVAcosh, std::shared_ptr<VPUNN::SHVAcosh>, PyCallBack_VPUNN_SHVAcosh, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAcosh", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAcosh const &o){ return new PyCallBack_VPUNN_SHVAcosh(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAcosh const &o){ return new VPUNN::SHVAcosh(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAbs file: line:19
		pybind11::class_<VPUNN::SHVAbs, std::shared_ptr<VPUNN::SHVAbs>, PyCallBack_VPUNN_SHVAbs, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAbs", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAbs const &o){ return new PyCallBack_VPUNN_SHVAbs(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAbs const &o){ return new VPUNN::SHVAbs(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAtan file: line:19
		pybind11::class_<VPUNN::SHVAtan, std::shared_ptr<VPUNN::SHVAtan>, PyCallBack_VPUNN_SHVAtan, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAtan", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAtan const &o){ return new PyCallBack_VPUNN_SHVAtan(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAtan const &o){ return new VPUNN::SHVAtan(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAsin file: line:19
		pybind11::class_<VPUNN::SHVAsin, std::shared_ptr<VPUNN::SHVAsin>, PyCallBack_VPUNN_SHVAsin, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAsin", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAsin const &o){ return new PyCallBack_VPUNN_SHVAsin(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAsin const &o){ return new VPUNN::SHVAsin(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAcos file: line:19
		pybind11::class_<VPUNN::SHVAcos, std::shared_ptr<VPUNN::SHVAcos>, PyCallBack_VPUNN_SHVAcos, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAcos", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAcos const &o){ return new PyCallBack_VPUNN_SHVAcos(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAcos const &o){ return new VPUNN::SHVAcos(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAtanh file: line:19
		pybind11::class_<VPUNN::SHVAtanh, std::shared_ptr<VPUNN::SHVAtanh>, PyCallBack_VPUNN_SHVAtanh, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVAtanh", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAtanh const &o){ return new PyCallBack_VPUNN_SHVAtanh(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAtanh const &o){ return new VPUNN::SHVAtanh(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVLog file: line:19
		pybind11::class_<VPUNN::SHVLog, std::shared_ptr<VPUNN::SHVLog>, PyCallBack_VPUNN_SHVLog, VPUNN::SHVActivation<742,3824>> cl(M("VPUNN"), "SHVLog", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVLog const &o){ return new PyCallBack_VPUNN_SHVLog(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVLog const &o){ return new VPUNN::SHVLog(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSelu file: line:19
		pybind11::class_<VPUNN::SHVSelu, std::shared_ptr<VPUNN::SHVSelu>, PyCallBack_VPUNN_SHVSelu, VPUNN::SHVActivation<397,5138>> cl(M("VPUNN"), "SHVSelu", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSelu const &o){ return new PyCallBack_VPUNN_SHVSelu(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSelu const &o){ return new VPUNN::SHVSelu(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVGelu file: line:19
		pybind11::class_<VPUNN::SHVGelu, std::shared_ptr<VPUNN::SHVGelu>, PyCallBack_VPUNN_SHVGelu, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVGelu", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVGelu const &o){ return new PyCallBack_VPUNN_SHVGelu(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVGelu const &o){ return new VPUNN::SHVGelu(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVExp file: line:19
		pybind11::class_<VPUNN::SHVExp, std::shared_ptr<VPUNN::SHVExp>, PyCallBack_VPUNN_SHVExp, VPUNN::SHVActivation<742,3831>> cl(M("VPUNN"), "SHVExp", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVExp const &o){ return new PyCallBack_VPUNN_SHVExp(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVExp const &o){ return new VPUNN::SHVExp(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVFloor file: line:19
		pybind11::class_<VPUNN::SHVFloor, std::shared_ptr<VPUNN::SHVFloor>, PyCallBack_VPUNN_SHVFloor, VPUNN::SHVActivation<10,8482>> cl(M("VPUNN"), "SHVFloor", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVFloor const &o){ return new PyCallBack_VPUNN_SHVFloor(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVFloor const &o){ return new VPUNN::SHVFloor(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVRound file: line:19
		pybind11::class_<VPUNN::SHVRound, std::shared_ptr<VPUNN::SHVRound>, PyCallBack_VPUNN_SHVRound, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVRound", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVRound const &o){ return new PyCallBack_VPUNN_SHVRound(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVRound const &o){ return new VPUNN::SHVRound(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVMish file: line:19
		pybind11::class_<VPUNN::SHVMish, std::shared_ptr<VPUNN::SHVMish>, PyCallBack_VPUNN_SHVMish, VPUNN::SHVActivation<291,6349>> cl(M("VPUNN"), "SHVMish", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVMish const &o){ return new PyCallBack_VPUNN_SHVMish(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVMish const &o){ return new VPUNN::SHVMish(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVErf file: line:19
		pybind11::class_<VPUNN::SHVErf, std::shared_ptr<VPUNN::SHVErf>, PyCallBack_VPUNN_SHVErf, VPUNN::SHVActivation<69,27428>> cl(M("VPUNN"), "SHVErf", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVErf const &o){ return new PyCallBack_VPUNN_SHVErf(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVErf const &o){ return new VPUNN::SHVErf(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVNegative file: line:19
		pybind11::class_<VPUNN::SHVNegative, std::shared_ptr<VPUNN::SHVNegative>, PyCallBack_VPUNN_SHVNegative, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVNegative", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVNegative const &o){ return new PyCallBack_VPUNN_SHVNegative(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVNegative const &o){ return new VPUNN::SHVNegative(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSign file: line:19
		pybind11::class_<VPUNN::SHVSign, std::shared_ptr<VPUNN::SHVSign>, PyCallBack_VPUNN_SHVSign, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVSign", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSign const &o){ return new PyCallBack_VPUNN_SHVSign(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSign const &o){ return new VPUNN::SHVSign(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVScaleShift file: line:19
		pybind11::class_<VPUNN::SHVScaleShift, std::shared_ptr<VPUNN::SHVScaleShift>, PyCallBack_VPUNN_SHVScaleShift, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVScaleShift", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVScaleShift const &o){ return new PyCallBack_VPUNN_SHVScaleShift(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVScaleShift const &o){ return new VPUNN::SHVScaleShift(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVYuvToRgb file: line:19
		pybind11::class_<VPUNN::SHVYuvToRgb, std::shared_ptr<VPUNN::SHVYuvToRgb>, PyCallBack_VPUNN_SHVYuvToRgb, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVYuvToRgb", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVYuvToRgb const &o){ return new PyCallBack_VPUNN_SHVYuvToRgb(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVYuvToRgb const &o){ return new VPUNN::SHVYuvToRgb(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSoftPlus file: line:19
		pybind11::class_<VPUNN::SHVSoftPlus, std::shared_ptr<VPUNN::SHVSoftPlus>, PyCallBack_VPUNN_SHVSoftPlus, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVSoftPlus", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSoftPlus const &o){ return new PyCallBack_VPUNN_SHVSoftPlus(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSoftPlus const &o){ return new VPUNN::SHVSoftPlus(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSwish file: line:19
		pybind11::class_<VPUNN::SHVSwish, std::shared_ptr<VPUNN::SHVSwish>, PyCallBack_VPUNN_SHVSwish, VPUNN::SHVActivation<1000,0>> cl(M("VPUNN"), "SHVSwish", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSwish const &o){ return new PyCallBack_VPUNN_SHVSwish(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSwish const &o){ return new VPUNN::SHVSwish(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVMVN file: line:19
		pybind11::class_<VPUNN::SHVMVN, std::shared_ptr<VPUNN::SHVMVN>, PyCallBack_VPUNN_SHVMVN, VPUNN::SHVActivation<830,2810>> cl(M("VPUNN"), "SHVMVN", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVMVN const &o){ return new PyCallBack_VPUNN_SHVMVN(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVMVN const &o){ return new VPUNN::SHVMVN(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVCeiling file: line:19
		pybind11::class_<VPUNN::SHVCeiling, std::shared_ptr<VPUNN::SHVCeiling>, PyCallBack_VPUNN_SHVCeiling, VPUNN::SHVActivation<306,8391>> cl(M("VPUNN"), "SHVCeiling", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVCeiling const &o){ return new PyCallBack_VPUNN_SHVCeiling(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVCeiling const &o){ return new VPUNN::SHVCeiling(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVPower file: line:19
		pybind11::class_<VPUNN::SHVPower, std::shared_ptr<VPUNN::SHVPower>, PyCallBack_VPUNN_SHVPower, VPUNN::SHVElementwise<4,15829>> cl(M("VPUNN"), "SHVPower", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVPower const &o){ return new PyCallBack_VPUNN_SHVPower(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVPower const &o){ return new VPUNN::SHVPower(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAdd file: line:19
		pybind11::class_<VPUNN::SHVAdd, std::shared_ptr<VPUNN::SHVAdd>, PyCallBack_VPUNN_SHVAdd, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVAdd", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAdd const &o){ return new PyCallBack_VPUNN_SHVAdd(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAdd const &o){ return new VPUNN::SHVAdd(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
}


// File: VPUNN_30.cpp
#include <sstream> // __str__

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SHVDivide file: line:19
struct PyCallBack_VPUNN_SHVDivide : public VPUNN::SHVDivide {
	using VPUNN::SHVDivide::SHVDivide;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVDivide *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVSquaredDiff file: line:19
struct PyCallBack_VPUNN_SHVSquaredDiff : public VPUNN::SHVSquaredDiff {
	using VPUNN::SHVSquaredDiff::SHVSquaredDiff;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSquaredDiff *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVFloorMod file: line:19
struct PyCallBack_VPUNN_SHVFloorMod : public VPUNN::SHVFloorMod {
	using VPUNN::SHVFloorMod::SHVFloorMod;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVFloorMod *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVLess file: line:19
struct PyCallBack_VPUNN_SHVLess : public VPUNN::SHVLess {
	using VPUNN::SHVLess::SHVLess;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVLess *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVLessEqual file: line:19
struct PyCallBack_VPUNN_SHVLessEqual : public VPUNN::SHVLessEqual {
	using VPUNN::SHVLessEqual::SHVLessEqual;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVLessEqual *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVGreater file: line:19
struct PyCallBack_VPUNN_SHVGreater : public VPUNN::SHVGreater {
	using VPUNN::SHVGreater::SHVGreater;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVGreater *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVGreaterEqual file: line:19
struct PyCallBack_VPUNN_SHVGreaterEqual : public VPUNN::SHVGreaterEqual {
	using VPUNN::SHVGreaterEqual::SHVGreaterEqual;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVGreaterEqual *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVLogicalOr file: line:19
struct PyCallBack_VPUNN_SHVLogicalOr : public VPUNN::SHVLogicalOr {
	using VPUNN::SHVLogicalOr::SHVLogicalOr;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVLogicalOr *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVLogicalNot file: line:19
struct PyCallBack_VPUNN_SHVLogicalNot : public VPUNN::SHVLogicalNot {
	using VPUNN::SHVLogicalNot::SHVLogicalNot;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVLogicalNot *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVLogicalXor file: line:19
struct PyCallBack_VPUNN_SHVLogicalXor : public VPUNN::SHVLogicalXor {
	using VPUNN::SHVLogicalXor::SHVLogicalXor;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVLogicalXor *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVMultiply file: line:19
struct PyCallBack_VPUNN_SHVMultiply : public VPUNN::SHVMultiply {
	using VPUNN::SHVMultiply::SHVMultiply;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVMultiply *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVAnd file: line:19
struct PyCallBack_VPUNN_SHVAnd : public VPUNN::SHVAnd {
	using VPUNN::SHVAnd::SHVAnd;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAnd *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVMinimum file: line:19
struct PyCallBack_VPUNN_SHVMinimum : public VPUNN::SHVMinimum {
	using VPUNN::SHVMinimum::SHVMinimum;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVMinimum *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVMaximum file: line:19
struct PyCallBack_VPUNN_SHVMaximum : public VPUNN::SHVMaximum {
	using VPUNN::SHVMaximum::SHVMaximum;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVMaximum *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVSubtract file: line:19
struct PyCallBack_VPUNN_SHVSubtract : public VPUNN::SHVSubtract {
	using VPUNN::SHVSubtract::SHVSubtract;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSubtract *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVNotEqual file: line:19
struct PyCallBack_VPUNN_SHVNotEqual : public VPUNN::SHVNotEqual {
	using VPUNN::SHVNotEqual::SHVNotEqual;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVNotEqual *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVEqual file: line:19
struct PyCallBack_VPUNN_SHVEqual : public VPUNN::SHVEqual {
	using VPUNN::SHVEqual::SHVEqual;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVEqual *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVElementwise::cycles();
	}
};

// VPUNN::SHVRoll file: line:19
struct PyCallBack_VPUNN_SHVRoll : public VPUNN::SHVRoll {
	using VPUNN::SHVRoll::SHVRoll;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVRoll *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVShuffleChannels file: line:19
struct PyCallBack_VPUNN_SHVShuffleChannels : public VPUNN::SHVShuffleChannels {
	using VPUNN::SHVShuffleChannels::SHVShuffleChannels;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVShuffleChannels *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVGather file: line:19
struct PyCallBack_VPUNN_SHVGather : public VPUNN::SHVGather {
	using VPUNN::SHVGather::SHVGather;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVGather *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVScatterNDUpdate file: line:19
struct PyCallBack_VPUNN_SHVScatterNDUpdate : public VPUNN::SHVScatterNDUpdate {
	using VPUNN::SHVScatterNDUpdate::SHVScatterNDUpdate;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVScatterNDUpdate *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVScatterUpdate file: line:19
struct PyCallBack_VPUNN_SHVScatterUpdate : public VPUNN::SHVScatterUpdate {
	using VPUNN::SHVScatterUpdate::SHVScatterUpdate;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVScatterUpdate *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVReshape file: line:19
struct PyCallBack_VPUNN_SHVReshape : public VPUNN::SHVReshape {
	using VPUNN::SHVReshape::SHVReshape;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVReshape *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVSqueeze file: line:19
struct PyCallBack_VPUNN_SHVSqueeze : public VPUNN::SHVSqueeze {
	using VPUNN::SHVSqueeze::SHVSqueeze;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSqueeze *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

void bind_VPUNN_30(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SHVDivide file: line:19
		pybind11::class_<VPUNN::SHVDivide, std::shared_ptr<VPUNN::SHVDivide>, PyCallBack_VPUNN_SHVDivide, VPUNN::SHVElementwise<12,11587>> cl(M("VPUNN"), "SHVDivide", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVDivide const &o){ return new PyCallBack_VPUNN_SHVDivide(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVDivide const &o){ return new VPUNN::SHVDivide(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSquaredDiff file: line:19
		pybind11::class_<VPUNN::SHVSquaredDiff, std::shared_ptr<VPUNN::SHVSquaredDiff>, PyCallBack_VPUNN_SHVSquaredDiff, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVSquaredDiff", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSquaredDiff const &o){ return new PyCallBack_VPUNN_SHVSquaredDiff(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSquaredDiff const &o){ return new VPUNN::SHVSquaredDiff(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVFloorMod file: line:19
		pybind11::class_<VPUNN::SHVFloorMod, std::shared_ptr<VPUNN::SHVFloorMod>, PyCallBack_VPUNN_SHVFloorMod, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVFloorMod", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVFloorMod const &o){ return new PyCallBack_VPUNN_SHVFloorMod(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVFloorMod const &o){ return new VPUNN::SHVFloorMod(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVLess file: line:19
		pybind11::class_<VPUNN::SHVLess, std::shared_ptr<VPUNN::SHVLess>, PyCallBack_VPUNN_SHVLess, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVLess", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVLess const &o){ return new PyCallBack_VPUNN_SHVLess(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVLess const &o){ return new VPUNN::SHVLess(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVLessEqual file: line:19
		pybind11::class_<VPUNN::SHVLessEqual, std::shared_ptr<VPUNN::SHVLessEqual>, PyCallBack_VPUNN_SHVLessEqual, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVLessEqual", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVLessEqual const &o){ return new PyCallBack_VPUNN_SHVLessEqual(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVLessEqual const &o){ return new VPUNN::SHVLessEqual(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVGreater file: line:19
		pybind11::class_<VPUNN::SHVGreater, std::shared_ptr<VPUNN::SHVGreater>, PyCallBack_VPUNN_SHVGreater, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVGreater", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVGreater const &o){ return new PyCallBack_VPUNN_SHVGreater(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVGreater const &o){ return new VPUNN::SHVGreater(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVGreaterEqual file: line:19
		pybind11::class_<VPUNN::SHVGreaterEqual, std::shared_ptr<VPUNN::SHVGreaterEqual>, PyCallBack_VPUNN_SHVGreaterEqual, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVGreaterEqual", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVGreaterEqual const &o){ return new PyCallBack_VPUNN_SHVGreaterEqual(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVGreaterEqual const &o){ return new VPUNN::SHVGreaterEqual(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVLogicalOr file: line:19
		pybind11::class_<VPUNN::SHVLogicalOr, std::shared_ptr<VPUNN::SHVLogicalOr>, PyCallBack_VPUNN_SHVLogicalOr, VPUNN::SHVElementwise<8,13192>> cl(M("VPUNN"), "SHVLogicalOr", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVLogicalOr const &o){ return new PyCallBack_VPUNN_SHVLogicalOr(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVLogicalOr const &o){ return new VPUNN::SHVLogicalOr(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVLogicalNot file: line:19
		pybind11::class_<VPUNN::SHVLogicalNot, std::shared_ptr<VPUNN::SHVLogicalNot>, PyCallBack_VPUNN_SHVLogicalNot, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVLogicalNot", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVLogicalNot const &o){ return new PyCallBack_VPUNN_SHVLogicalNot(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVLogicalNot const &o){ return new VPUNN::SHVLogicalNot(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVLogicalXor file: line:19
		pybind11::class_<VPUNN::SHVLogicalXor, std::shared_ptr<VPUNN::SHVLogicalXor>, PyCallBack_VPUNN_SHVLogicalXor, VPUNN::SHVElementwise<8,13036>> cl(M("VPUNN"), "SHVLogicalXor", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVLogicalXor const &o){ return new PyCallBack_VPUNN_SHVLogicalXor(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVLogicalXor const &o){ return new VPUNN::SHVLogicalXor(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVMultiply file: line:19
		pybind11::class_<VPUNN::SHVMultiply, std::shared_ptr<VPUNN::SHVMultiply>, PyCallBack_VPUNN_SHVMultiply, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVMultiply", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVMultiply const &o){ return new PyCallBack_VPUNN_SHVMultiply(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVMultiply const &o){ return new VPUNN::SHVMultiply(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAnd file: line:19
		pybind11::class_<VPUNN::SHVAnd, std::shared_ptr<VPUNN::SHVAnd>, PyCallBack_VPUNN_SHVAnd, VPUNN::SHVElementwise<45,30591>> cl(M("VPUNN"), "SHVAnd", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAnd const &o){ return new PyCallBack_VPUNN_SHVAnd(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAnd const &o){ return new VPUNN::SHVAnd(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVMinimum file: line:19
		pybind11::class_<VPUNN::SHVMinimum, std::shared_ptr<VPUNN::SHVMinimum>, PyCallBack_VPUNN_SHVMinimum, VPUNN::SHVElementwise<15,11047>> cl(M("VPUNN"), "SHVMinimum", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVMinimum const &o){ return new PyCallBack_VPUNN_SHVMinimum(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVMinimum const &o){ return new VPUNN::SHVMinimum(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVMaximum file: line:19
		pybind11::class_<VPUNN::SHVMaximum, std::shared_ptr<VPUNN::SHVMaximum>, PyCallBack_VPUNN_SHVMaximum, VPUNN::SHVElementwise<15,11000>> cl(M("VPUNN"), "SHVMaximum", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVMaximum const &o){ return new PyCallBack_VPUNN_SHVMaximum(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVMaximum const &o){ return new VPUNN::SHVMaximum(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSubtract file: line:19
		pybind11::class_<VPUNN::SHVSubtract, std::shared_ptr<VPUNN::SHVSubtract>, PyCallBack_VPUNN_SHVSubtract, VPUNN::SHVElementwise<789,12946>> cl(M("VPUNN"), "SHVSubtract", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSubtract const &o){ return new PyCallBack_VPUNN_SHVSubtract(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSubtract const &o){ return new VPUNN::SHVSubtract(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVNotEqual file: line:19
		pybind11::class_<VPUNN::SHVNotEqual, std::shared_ptr<VPUNN::SHVNotEqual>, PyCallBack_VPUNN_SHVNotEqual, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVNotEqual", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVNotEqual const &o){ return new PyCallBack_VPUNN_SHVNotEqual(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVNotEqual const &o){ return new VPUNN::SHVNotEqual(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVEqual file: line:19
		pybind11::class_<VPUNN::SHVEqual, std::shared_ptr<VPUNN::SHVEqual>, PyCallBack_VPUNN_SHVEqual, VPUNN::SHVElementwise<1000,0>> cl(M("VPUNN"), "SHVEqual", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVEqual const &o){ return new PyCallBack_VPUNN_SHVEqual(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVEqual const &o){ return new VPUNN::SHVEqual(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class std::vector<class VPUNN::VPUTensor, class std::allocator<class VPUNN::VPUTensor> > &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("inputs"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVRoll file: line:19
		pybind11::class_<VPUNN::SHVRoll, std::shared_ptr<VPUNN::SHVRoll>, PyCallBack_VPUNN_SHVRoll, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVRoll", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVRoll const &o){ return new PyCallBack_VPUNN_SHVRoll(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVRoll const &o){ return new VPUNN::SHVRoll(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVShuffleChannels file: line:19
		pybind11::class_<VPUNN::SHVShuffleChannels, std::shared_ptr<VPUNN::SHVShuffleChannels>, PyCallBack_VPUNN_SHVShuffleChannels, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVShuffleChannels", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVShuffleChannels const &o){ return new PyCallBack_VPUNN_SHVShuffleChannels(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVShuffleChannels const &o){ return new VPUNN::SHVShuffleChannels(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVGather file: line:19
		pybind11::class_<VPUNN::SHVGather, std::shared_ptr<VPUNN::SHVGather>, PyCallBack_VPUNN_SHVGather, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVGather", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVGather const &o){ return new PyCallBack_VPUNN_SHVGather(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVGather const &o){ return new VPUNN::SHVGather(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVScatterNDUpdate file: line:19
		pybind11::class_<VPUNN::SHVScatterNDUpdate, std::shared_ptr<VPUNN::SHVScatterNDUpdate>, PyCallBack_VPUNN_SHVScatterNDUpdate, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVScatterNDUpdate", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVScatterNDUpdate const &o){ return new PyCallBack_VPUNN_SHVScatterNDUpdate(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVScatterNDUpdate const &o){ return new VPUNN::SHVScatterNDUpdate(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVScatterUpdate file: line:19
		pybind11::class_<VPUNN::SHVScatterUpdate, std::shared_ptr<VPUNN::SHVScatterUpdate>, PyCallBack_VPUNN_SHVScatterUpdate, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVScatterUpdate", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVScatterUpdate const &o){ return new PyCallBack_VPUNN_SHVScatterUpdate(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVScatterUpdate const &o){ return new VPUNN::SHVScatterUpdate(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVReshape file: line:19
		pybind11::class_<VPUNN::SHVReshape, std::shared_ptr<VPUNN::SHVReshape>, PyCallBack_VPUNN_SHVReshape, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVReshape", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVReshape const &o){ return new PyCallBack_VPUNN_SHVReshape(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVReshape const &o){ return new VPUNN::SHVReshape(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSqueeze file: line:19
		pybind11::class_<VPUNN::SHVSqueeze, std::shared_ptr<VPUNN::SHVSqueeze>, PyCallBack_VPUNN_SHVSqueeze, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVSqueeze", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSqueeze const &o){ return new PyCallBack_VPUNN_SHVSqueeze(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSqueeze const &o){ return new VPUNN::SHVSqueeze(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
}


// File: VPUNN_31.cpp
#include <sstream> // __str__

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::SHVUnsqueeze file: line:19
struct PyCallBack_VPUNN_SHVUnsqueeze : public VPUNN::SHVUnsqueeze {
	using VPUNN::SHVUnsqueeze::SHVUnsqueeze;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVUnsqueeze *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVBroadcast file: line:19
struct PyCallBack_VPUNN_SHVBroadcast : public VPUNN::SHVBroadcast {
	using VPUNN::SHVBroadcast::SHVBroadcast;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVBroadcast *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVTranspose file: line:19
struct PyCallBack_VPUNN_SHVTranspose : public VPUNN::SHVTranspose {
	using VPUNN::SHVTranspose::SHVTranspose;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVTranspose *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVConcat file: line:19
struct PyCallBack_VPUNN_SHVConcat : public VPUNN::SHVConcat {
	using VPUNN::SHVConcat::SHVConcat;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVConcat *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVAffineReshape file: line:19
struct PyCallBack_VPUNN_SHVAffineReshape : public VPUNN::SHVAffineReshape {
	using VPUNN::SHVAffineReshape::SHVAffineReshape;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVAffineReshape *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVPermuteQuantize file: line:19
struct PyCallBack_VPUNN_SHVPermuteQuantize : public VPUNN::SHVPermuteQuantize {
	using VPUNN::SHVPermuteQuantize::SHVPermuteQuantize;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVPermuteQuantize *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVDepthToSpace file: line:19
struct PyCallBack_VPUNN_SHVDepthToSpace : public VPUNN::SHVDepthToSpace {
	using VPUNN::SHVDepthToSpace::SHVDepthToSpace;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVDepthToSpace *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVSpaceToDepthOp file: line:19
struct PyCallBack_VPUNN_SHVSpaceToDepthOp : public VPUNN::SHVSpaceToDepthOp {
	using VPUNN::SHVSpaceToDepthOp::SHVSpaceToDepthOp;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVSpaceToDepthOp *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVMemPermute file: line:19
struct PyCallBack_VPUNN_SHVMemPermute : public VPUNN::SHVMemPermute {
	using VPUNN::SHVMemPermute::SHVMemPermute;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVMemPermute *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

// VPUNN::SHVPermuteCast file: line:19
struct PyCallBack_VPUNN_SHVPermuteCast : public VPUNN::SHVPermuteCast {
	using VPUNN::SHVPermuteCast::SHVPermuteCast;

	unsigned int cycles() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::SHVPermuteCast *>(this), "cycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		return SHVDataMovement::cycles();
	}
};

void bind_VPUNN_31(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::SHVUnsqueeze file: line:19
		pybind11::class_<VPUNN::SHVUnsqueeze, std::shared_ptr<VPUNN::SHVUnsqueeze>, PyCallBack_VPUNN_SHVUnsqueeze, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVUnsqueeze", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVUnsqueeze const &o){ return new PyCallBack_VPUNN_SHVUnsqueeze(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVUnsqueeze const &o){ return new VPUNN::SHVUnsqueeze(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVBroadcast file: line:19
		pybind11::class_<VPUNN::SHVBroadcast, std::shared_ptr<VPUNN::SHVBroadcast>, PyCallBack_VPUNN_SHVBroadcast, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVBroadcast", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVBroadcast const &o){ return new PyCallBack_VPUNN_SHVBroadcast(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVBroadcast const &o){ return new VPUNN::SHVBroadcast(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVTranspose file: line:19
		pybind11::class_<VPUNN::SHVTranspose, std::shared_ptr<VPUNN::SHVTranspose>, PyCallBack_VPUNN_SHVTranspose, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVTranspose", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVTranspose const &o){ return new PyCallBack_VPUNN_SHVTranspose(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVTranspose const &o){ return new VPUNN::SHVTranspose(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVConcat file: line:19
		pybind11::class_<VPUNN::SHVConcat, std::shared_ptr<VPUNN::SHVConcat>, PyCallBack_VPUNN_SHVConcat, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVConcat", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVConcat const &o){ return new PyCallBack_VPUNN_SHVConcat(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVConcat const &o){ return new VPUNN::SHVConcat(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVAffineReshape file: line:19
		pybind11::class_<VPUNN::SHVAffineReshape, std::shared_ptr<VPUNN::SHVAffineReshape>, PyCallBack_VPUNN_SHVAffineReshape, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVAffineReshape", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVAffineReshape const &o){ return new PyCallBack_VPUNN_SHVAffineReshape(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVAffineReshape const &o){ return new VPUNN::SHVAffineReshape(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVPermuteQuantize file: line:19
		pybind11::class_<VPUNN::SHVPermuteQuantize, std::shared_ptr<VPUNN::SHVPermuteQuantize>, PyCallBack_VPUNN_SHVPermuteQuantize, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVPermuteQuantize", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVPermuteQuantize const &o){ return new PyCallBack_VPUNN_SHVPermuteQuantize(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVPermuteQuantize const &o){ return new VPUNN::SHVPermuteQuantize(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVDepthToSpace file: line:19
		pybind11::class_<VPUNN::SHVDepthToSpace, std::shared_ptr<VPUNN::SHVDepthToSpace>, PyCallBack_VPUNN_SHVDepthToSpace, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVDepthToSpace", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVDepthToSpace const &o){ return new PyCallBack_VPUNN_SHVDepthToSpace(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVDepthToSpace const &o){ return new VPUNN::SHVDepthToSpace(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVSpaceToDepthOp file: line:19
		pybind11::class_<VPUNN::SHVSpaceToDepthOp, std::shared_ptr<VPUNN::SHVSpaceToDepthOp>, PyCallBack_VPUNN_SHVSpaceToDepthOp, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVSpaceToDepthOp", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVSpaceToDepthOp const &o){ return new PyCallBack_VPUNN_SHVSpaceToDepthOp(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVSpaceToDepthOp const &o){ return new VPUNN::SHVSpaceToDepthOp(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVMemPermute file: line:19
		pybind11::class_<VPUNN::SHVMemPermute, std::shared_ptr<VPUNN::SHVMemPermute>, PyCallBack_VPUNN_SHVMemPermute, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVMemPermute", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVMemPermute const &o){ return new PyCallBack_VPUNN_SHVMemPermute(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVMemPermute const &o){ return new VPUNN::SHVMemPermute(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
	{ // VPUNN::SHVPermuteCast file: line:19
		pybind11::class_<VPUNN::SHVPermuteCast, std::shared_ptr<VPUNN::SHVPermuteCast>, PyCallBack_VPUNN_SHVPermuteCast, VPUNN::SHVDataMovement<1000,0>> cl(M("VPUNN"), "SHVPermuteCast", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_SHVPermuteCast const &o){ return new PyCallBack_VPUNN_SHVPermuteCast(o); } ) );
		cl.def( pybind11::init( [](VPUNN::SHVPermuteCast const &o){ return new VPUNN::SHVPermuteCast(o); } ) );
		cl.def( pybind11::init<const enum VPUNN::VPUDevice &, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &>(), pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output") );

	}
}


// File: VPUNN_32.cpp
#include <chrono> // std::chrono::_V2::system_clock
#include <chrono> // std::chrono::duration
#include <chrono> // std::chrono::time_point
#include <ratio> // std::ratio
#include <sstream> // __str__

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_32(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // std::chrono::duration file:chrono line:300
		pybind11::class_<std::chrono::duration<long,std::ratio<1, 1000000000>>, std::shared_ptr<std::chrono::duration<long,std::ratio<1, 1000000000>>>> cl(M("std::chrono"), "duration_long_std_ratio_1_1000000000_t", "");
		cl.def( pybind11::init( [](){ return new std::chrono::duration<long,std::ratio<1, 1000000000>>(); } ) );
		cl.def( pybind11::init( [](std::chrono::duration<long,std::ratio<1, 1000000000>> const &o){ return new std::chrono::duration<long,std::ratio<1, 1000000000>>(o); } ) );
		cl.def("assign", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator=, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator=(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
		cl.def("count", (long (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)() const) &std::chrono::duration<long, std::ratio<1, 1000000000> >::count, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::count() const --> long");
		cl.def("__pos__", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)() const) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator+, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator+() const --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >");
		cl.def("__neg__", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)() const) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator-, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator-() const --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >");
		cl.def("pre_increment", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)()) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator++, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator++() --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic);
		cl.def("post_increment", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(int)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator++, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator++(int) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >", pybind11::arg(""));
		cl.def("pre_decrement", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)()) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator--, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator--() --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic);
		cl.def("post_decrement", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(int)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator--, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator--(int) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >", pybind11::arg(""));
		cl.def("__iadd__", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator+=, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator+=(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic, pybind11::arg("__d"));
		cl.def("__isub__", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator-=, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator-=(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic, pybind11::arg("__d"));
		cl.def("__imul__", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(const long &)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator*=, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator*=(const long &) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic, pybind11::arg("__rhs"));
		cl.def("__itruediv__", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > & (std::chrono::duration<long,std::ratio<1, 1000000000>>::*)(const long &)) &std::chrono::duration<long, std::ratio<1, 1000000000> >::operator/=, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::operator/=(const long &) --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &", pybind11::return_value_policy::automatic, pybind11::arg("__rhs"));
		cl.def_static("zero", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (*)()) &std::chrono::duration<long, std::ratio<1, 1000000000> >::zero, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::zero() --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >");
		cl.def_static("min", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (*)()) &std::chrono::duration<long, std::ratio<1, 1000000000> >::min, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::min() --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >");
		cl.def_static("max", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (*)()) &std::chrono::duration<long, std::ratio<1, 1000000000> >::max, "C++: std::chrono::duration<long, std::ratio<1, 1000000000> >::max() --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >");
	}
	{ // std::chrono::time_point file:chrono line:626
		pybind11::class_<std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>, std::shared_ptr<std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>>> cl(M("std::chrono"), "time_point_std_chrono__V2_system_clock_std_chrono_duration_long_std_ratio_1_1000000000_t", "");
		cl.def( pybind11::init( [](){ return new std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>(); } ) );
		cl.def( pybind11::init<const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &>(), pybind11::arg("__dur") );

		cl.def( pybind11::init( [](std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >> const &o){ return new std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>(o); } ) );
		cl.def("time_since_epoch", (struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > (std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>::*)() const) &std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::time_since_epoch, "C++: std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::time_since_epoch() const --> struct std::chrono::duration<long, struct std::ratio<1, 1000000000> >");
		cl.def("__iadd__", (struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > & (std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>::*)(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &)) &std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::operator+=, "C++: std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::operator+=(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &) --> struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > &", pybind11::return_value_policy::automatic, pybind11::arg("__dur"));
		cl.def("__isub__", (struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > & (std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>::*)(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &)) &std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::operator-=, "C++: std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::operator-=(const struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > &) --> struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > &", pybind11::return_value_policy::automatic, pybind11::arg("__dur"));
		cl.def_static("min", (struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > (*)()) &std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::min, "C++: std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::min() --> struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > >");
		cl.def_static("max", (struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > (*)()) &std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::max, "C++: std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::max() --> struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > >");
		cl.def("assign", (struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > & (std::chrono::time_point<std::chrono::_V2::system_clock,std::chrono::duration<long, std::ratio<1, 1000000000> >>::*)(const struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > &)) &std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::operator=, "C++: std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<long, std::ratio<1, 1000000000> > >::operator=(const struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > &) --> struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: VPUNN_33.cpp
#include <chrono> // std::chrono::_V2::system_clock
#include <chrono> // std::chrono::duration
#include <chrono> // std::chrono::time_point
#include <ratio> // std::ratio

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_33(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::tick() file: line:22
	M("VPUNN").def("tick", (struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > > (*)()) &VPUNN::tick, "Get the current timestamp\n\n \n auto\n\nC++: VPUNN::tick() --> struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > >");

	// VPUNN::tock(struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > >) file: line:32
	M("VPUNN").def("tock", (double (*)(struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > >)) &VPUNN::tock, "Get the current timestamp and return the elapsed time with the previous timestamp\n\n \n the previous timestamp\n \n\n double the elapsed time in millisecond\n\nC++: VPUNN::tock(struct std::chrono::time_point<struct std::chrono::_V2::system_clock, struct std::chrono::duration<long, struct std::ratio<1, 1000000000> > >) --> double", pybind11::arg("t1"));

}


// File: VPUNN_34.cpp
#include <ios> // std::_Ios_Seekdir
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::char_traits
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_34(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::Tensor file: line:31
		pybind11::class_<VPUNN::Tensor<float>, std::shared_ptr<VPUNN::Tensor<float>>> cl(M("VPUNN"), "Tensor_float_t", "");
		cl.def( pybind11::init<const class std::vector<unsigned int, class std::allocator<unsigned int> > &>(), pybind11::arg("dimensions") );

		cl.def( pybind11::init<float *, const class std::vector<unsigned int, class std::allocator<unsigned int> > &>(), pybind11::arg("data"), pybind11::arg("dimensions") );

		cl.def( pybind11::init<const class std::vector<unsigned int, class std::allocator<unsigned int> > &, float>(), pybind11::arg("dimensions"), pybind11::arg("value") );

		cl.def( pybind11::init( [](VPUNN::Tensor<float> const &o){ return new VPUNN::Tensor<float>(o); } ) );
		cl.def("assign", (class VPUNN::Tensor<float> & (VPUNN::Tensor<float>::*)(const class VPUNN::Tensor<float> &)) &VPUNN::Tensor<float>::operator=, "C++: VPUNN::Tensor<float>::operator=(const class VPUNN::Tensor<float> &) --> class VPUNN::Tensor<float> &", pybind11::return_value_policy::automatic, pybind11::arg("tensor"));
		cl.def("data", (float * (VPUNN::Tensor<float>::*)()) &VPUNN::Tensor<float>::data, "C++: VPUNN::Tensor<float>::data() --> float *", pybind11::return_value_policy::automatic);
		cl.def("data_vector", (class std::vector<float, class std::allocator<float> > (VPUNN::Tensor<float>::*)()) &VPUNN::Tensor<float>::data_vector, "C++: VPUNN::Tensor<float>::data_vector() --> class std::vector<float, class std::allocator<float> >");
		cl.def("c_ptr", (const float * (VPUNN::Tensor<float>::*)() const) &VPUNN::Tensor<float>::c_ptr, "C++: VPUNN::Tensor<float>::c_ptr() const --> const float *", pybind11::return_value_policy::automatic);
		cl.def("assign", (class VPUNN::Tensor<float> & (VPUNN::Tensor<float>::*)(const float *, const unsigned int)) &VPUNN::Tensor<float>::assign, "C++: VPUNN::Tensor<float>::assign(const float *, const unsigned int) --> class VPUNN::Tensor<float> &", pybind11::return_value_policy::automatic, pybind11::arg("data"), pybind11::arg("size_in_bytes"));
		cl.def("__getitem__", (float & (VPUNN::Tensor<float>::*)(const int)) &VPUNN::Tensor<float>::operator[], "C++: VPUNN::Tensor<float>::operator[](const int) --> float &", pybind11::return_value_policy::automatic, pybind11::arg("idx"));
		cl.def("size", (int (VPUNN::Tensor<float>::*)() const) &VPUNN::Tensor<float>::size, "C++: VPUNN::Tensor<float>::size() const --> int");
		cl.def("shape", (const class std::vector<unsigned int, class std::allocator<unsigned int> > & (VPUNN::Tensor<float>::*)() const) &VPUNN::Tensor<float>::shape, "C++: VPUNN::Tensor<float>::shape() const --> const class std::vector<unsigned int, class std::allocator<unsigned int> > &", pybind11::return_value_policy::automatic);
		cl.def("fill", (class VPUNN::Tensor<float> & (VPUNN::Tensor<float>::*)(float)) &VPUNN::Tensor<float>::fill, "C++: VPUNN::Tensor<float>::fill(float) --> class VPUNN::Tensor<float> &", pybind11::return_value_policy::automatic, pybind11::arg("value"));

		cl.def("__str__", [](VPUNN::Tensor<float> const &o) -> std::string { std::ostringstream s; operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::BiasOp file: line:18
		pybind11::class_<VPUNN::BiasOp, std::shared_ptr<VPUNN::BiasOp>> cl(M("VPUNN"), "BiasOp", "Floating point bias layer (float). The instance has a helper memory for the constant buffer");
		cl.def( pybind11::init( [](){ return new VPUNN::BiasOp(); } ) );
		cl.def( pybind11::init( [](VPUNN::BiasOp const &o){ return new VPUNN::BiasOp(o); } ) );
		cl.def("Bias", (void (VPUNN::BiasOp::*)(const class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *) const) &VPUNN::BiasOp::Bias, "Floating point bias layer (float). The operation is done implace in the output tensor\n\n \n a VPUNN::Tensor containing the bias\n \n\n the input/output tensor\n\nC++: VPUNN::BiasOp::Bias(const class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *) const --> void", pybind11::arg("bias"), pybind11::arg("output"));
		cl.def("reserve_bias_space", [](VPUNN::BiasOp &o, int const & a0) -> void { return o.reserve_bias_space(a0); }, "", pybind11::arg("space_required"));
		cl.def("reserve_bias_space", (void (VPUNN::BiasOp::*)(int, float)) &VPUNN::BiasOp::reserve_bias_space, "ensures a minimim allocated space for the constant bias buffer\n\nC++: VPUNN::BiasOp::reserve_bias_space(int, float) --> void", pybind11::arg("space_required"), pybind11::arg("fill_value"));
		cl.def("assign", (class VPUNN::BiasOp & (VPUNN::BiasOp::*)(const class VPUNN::BiasOp &)) &VPUNN::BiasOp::operator=, "C++: VPUNN::BiasOp::operator=(const class VPUNN::BiasOp &) --> class VPUNN::BiasOp &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// VPUNN::Dense(const class VPUNN::Tensor<float> *, const class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *) file: line:25
	M("VPUNN").def("Dense", (void (*)(const class VPUNN::Tensor<float> *, const class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *)) &VPUNN::Dense, "Floating point FC layer (float)\n\n \n a VPUNN::Tensor containing the FC layer weights\n \n\n the input tensor\n \n\n the output tensor\n\nC++: VPUNN::Dense(const class VPUNN::Tensor<float> *, const class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *) --> void", pybind11::arg("weights"), pybind11::arg("activations"), pybind11::arg("output"));

	// VPUNN::kNN(class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, unsigned int) file: line:27
	M("VPUNN").def("kNN", [](class VPUNN::Tensor<float> * a0, class VPUNN::Tensor<float> * a1, class VPUNN::Tensor<float> * a2, class VPUNN::Tensor<float> * a3) -> void { return VPUNN::kNN(a0, a1, a2, a3); }, "", pybind11::arg("weights"), pybind11::arg("targets"), pybind11::arg("activations"), pybind11::arg("output"));
	M("VPUNN").def("kNN", (void (*)(class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, unsigned int)) &VPUNN::kNN, "Floating point k-Nearest-Neighbor (kNN) layer (float)\n\n \n a VPUNN::Tensor containing the kNN layer weights\n \n\n a VPUNN::Tensor containing the kNN layer targets\n \n\n the input tensor\n \n\n the output tensor\n \n\n number of neighbors to consider. must be >=1\n\nC++: VPUNN::kNN(class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *, unsigned int) --> void", pybind11::arg("weights"), pybind11::arg("targets"), pybind11::arg("activations"), pybind11::arg("output"), pybind11::arg("n_neighbours"));

	// VPUNN::L2Normalization(class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *) file: line:23
	M("VPUNN").def("L2Normalization", (void (*)(class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *)) &VPUNN::L2Normalization, "Floating point L2 normalization layer (float)\n\n \n the input tensor\n \n\n the output tensor\n\nC++: VPUNN::L2Normalization(class VPUNN::Tensor<float> *, class VPUNN::Tensor<float> *) --> void", pybind11::arg("activations"), pybind11::arg("output"));

	// VPUNN::Sigmoid(class VPUNN::Tensor<float> *) file: line:22
	M("VPUNN").def("Sigmoid", (void (*)(class VPUNN::Tensor<float> *)) &VPUNN::Sigmoid, "Floating point sigmoid layer (float). The operation is done implace\n\n \n the input/output tensor\n\nC++: VPUNN::Sigmoid(class VPUNN::Tensor<float> *) --> void", pybind11::arg("output"));

}


// File: VPUNN_35.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_35(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::InferenceModel file: line:37
		pybind11::class_<VPUNN::InferenceModel, std::shared_ptr<VPUNN::InferenceModel>> cl(M("VPUNN"), "InferenceModel", "VPUNN inference model\n After creation can be used only if the model was initialized, otherwise will crash\n\n ");
		cl.def( pybind11::init<const char *>(), pybind11::arg("filename") );

		cl.def( pybind11::init<const char *, unsigned long, bool>(), pybind11::arg("data"), pybind11::arg("length"), pybind11::arg("with_copy") );

		cl.def( pybind11::init( [](VPUNN::InferenceModel const &o){ return new VPUNN::InferenceModel(o); } ) );
		cl.def("set_inputs", (void (VPUNN::InferenceModel::*)(const float *, const unsigned int)) &VPUNN::InferenceModel::set_inputs<float>, "C++: VPUNN::InferenceModel::set_inputs(const float *, const unsigned int) --> void", pybind11::arg("inputs"), pybind11::arg("size"));
		cl.def("get_outputs", (const float * (VPUNN::InferenceModel::*)()) &VPUNN::InferenceModel::get_outputs<float>, "C++: VPUNN::InferenceModel::get_outputs() --> const float *", pybind11::return_value_policy::automatic);
		cl.def("get_outputs_vector", (const class std::vector<float, class std::allocator<float> > (VPUNN::InferenceModel::*)()) &VPUNN::InferenceModel::get_outputs_vector<float>, "C++: VPUNN::InferenceModel::get_outputs_vector() --> const class std::vector<float, class std::allocator<float> >");
		cl.def("network_name", (std::string (VPUNN::InferenceModel::*)()) &VPUNN::InferenceModel::network_name, "Name of the network stored in the model\n\n \n unaltered name of the model as it is stored in the source/filename of the loaded model\n\nC++: VPUNN::InferenceModel::network_name() --> std::string");
		cl.def("is_initialized", (bool (VPUNN::InferenceModel::*)() const) &VPUNN::InferenceModel::is_initialized, "Check if the NN model is initialized\n\n \n true if the NN model is initialized\n \n\n false if the NN model is not initialized\n\nC++: VPUNN::InferenceModel::is_initialized() const --> bool");
		cl.def("allocate_tensors", (void (VPUNN::InferenceModel::*)(const unsigned int)) &VPUNN::InferenceModel::allocate_tensors, "Creates/Allocates the activation and weights buffer in memory\n\n \n the VPUNN inference batch size\n \n\n runtime_error in case tensors and buffers have a mismatch problem\n\nC++: VPUNN::InferenceModel::allocate_tensors(const unsigned int) --> void", pybind11::arg("batch"));
		cl.def("input_tensors", (class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> > (VPUNN::InferenceModel::*)()) &VPUNN::InferenceModel::input_tensors, "Get the VPUNN input tensors\n\n \n std::vector<Tensor<float>*> the NN input tensors\n\nC++: VPUNN::InferenceModel::input_tensors() --> class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> >");
		cl.def("output_tensors", (class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> > (VPUNN::InferenceModel::*)()) &VPUNN::InferenceModel::output_tensors, "Get the VPUNN output tensors\n\n \n std::vector<Tensor<float>*> the NN output tensors\n\nC++: VPUNN::InferenceModel::output_tensors() --> class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> >");
		cl.def("predict", (void (VPUNN::InferenceModel::*)()) &VPUNN::InferenceModel::predict, "Run the inference\n\n     \n\nC++: VPUNN::InferenceModel::predict() --> void");
	}
	{ // VPUNN::ModelVersion file: line:205
		pybind11::class_<VPUNN::ModelVersion, std::shared_ptr<VPUNN::ModelVersion>> cl(M("VPUNN"), "ModelVersion", "Extracts and keeps  the version out of a NN raw name");
		cl.def( pybind11::init( [](){ return new VPUNN::ModelVersion(); } ) );
		cl.def( pybind11::init( [](VPUNN::ModelVersion const &o){ return new VPUNN::ModelVersion(o); } ) );
		cl.def("get_input_interface_version", (int (VPUNN::ModelVersion::*)() const) &VPUNN::ModelVersion::get_input_interface_version, "value of input interface version, the one for the input descriptor\n\nC++: VPUNN::ModelVersion::get_input_interface_version() const --> int");
		cl.def("get_output_interface_version", (int (VPUNN::ModelVersion::*)() const) &VPUNN::ModelVersion::get_output_interface_version, "value of output interface version, the one for the NN provided value(s)\n\nC++: VPUNN::ModelVersion::get_output_interface_version() const --> int");
		cl.def("get_NN_name", (std::string (VPUNN::ModelVersion::*)() const) &VPUNN::ModelVersion::get_NN_name, "name of the NN, without version info\n\nC++: VPUNN::ModelVersion::get_NN_name() const --> std::string");
		cl.def("get_raw_name", (std::string (VPUNN::ModelVersion::*)() const) &VPUNN::ModelVersion::get_raw_name, "initial, raw name of the VPUNN model. contains version info\n\nC++: VPUNN::ModelVersion::get_raw_name() const --> std::string");
		cl.def("parse_name", (void (VPUNN::ModelVersion::*)(const std::string &)) &VPUNN::ModelVersion::parse_name, "parsed the name and extracts the information\n get the name based on separator \"-\", template: NNNNNNN-VI-VO\n VI and VO must be integers, and only the integers part will be considered when converting\n NNNNNN cannot be empty, it will be replaced with \"none\" if empty\n Only first three parts of the name are considered, rest are ignored.\n Missing pars will be considered default: none-1-1\n\n \n invalid_argument, out_of_range\n\nC++: VPUNN::ModelVersion::parse_name(const std::string &) --> void", pybind11::arg("raw_NN_name"));
	}
	// VPUNN::NNOutputVersions file: line:295
	pybind11::enum_<VPUNN::NNOutputVersions>(M("VPUNN"), "NNOutputVersions", "enum for NN output versions")
		.value("OUT_LATEST", VPUNN::NNOutputVersions::OUT_LATEST)
		.value("OUT_HW_OVERHEAD_BOUNDED", VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_BOUNDED)
		.value("OUT_CYCLES", VPUNN::NNOutputVersions::OUT_CYCLES)
		.value("OUT_HW_OVERHEAD_UNBOUNDED", VPUNN::NNOutputVersions::OUT_HW_OVERHEAD_UNBOUNDED);

;

	{ // VPUNN::PostProcessSupport file: line:308
		pybind11::class_<VPUNN::PostProcessSupport, std::shared_ptr<VPUNN::PostProcessSupport>> cl(M("VPUNN"), "PostProcessSupport", "Configuration options concerning the interpretation and post processing of inferred values\n This class have the goal to check if we know something about the output version of the model and if we don't know we\n will not support the output for the CostModel. We are going to use the output version parsed by the ModelVersion and\n use it to determine based on known output version if we support it or not. In case that we don't know the version we\n are going to not supprt the output.");
		cl.def( pybind11::init<int>(), pybind11::arg("output_version") );

		cl.def( pybind11::init( [](VPUNN::PostProcessSupport const &o){ return new VPUNN::PostProcessSupport(o); } ) );
		cl.def("is_output_supported", (bool (VPUNN::PostProcessSupport::*)() const) &VPUNN::PostProcessSupport::is_output_supported, "a method to see if we support the output\n\nC++: VPUNN::PostProcessSupport::is_output_supported() const --> bool");
	}
	{ // VPUNN::Runtime file: line:24
		pybind11::class_<VPUNN::Runtime, std::shared_ptr<VPUNN::Runtime>> cl(M("VPUNN"), "Runtime", "VPUNN runtime model\n\n ");
		cl.def( pybind11::init( [](const std::string & a0){ return new VPUNN::Runtime(a0); } ), "doc" , pybind11::arg("filename"));
		cl.def( pybind11::init( [](const std::string & a0, const unsigned int & a1){ return new VPUNN::Runtime(a0, a1); } ), "doc" , pybind11::arg("filename"), pybind11::arg("batch"));
		cl.def( pybind11::init<const std::string &, const unsigned int, bool>(), pybind11::arg("filename"), pybind11::arg("batch"), pybind11::arg("profile") );

		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2){ return new VPUNN::Runtime(a0, a1, a2); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, const unsigned int & a3){ return new VPUNN::Runtime(a0, a1, a2, a3); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("batch"));
		cl.def( pybind11::init<const char *, unsigned long, bool, const unsigned int, bool>(), pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("batch"), pybind11::arg("profile") );

		cl.def( pybind11::init( [](VPUNN::Runtime const &o){ return new VPUNN::Runtime(o); } ) );
		cl.def("predict", (const float * (VPUNN::Runtime::*)(const float *, const unsigned int)) &VPUNN::Runtime::predict<float>, "C++: VPUNN::Runtime::predict(const float *, const unsigned int) --> const float *", pybind11::return_value_policy::automatic, pybind11::arg("input_array"), pybind11::arg("input_size"));
		cl.def("predict", (const class std::vector<float, class std::allocator<float> > (VPUNN::Runtime::*)(const class std::vector<float, class std::allocator<float> > &)) &VPUNN::Runtime::predict<float>, "C++: VPUNN::Runtime::predict(const class std::vector<float, class std::allocator<float> > &) --> const class std::vector<float, class std::allocator<float> >", pybind11::arg("input_tensor"));
		cl.def("model_version_info", (const class VPUNN::ModelVersion & (VPUNN::Runtime::*)()) &VPUNN::Runtime::model_version_info, "provides version info for loaded model\n The provided reference is always up to date with the loaded model\n The info are conditioned(otherwise default) by the successful loading of the model.\n\n \n a long lived reference to the version information\n\nC++: VPUNN::Runtime::model_version_info() --> const class VPUNN::ModelVersion &", pybind11::return_value_policy::automatic);
		cl.def("initialized", (bool (VPUNN::Runtime::*)() const) &VPUNN::Runtime::initialized, "Check if the NN model is initialized\n\n \n true if the NN model is initialized\n \n\n false if the NN model is not initialized\n\nC++: VPUNN::Runtime::initialized() const --> bool");
		cl.def("input_tensors", (class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> > (VPUNN::Runtime::*)()) &VPUNN::Runtime::input_tensors, "Get the model input tensors\n\n \n std::vector<Tensor<float>*>\n\nC++: VPUNN::Runtime::input_tensors() --> class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> >");
		cl.def("output_tensors", (class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> > (VPUNN::Runtime::*)()) &VPUNN::Runtime::output_tensors, "Get the model output tensors\n\n \n std::vector<Tensor<float>*>\n\nC++: VPUNN::Runtime::output_tensors() --> class std::vector<class VPUNN::Tensor<float> *, class std::allocator<class VPUNN::Tensor<float> *> >");
		cl.def("input_shapes", (class std::vector<class std::vector<unsigned int, class std::allocator<unsigned int> >, class std::allocator<class std::vector<unsigned int, class std::allocator<unsigned int> > > > (VPUNN::Runtime::*)()) &VPUNN::Runtime::input_shapes, "Get the model input tensors shapes\n\n \n std::vector<std::vector<unsigned int>>\n\nC++: VPUNN::Runtime::input_shapes() --> class std::vector<class std::vector<unsigned int, class std::allocator<unsigned int> >, class std::allocator<class std::vector<unsigned int, class std::allocator<unsigned int> > > >");
		cl.def("output_shapes", (class std::vector<class std::vector<unsigned int, class std::allocator<unsigned int> >, class std::allocator<class std::vector<unsigned int, class std::allocator<unsigned int> > > > (VPUNN::Runtime::*)()) &VPUNN::Runtime::output_shapes, "Get the model output tensors shapes\n\n \n std::vector<std::vector<unsigned int>>\n\nC++: VPUNN::Runtime::output_shapes() --> class std::vector<class std::vector<unsigned int, class std::allocator<unsigned int> >, class std::allocator<class std::vector<unsigned int, class std::allocator<unsigned int> > > >");
	}
	// VPUNN::get_dpu_fclk(enum VPUNN::VPUDevice) file: line:29
	M("VPUNN").def("get_dpu_fclk", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_dpu_fclk, "Get the DPU default frequency in MHz\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::get_dpu_fclk(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::get_cmx_fclk(enum VPUNN::VPUDevice) file: line:50
	M("VPUNN").def("get_cmx_fclk", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_cmx_fclk, "Get the CMX default frequency in MHz\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::get_cmx_fclk(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::get_cmx_word_size_bytes(enum VPUNN::VPUDevice) file: line:71
	M("VPUNN").def("get_cmx_word_size_bytes", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_cmx_word_size_bytes, "Get CMX word size in bytes\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::get_cmx_word_size_bytes(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::get_dpu_cmx_num_read_ports(enum VPUNN::VPUDevice) file: line:90
	M("VPUNN").def("get_dpu_cmx_num_read_ports", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_dpu_cmx_num_read_ports, "Get DPU number of CMX read ports\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::get_dpu_cmx_num_read_ports(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::get_dram_bandwidth_MBps(enum VPUNN::VPUDevice) file: line:110
	M("VPUNN").def("get_dram_bandwidth_MBps", (float (*)(enum VPUNN::VPUDevice)) &VPUNN::get_dram_bandwidth_MBps, "Get the DRAM bandwidth in MB/s for a specific VPU IP\n\n \n a VPUDevice\n \n\n float\n\nC++: VPUNN::get_dram_bandwidth_MBps(enum VPUNN::VPUDevice) --> float", pybind11::arg("device"));

	// VPUNN::get_sram_word_size(bool) file: line:130
	M("VPUNN").def("get_sram_word_size", (unsigned int (*)(bool)) &VPUNN::get_sram_word_size, "Get the sram word size\n\n \n if compression is enabled or not\n \n\n unsigned int\n\nC++: VPUNN::get_sram_word_size(bool) --> unsigned int", pybind11::arg("compression"));

	// VPUNN::get_sram_word_size(const class VPUNN::VPUTensor &, bool, bool) file: line:142
	M("VPUNN").def("get_sram_word_size", (unsigned int (*)(const class VPUNN::VPUTensor &, bool, bool)) &VPUNN::get_sram_word_size, "Get the sram word size\n\n \n a VPUTensor\n \n\n if compression is enabled or not\n \n\n if a permute operation is required\n \n\n  unsigned int\n\nC++: VPUNN::get_sram_word_size(const class VPUNN::VPUTensor &, bool, bool) --> unsigned int", pybind11::arg("tensor"), pybind11::arg("compression"), pybind11::arg("permute"));

	// VPUNN::get_bandwidth_cycles_per_bytes(const class VPUNN::VPUTensor &, enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation, bool, bool) file: line:161
	M("VPUNN").def("get_bandwidth_cycles_per_bytes", [](const class VPUNN::VPUTensor & a0, enum VPUNN::VPUDevice const & a1, enum VPUNN::MemoryLocation const & a2) -> float { return VPUNN::get_bandwidth_cycles_per_bytes(a0, a1, a2); }, "", pybind11::arg("tensor"), pybind11::arg("device"), pybind11::arg("location"));
	M("VPUNN").def("get_bandwidth_cycles_per_bytes", [](const class VPUNN::VPUTensor & a0, enum VPUNN::VPUDevice const & a1, enum VPUNN::MemoryLocation const & a2, bool const & a3) -> float { return VPUNN::get_bandwidth_cycles_per_bytes(a0, a1, a2, a3); }, "", pybind11::arg("tensor"), pybind11::arg("device"), pybind11::arg("location"), pybind11::arg("compression"));
	M("VPUNN").def("get_bandwidth_cycles_per_bytes", (float (*)(const class VPUNN::VPUTensor &, enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation, bool, bool)) &VPUNN::get_bandwidth_cycles_per_bytes, "Get the DMA bandwidth in DPU cycles/bytes for a specific VPU IP\n\n \n a VPUTensor\n \n\n a VPUDevice\n \n\n a memory location\n \n\n is compression enabled\n \n\n if a permute operation is required\n \n\n float\n\nC++: VPUNN::get_bandwidth_cycles_per_bytes(const class VPUNN::VPUTensor &, enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation, bool, bool) --> float", pybind11::arg("tensor"), pybind11::arg("device"), pybind11::arg("location"), pybind11::arg("compression"), pybind11::arg("permute"));

	// VPUNN::get_bandwidth_MBps(const class VPUNN::VPUTensor &, enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation, bool, bool) file: line:184
	M("VPUNN").def("get_bandwidth_MBps", [](const class VPUNN::VPUTensor & a0, enum VPUNN::VPUDevice const & a1, enum VPUNN::MemoryLocation const & a2) -> float { return VPUNN::get_bandwidth_MBps(a0, a1, a2); }, "", pybind11::arg("tensor"), pybind11::arg("device"), pybind11::arg("location"));
	M("VPUNN").def("get_bandwidth_MBps", [](const class VPUNN::VPUTensor & a0, enum VPUNN::VPUDevice const & a1, enum VPUNN::MemoryLocation const & a2, bool const & a3) -> float { return VPUNN::get_bandwidth_MBps(a0, a1, a2, a3); }, "", pybind11::arg("tensor"), pybind11::arg("device"), pybind11::arg("location"), pybind11::arg("compression"));
	M("VPUNN").def("get_bandwidth_MBps", (float (*)(const class VPUNN::VPUTensor &, enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation, bool, bool)) &VPUNN::get_bandwidth_MBps, "Get the DMA bandwidth in MB/s for a specific VPU IP\n\n \n a VPUTensor\n \n\n a VPUDevice\n \n\n a memory location\n \n\n is compression enabled\n \n\n if a permute operation is required\n \n\n float\n\nC++: VPUNN::get_bandwidth_MBps(const class VPUNN::VPUTensor &, enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation, bool, bool) --> float", pybind11::arg("tensor"), pybind11::arg("device"), pybind11::arg("location"), pybind11::arg("compression"), pybind11::arg("permute"));

}


// File: VPUNN_36.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <sstream> // __str__
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::ShaveOpExecutor file: line:21
struct PyCallBack_VPUNN_ShaveOpExecutor : public VPUNN::ShaveOpExecutor {
	using VPUNN::ShaveOpExecutor::ShaveOpExecutor;

	unsigned int dpuCycles(const class VPUNN::SHAVEWorkload & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ShaveOpExecutor *>(this), "dpuCycles");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<unsigned int>::value) {
				static pybind11::detail::override_caster_t<unsigned int> caster;
				return pybind11::detail::cast_ref<unsigned int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<unsigned int>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"ShaveOpExecutor::dpuCycles\"");
	}
	std::string getName() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ShaveOpExecutor *>(this), "getName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<std::string>::value) {
				static pybind11::detail::override_caster_t<std::string> caster;
				return pybind11::detail::cast_ref<std::string>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<std::string>(std::move(o));
		}
		return ShaveOpExecutor::getName();
	}
};

void bind_VPUNN_36(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::get_DMA_latency(enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation) file: line:198
	M("VPUNN").def("get_DMA_latency", (unsigned int (*)(enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation)) &VPUNN::get_DMA_latency, "Get the DMA latency in DPU cycles\n\n \n a VPUDevice\n \n\n what memory is used\n \n\n CyclesInterfaceType\n\nC++: VPUNN::get_DMA_latency(enum VPUNN::VPUDevice, enum VPUNN::MemoryLocation) --> unsigned int", pybind11::arg("device"), pybind11::arg("location"));

	// VPUNN::get_nr_macs(enum VPUNN::VPUDevice) file: line:237
	M("VPUNN").def("get_nr_macs", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_nr_macs, "Get the DPU number of MACs\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::get_nr_macs(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::get_fp_ratio(enum VPUNN::VPUDevice) file: line:256
	M("VPUNN").def("get_fp_ratio", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_fp_ratio, "Get the ratio of int compute to fp16 compute\n\n \n a VPUDevice\n \n\n\n\n\nC++: VPUNN::get_fp_ratio(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::get_nr_ppe(enum VPUNN::VPUDevice) file: line:272
	M("VPUNN").def("get_nr_ppe", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_nr_ppe, "Get the number of PPE\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::get_nr_ppe(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::native_comp_is_fp(const struct VPUNN::DPUWorkload &) file: line:291
	M("VPUNN").def("native_comp_is_fp", (bool (*)(const struct VPUNN::DPUWorkload &)) &VPUNN::native_comp_is_fp, "Determine whether native computation for workload is floating point or int\n\n \n a DPUWorkload\n \n\n bool\n\nC++: VPUNN::native_comp_is_fp(const struct VPUNN::DPUWorkload &) --> bool", pybind11::arg("wl"));

	// VPUNN::input_channels_mac(enum VPUNN::VPUDevice) file: line:306
	M("VPUNN").def("input_channels_mac", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::input_channels_mac, "Get the MAC/input channels/cycles for a specific VPU IP\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::input_channels_mac(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	// VPUNN::nDPU_per_tile(enum VPUNN::VPUDevice) file: line:323
	M("VPUNN").def("nDPU_per_tile", (unsigned int (*)(enum VPUNN::VPUDevice)) &VPUNN::nDPU_per_tile, "Get the MAC/input channels/cycles for a specific VPU IP\n\n \n a VPUDevice\n \n\n unsigned int\n\nC++: VPUNN::nDPU_per_tile(enum VPUNN::VPUDevice) --> unsigned int", pybind11::arg("device"));

	{ // VPUNN::VPUNNPerformanceModel file: line:337
		pybind11::class_<VPUNN::VPUNNPerformanceModel, std::shared_ptr<VPUNN::VPUNNPerformanceModel>> cl(M("VPUNN"), "VPUNNPerformanceModel", "VPUNN performance model\n\n ");
		cl.def( pybind11::init( [](VPUNN::VPUNNPerformanceModel const &o){ return new VPUNN::VPUNNPerformanceModel(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::VPUNNPerformanceModel(); } ) );
		cl.def("DPU_Power_IdealCycles", (unsigned long (VPUNN::VPUNNPerformanceModel::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::VPUNNPerformanceModel::DPU_Power_IdealCycles, "Compute the DPU ideal cycles, considers HW optimizations like sparsity\n \n\n Calculates cycles that a single issue scalar CPU would require to execute\n a DPUWorkload then divides by number of MACs which can be performed in\n parallel by DPU. All operations are base-lined in the same manner with no\n non ideal factors considered at all.\n Like: Number of cycles if all the MAC resources are used 100%.\n Sparsity is considered only for weights!\n\n \n a DPUWorkload\n \n\n  ideal execution DPU cycles\n\nC++: VPUNN::VPUNNPerformanceModel::DPU_Power_IdealCycles(const struct VPUNN::DPUWorkload &) const --> unsigned long", pybind11::arg("wl"));
		cl.def("DPU_Efficency_IdealCycles", (unsigned long (VPUNN::VPUNNPerformanceModel::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::VPUNNPerformanceModel::DPU_Efficency_IdealCycles, "Compute the DPU ideal cycles, pure MAC based, no hw optimizations\n \n\n Calculates cycles that a single issue scalar CPU would require to execute\n a DPUWorkload then divides by number of MACs which can be performed in\n parallel by DPU. All operations are base-lined in the same manner with no\n non ideal factors considered at all.\n Like: Number of cycles if all the MAC resources are used 100%.\n\n \n a DPUWorkload\n \n\n  ideal execution DPU cycles\n\nC++: VPUNN::VPUNNPerformanceModel::DPU_Efficency_IdealCycles(const struct VPUNN::DPUWorkload &) const --> unsigned long", pybind11::arg("wl"));
		cl.def("DPUTheoreticalCycles", (unsigned long (VPUNN::VPUNNPerformanceModel::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::VPUNNPerformanceModel::DPUTheoreticalCycles, "Compute the DPU theoretical cycles, maximum HW knowledge\n \n\n Calculates cycles that a single issue scalar CPU would require to execute\n          a DPUWorkload then divides by number of MACs which can be performed in\n          parallel by DPU. Also considers data type, CMX memory bandwidth and some\n          other (non-ideal) factors.\n NO sparsity is considered.\n \n\n a DPUWorkload\n \n\n unsigned long int theoretical execution cycles\n\nC++: VPUNN::VPUNNPerformanceModel::DPUTheoreticalCycles(const struct VPUNN::DPUWorkload &) const --> unsigned long", pybind11::arg("wl"));
		cl.def("DMATheoreticalCycles", (unsigned long (VPUNN::VPUNNPerformanceModel::*)(const struct VPUNN::DMAWorkload &) const) &VPUNN::VPUNNPerformanceModel::DMATheoreticalCycles, "Compute the DMA theoretical cycles\n\n \n a DMAWorkload\n \n\n unsigned long int theoretical execution cycles\n\nC++: VPUNN::VPUNNPerformanceModel::DMATheoreticalCycles(const struct VPUNN::DMAWorkload &) const --> unsigned long", pybind11::arg("wl"));
		cl.def("SHAVETheoreticalCycles", (unsigned int (VPUNN::VPUNNPerformanceModel::*)(const struct VPUNN::SWOperation &)) &VPUNN::VPUNNPerformanceModel::SHAVETheoreticalCycles, "Compute the Shave Kernel theoretical cycles\n\n \n a Shave Kernel\n \n\n unsigned int theoretical execution cycles\n\nC++: VPUNN::VPUNNPerformanceModel::SHAVETheoreticalCycles(const struct VPUNN::SWOperation &) --> unsigned int", pybind11::arg("swl"));
		cl.def("assign", (class VPUNN::VPUNNPerformanceModel & (VPUNN::VPUNNPerformanceModel::*)(const class VPUNN::VPUNNPerformanceModel &)) &VPUNN::VPUNNPerformanceModel::operator=, "C++: VPUNN::VPUNNPerformanceModel::operator=(const class VPUNN::VPUNNPerformanceModel &) --> class VPUNN::VPUNNPerformanceModel &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// VPUNN::get_dma_ports(enum VPUNN::VPUDevice) file: line:680
	M("VPUNN").def("get_dma_ports", (int (*)(enum VPUNN::VPUDevice)) &VPUNN::get_dma_ports, "Get the channels/Ports of DMA.\n Can be used to run one channel per separate tile (like for DDR to CM in case of weights for SOK),\n cannot be used to run multiple channels when transferring to same tile\n \n\n a VPUDevice\n \n\n int\n\nC++: VPUNN::get_dma_ports(enum VPUNN::VPUDevice) --> int", pybind11::arg("device"));

	{ // VPUNN::VPUPowerFactorLUT file: line:32
		pybind11::class_<VPUNN::VPUPowerFactorLUT, std::shared_ptr<VPUNN::VPUPowerFactorLUT>> cl(M("VPUNN"), "VPUPowerFactorLUT", "VPU Power factor LUTs\n \n\n The power factor LUT is lookup table that will be indexed by operation\n and will return another LUT that will be indexed by the number of input channels\n When there is no entry in the second LUT, the value returned will be the interpolation between its smaller and\n greater match in table");
		cl.def( pybind11::init( [](VPUNN::VPUPowerFactorLUT const &o){ return new VPUNN::VPUPowerFactorLUT(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::VPUPowerFactorLUT(); } ) );
		cl.def("getOperationAndPowerVirusAdjustementFactor", (float (VPUNN::VPUPowerFactorLUT::*)(const struct VPUNN::DPUWorkload &) const) &VPUNN::VPUPowerFactorLUT::getOperationAndPowerVirusAdjustementFactor, "Get the value from the LUT+ extra info for a specific workload, represents the relative power factor\n adjustment towards the PowerVirus (INT8). The factor will take in consideration all aspects of the WL ,\n operation, type, etc\n\n \n the workload for which to compute the factor.\n \n\n  the adjustment factor\n\nC++: VPUNN::VPUPowerFactorLUT::getOperationAndPowerVirusAdjustementFactor(const struct VPUNN::DPUWorkload &) const --> float", pybind11::arg("wl"));
		cl.def("getFP_overI8_maxPower_ratio", (float (VPUNN::VPUPowerFactorLUT::*)(enum VPUNN::VPUDevice) const) &VPUNN::VPUPowerFactorLUT::getFP_overI8_maxPower_ratio, "Scale the power factor value according to data type [FPmaxP/I8MaxP]\n \n\n The entries in the power factor LUTs above are based on UINT8 operation\n The power for FLOAT16 is different (depending on VPUDevice), here we approximate the\n difference by scaling by a fixed amount/ratio.\n\n \n that is used\n \n\n the float to int ratio for power considering the device\n\nC++: VPUNN::VPUPowerFactorLUT::getFP_overI8_maxPower_ratio(enum VPUNN::VPUDevice) const --> float", pybind11::arg("device"));
		cl.def("get_PowerVirus_exceed_factor", (float (VPUNN::VPUPowerFactorLUT::*)(enum VPUNN::VPUDevice) const) &VPUNN::VPUPowerFactorLUT::get_PowerVirus_exceed_factor, "C++: VPUNN::VPUPowerFactorLUT::get_PowerVirus_exceed_factor(enum VPUNN::VPUDevice) const --> float", pybind11::arg("device"));
	}
	{ // VPUNN::DPU_OperationSanitizer file: line:25
		pybind11::class_<VPUNN::DPU_OperationSanitizer, std::shared_ptr<VPUNN::DPU_OperationSanitizer>> cl(M("VPUNN"), "DPU_OperationSanitizer", "Sanitizes a Workload  based on device rules");
		cl.def( pybind11::init( [](VPUNN::DPU_OperationSanitizer const &o){ return new VPUNN::DPU_OperationSanitizer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DPU_OperationSanitizer(); } ) );
		cl.def("check_and_sanitize", (void (VPUNN::DPU_OperationSanitizer::*)(struct VPUNN::DPUWorkload &, struct VPUNN::SanityReport &) const) &VPUNN::DPU_OperationSanitizer::check_and_sanitize, "Checks if the workload is usable, makes sanitization changes , and lets the user know the sanitized\n workload and its conclusion\n\n Sanitization means that some parameters of the workload are automatically adjusted, but still the new WL is\n relevant (or equivalent regarding cost) to the original one. Sanitization performed:\n - input and output tensors data type are restricted to one type per category. UINT8 for ints on 8 bit, FLOAT16\n for all16 bit floats. \n\n IDeviceValidValues around valid_datatypes usage\n  - ...\n\n Usability checks will check general characteristics:\n - if the device is supported\n - if workload fits in CMX memory\n - if the operation is supported\n - ...\n\n \n [in, out] the workload to analyze and sanitize\n \n\n [out] status of check. if not OK (not NO_ERROR) the wl cannot be used\n\nC++: VPUNN::DPU_OperationSanitizer::check_and_sanitize(struct VPUNN::DPUWorkload &, struct VPUNN::SanityReport &) const --> void", pybind11::arg("wl"), pybind11::arg("result"));
		cl.def("check_data_consistency", (void (VPUNN::DPU_OperationSanitizer::*)(struct VPUNN::DPUWorkload &, struct VPUNN::SanityReport &) const) &VPUNN::DPU_OperationSanitizer::check_data_consistency, "C++: VPUNN::DPU_OperationSanitizer::check_data_consistency(struct VPUNN::DPUWorkload &, struct VPUNN::SanityReport &) const --> void", pybind11::arg("wl"), pybind11::arg("result"));
	}
	{ // VPUNN::ShaveOpExecutor file: line:21
		pybind11::class_<VPUNN::ShaveOpExecutor, VPUNN::ShaveOpExecutor*, PyCallBack_VPUNN_ShaveOpExecutor> cl(M("VPUNN"), "ShaveOpExecutor", "Interface for a shave model from the execution(obtaining the runtime) perspective.\n Instances are properly configured Shave models, for a particular model.\n Cannot be deleted by the user.");
		cl.def(pybind11::init<PyCallBack_VPUNN_ShaveOpExecutor const &>());
		cl.def("dpuCycles", (unsigned int (VPUNN::ShaveOpExecutor::*)(const class VPUNN::SHAVEWorkload &) const) &VPUNN::ShaveOpExecutor::dpuCycles, "Return the number of cycles of the sw operation\n\n \n the workload descriptor. Some fields may be ignored . Eg even if the Device is not matching the one that\n the model was built for, it will run the estimation as if it had a good device in workload\n\n \n cycles in dpu frequency\n\nC++: VPUNN::ShaveOpExecutor::dpuCycles(const class VPUNN::SHAVEWorkload &) const --> unsigned int", pybind11::arg("w"));
		cl.def("getName", (std::string (VPUNN::ShaveOpExecutor::*)() const) &VPUNN::ShaveOpExecutor::getName, "the name of the modeled SHAVE function\n\nC++: VPUNN::ShaveOpExecutor::getName() const --> std::string");
		cl.def("assign", (class VPUNN::ShaveOpExecutor & (VPUNN::ShaveOpExecutor::*)(const class VPUNN::ShaveOpExecutor &)) &VPUNN::ShaveOpExecutor::operator=, "C++: VPUNN::ShaveOpExecutor::operator=(const class VPUNN::ShaveOpExecutor &) --> class VPUNN::ShaveOpExecutor &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::DeviceShaveContainer file: line:28
		pybind11::class_<VPUNN::DeviceShaveContainer, std::shared_ptr<VPUNN::DeviceShaveContainer>> cl(M("VPUNN"), "DeviceShaveContainer", "list of shaves attached to a device\n must have access to the destructor of the ShaveOpExecutor.\n     owns the executer concrete instances (creation and destruction is in its responsibility)");
		cl.def( pybind11::init<enum VPUNN::VPUDevice>(), pybind11::arg("device") );

		cl.def("getDevice", (enum VPUNN::VPUDevice (VPUNN::DeviceShaveContainer::*)() const) &VPUNN::DeviceShaveContainer::getDevice, "C++: VPUNN::DeviceShaveContainer::getDevice() const --> enum VPUNN::VPUDevice");
		cl.def("existsShave", (bool (VPUNN::DeviceShaveContainer::*)(const std::string) const) &VPUNN::DeviceShaveContainer::existsShave, "C++: VPUNN::DeviceShaveContainer::existsShave(const std::string) const --> bool", pybind11::arg("sw"));
		cl.def("getShaveList", (class std::vector<std::string, class std::allocator<std::string > > (VPUNN::DeviceShaveContainer::*)() const) &VPUNN::DeviceShaveContainer::getShaveList, "C++: VPUNN::DeviceShaveContainer::getShaveList() const --> class std::vector<std::string, class std::allocator<std::string > >");
		cl.def("getShaveExecutor", (const class VPUNN::ShaveOpExecutor & (VPUNN::DeviceShaveContainer::*)(const std::string &) const) &VPUNN::DeviceShaveContainer::getShaveExecutor, "C++: VPUNN::DeviceShaveContainer::getShaveExecutor(const std::string &) const --> const class VPUNN::ShaveOpExecutor &", pybind11::return_value_policy::automatic, pybind11::arg("sw"));
	}
	{ // VPUNN::ShaveInstanceHolder_VPU27 file: line:111
		pybind11::class_<VPUNN::ShaveInstanceHolder_VPU27, std::shared_ptr<VPUNN::ShaveInstanceHolder_VPU27>, VPUNN::DeviceShaveContainer> cl(M("VPUNN"), "ShaveInstanceHolder_VPU27", "");
		cl.def( pybind11::init( [](){ return new VPUNN::ShaveInstanceHolder_VPU27(); } ) );
		cl.def("getDevice", [](VPUNN::ShaveInstanceHolder_VPU27 const &o) -> VPUNN::VPUDevice { return o.getDevice(); }, "");
		cl.def("getContainer", (const class VPUNN::DeviceShaveContainer & (VPUNN::ShaveInstanceHolder_VPU27::*)() const) &VPUNN::ShaveInstanceHolder_VPU27::getContainer, "C++: VPUNN::ShaveInstanceHolder_VPU27::getContainer() const --> const class VPUNN::DeviceShaveContainer &", pybind11::return_value_policy::automatic);
		cl.def("populate", (void (VPUNN::ShaveInstanceHolder_VPU27::*)()) &VPUNN::ShaveInstanceHolder_VPU27::populate, "C++: VPUNN::ShaveInstanceHolder_VPU27::populate() --> void");
	}
	{ // VPUNN::ShaveInstanceHolder_VPU27CLassic file: line:125
		pybind11::class_<VPUNN::ShaveInstanceHolder_VPU27CLassic, std::shared_ptr<VPUNN::ShaveInstanceHolder_VPU27CLassic>, VPUNN::DeviceShaveContainer> cl(M("VPUNN"), "ShaveInstanceHolder_VPU27CLassic", "");
		cl.def( pybind11::init( [](){ return new VPUNN::ShaveInstanceHolder_VPU27CLassic(); } ) );
		cl.def("getDevice", [](VPUNN::ShaveInstanceHolder_VPU27CLassic const &o) -> VPUNN::VPUDevice { return o.getDevice(); }, "");
		cl.def("getContainer", (const class VPUNN::DeviceShaveContainer & (VPUNN::ShaveInstanceHolder_VPU27CLassic::*)() const) &VPUNN::ShaveInstanceHolder_VPU27CLassic::getContainer, "C++: VPUNN::ShaveInstanceHolder_VPU27CLassic::getContainer() const --> const class VPUNN::DeviceShaveContainer &", pybind11::return_value_policy::automatic);
		cl.def("populate", (void (VPUNN::ShaveInstanceHolder_VPU27CLassic::*)()) &VPUNN::ShaveInstanceHolder_VPU27CLassic::populate, "C++: VPUNN::ShaveInstanceHolder_VPU27CLassic::populate() --> void");
	}
}


// File: VPUNN_37.cpp
#include <array> // std::array
#include <ios> // std::_Ios_Seekdir
#include <iterator> // __gnu_cxx::__normal_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <tuple> // std::tuple
#include <utility> // std::pair
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::ShaveSelector file: line:26
struct PyCallBack_VPUNN_ShaveSelector : public VPUNN::ShaveSelector {
	using VPUNN::ShaveSelector::ShaveSelector;

	const class VPUNN::ShaveOpExecutor & getShaveFuntion(const std::string & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ShaveSelector *>(this), "getShaveFuntion");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class VPUNN::ShaveOpExecutor &>::value) {
				static pybind11::detail::override_caster_t<const class VPUNN::ShaveOpExecutor &> caster;
				return pybind11::detail::cast_ref<const class VPUNN::ShaveOpExecutor &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class VPUNN::ShaveOpExecutor &>(std::move(o));
		}
		return ShaveSelector::getShaveFuntion(a0);
	}
	using _binder_ret_0 = class std::vector<std::string, class std::allocator<std::string > >;
	_binder_ret_0 getShaveList() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ShaveSelector *>(this), "getShaveList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return ShaveSelector::getShaveList();
	}
};

// VPUNN::ShavePrioritySelector file: line:54
struct PyCallBack_VPUNN_ShavePrioritySelector : public VPUNN::ShavePrioritySelector {
	using VPUNN::ShavePrioritySelector::ShavePrioritySelector;

	const class VPUNN::ShaveOpExecutor & getShaveFuntion(const std::string & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ShavePrioritySelector *>(this), "getShaveFuntion");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<const class VPUNN::ShaveOpExecutor &>::value) {
				static pybind11::detail::override_caster_t<const class VPUNN::ShaveOpExecutor &> caster;
				return pybind11::detail::cast_ref<const class VPUNN::ShaveOpExecutor &>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<const class VPUNN::ShaveOpExecutor &>(std::move(o));
		}
		return ShavePrioritySelector::getShaveFuntion(a0);
	}
	using _binder_ret_0 = class std::vector<std::string, class std::allocator<std::string > >;
	_binder_ret_0 getShaveList() const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ShavePrioritySelector *>(this), "getShaveList");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return ShavePrioritySelector::getShaveList();
	}
};

void bind_VPUNN_37(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::ShaveSelector file: line:26
		pybind11::class_<VPUNN::ShaveSelector, std::shared_ptr<VPUNN::ShaveSelector>, PyCallBack_VPUNN_ShaveSelector> cl(M("VPUNN"), "ShaveSelector", "selects  in a trivial manner, from one container");
		cl.def( pybind11::init<enum VPUNN::VPUDevice, const class VPUNN::DeviceShaveContainer &>(), pybind11::arg("device"), pybind11::arg("shave_container") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_ShaveSelector const &o){ return new PyCallBack_VPUNN_ShaveSelector(o); } ) );
		cl.def( pybind11::init( [](VPUNN::ShaveSelector const &o){ return new VPUNN::ShaveSelector(o); } ) );
		cl.def("isDeviceSupported", (bool (VPUNN::ShaveSelector::*)(enum VPUNN::VPUDevice) const) &VPUNN::ShaveSelector::isDeviceSupported, "C++: VPUNN::ShaveSelector::isDeviceSupported(enum VPUNN::VPUDevice) const --> bool", pybind11::arg("device"));
		cl.def("getShaveFuntion", (const class VPUNN::ShaveOpExecutor & (VPUNN::ShaveSelector::*)(const std::string &) const) &VPUNN::ShaveSelector::getShaveFuntion, "C++: VPUNN::ShaveSelector::getShaveFuntion(const std::string &) const --> const class VPUNN::ShaveOpExecutor &", pybind11::return_value_policy::automatic, pybind11::arg("name"));
		cl.def("getShaveList", (class std::vector<std::string, class std::allocator<std::string > > (VPUNN::ShaveSelector::*)() const) &VPUNN::ShaveSelector::getShaveList, "C++: VPUNN::ShaveSelector::getShaveList() const --> class std::vector<std::string, class std::allocator<std::string > >");
	}
	{ // VPUNN::ShavePrioritySelector file: line:54
		pybind11::class_<VPUNN::ShavePrioritySelector, std::shared_ptr<VPUNN::ShavePrioritySelector>, PyCallBack_VPUNN_ShavePrioritySelector, VPUNN::ShaveSelector> cl(M("VPUNN"), "ShavePrioritySelector", "selects from 2 containers , 1st with priority");
		cl.def( pybind11::init<enum VPUNN::VPUDevice, const class VPUNN::DeviceShaveContainer &, const class VPUNN::DeviceShaveContainer &>(), pybind11::arg("device"), pybind11::arg("shave_container_first"), pybind11::arg("shave_container_second") );

		cl.def( pybind11::init( [](PyCallBack_VPUNN_ShavePrioritySelector const &o){ return new PyCallBack_VPUNN_ShavePrioritySelector(o); } ) );
		cl.def( pybind11::init( [](VPUNN::ShavePrioritySelector const &o){ return new VPUNN::ShavePrioritySelector(o); } ) );
		cl.def("getShaveFuntion", (const class VPUNN::ShaveOpExecutor & (VPUNN::ShavePrioritySelector::*)(const std::string &) const) &VPUNN::ShavePrioritySelector::getShaveFuntion, "C++: VPUNN::ShavePrioritySelector::getShaveFuntion(const std::string &) const --> const class VPUNN::ShaveOpExecutor &", pybind11::return_value_policy::automatic, pybind11::arg("name"));
		cl.def("getShaveList", (class std::vector<std::string, class std::allocator<std::string > > (VPUNN::ShavePrioritySelector::*)() const) &VPUNN::ShavePrioritySelector::getShaveList, "C++: VPUNN::ShavePrioritySelector::getShaveList() const --> class std::vector<std::string, class std::allocator<std::string > >");
	}
	{ // VPUNN::ShaveConfiguration file: line:79
		pybind11::class_<VPUNN::ShaveConfiguration, std::shared_ptr<VPUNN::ShaveConfiguration>> cl(M("VPUNN"), "ShaveConfiguration", "the shave configuration. Holds instances");
		cl.def( pybind11::init( [](){ return new VPUNN::ShaveConfiguration(); } ) );
		cl.def("computeCycles", (unsigned int (VPUNN::ShaveConfiguration::*)(const class VPUNN::SHAVEWorkload &, std::string &) const) &VPUNN::ShaveConfiguration::computeCycles, "C++: VPUNN::ShaveConfiguration::computeCycles(const class VPUNN::SHAVEWorkload &, std::string &) const --> unsigned int", pybind11::arg("swl"), pybind11::arg("infoOut"));
		cl.def("getShaveSupportedOperations", (class std::vector<std::string, class std::allocator<std::string > > (VPUNN::ShaveConfiguration::*)(enum VPUNN::VPUDevice) const) &VPUNN::ShaveConfiguration::getShaveSupportedOperations, "C++: VPUNN::ShaveConfiguration::getShaveSupportedOperations(enum VPUNN::VPUDevice) const --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg("device"));
	}
	{ // VPUNN::DPUInfoPack file: line:46
		pybind11::class_<VPUNN::DPUInfoPack, std::shared_ptr<VPUNN::DPUInfoPack>> cl(M("VPUNN"), "DPUInfoPack", "L1API info for a DPUWorkload.\n intention is to obtain all info at once, in a more efficient way.\n Zero values means either error or value could not be obtained\n See the original interface method for each field to understand its meaning");
		cl.def( pybind11::init( [](VPUNN::DPUInfoPack const &o){ return new VPUNN::DPUInfoPack(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DPUInfoPack(); } ) );
		cl.def_readwrite("DPUCycles", &VPUNN::DPUInfoPack::DPUCycles);
		cl.def_readwrite("errInfo", &VPUNN::DPUInfoPack::errInfo);
		cl.def_readwrite("energy", &VPUNN::DPUInfoPack::energy);
		cl.def_readwrite("power_activity_factor", &VPUNN::DPUInfoPack::power_activity_factor);
		cl.def_readwrite("power_mac_utilization", &VPUNN::DPUInfoPack::power_mac_utilization);
		cl.def_readwrite("power_ideal_cycles", &VPUNN::DPUInfoPack::power_ideal_cycles);
		cl.def_readwrite("sparse_mac_operations", &VPUNN::DPUInfoPack::sparse_mac_operations);
		cl.def_readwrite("efficiency_activity_factor", &VPUNN::DPUInfoPack::efficiency_activity_factor);
		cl.def_readwrite("efficiency_mac_utilization", &VPUNN::DPUInfoPack::efficiency_mac_utilization);
		cl.def_readwrite("efficiency_ideal_cycles", &VPUNN::DPUInfoPack::efficiency_ideal_cycles);
		cl.def_readwrite("dense_mac_operations", &VPUNN::DPUInfoPack::dense_mac_operations);
		cl.def_readwrite("hw_theoretical_cycles", &VPUNN::DPUInfoPack::hw_theoretical_cycles);

		cl.def("__str__", [](VPUNN::DPUInfoPack const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
	{ // VPUNN::VPUCostModel file: line:94
		pybind11::class_<VPUNN::VPUCostModel, std::shared_ptr<VPUNN::VPUCostModel>, VPUNN::VPUNNPerformanceModel> cl(M("VPUNN"), "VPUCostModel", "The VPUCostModel class\n\n Has behind a loaded CostModel neural network that infers cycle times for DPUWOrkloads\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUCostModel(); } ), "doc" );
		cl.def( pybind11::init( [](const std::string & a0){ return new VPUNN::VPUCostModel(a0); } ), "doc" , pybind11::arg("filename"));
		cl.def( pybind11::init( [](const std::string & a0, bool const & a1){ return new VPUNN::VPUCostModel(a0, a1); } ), "doc" , pybind11::arg("filename"), pybind11::arg("profile"));
		cl.def( pybind11::init( [](const std::string & a0, bool const & a1, const unsigned int & a2){ return new VPUNN::VPUCostModel(a0, a1, a2); } ), "doc" , pybind11::arg("filename"), pybind11::arg("profile"), pybind11::arg("cache_size"));
		cl.def( pybind11::init<const std::string &, bool, const unsigned int, const unsigned int>(), pybind11::arg("filename"), pybind11::arg("profile"), pybind11::arg("cache_size"), pybind11::arg("batch_size") );

		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2){ return new VPUNN::VPUCostModel(a0, a1, a2); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, bool const & a3){ return new VPUNN::VPUCostModel(a0, a1, a2, a3); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, bool const & a3, const unsigned int & a4){ return new VPUNN::VPUCostModel(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"), pybind11::arg("cache_size"));
		cl.def( pybind11::init<const char *, unsigned long, bool, bool, const unsigned int, const unsigned int>(), pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"), pybind11::arg("cache_size"), pybind11::arg("batch_size") );

		cl.def("get_NN_Valid_interval", (struct std::pair<float, float> (VPUNN::VPUCostModel::*)() const) &VPUNN::VPUCostModel::get_NN_Valid_interval, "provides the value interval where the NN raw outputs are considered valid and will be used to further\n compute information\n\n \n a pair containing (minimum_valid_value maximum_valid_value)\n\nC++: VPUNN::VPUCostModel::get_NN_Valid_interval() const --> struct std::pair<float, float>");
		cl.def("run_NN", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::run_NN, "Compute the NN Output of a specific DPUWorkload\n takes in consideration the cache\n no sanitation is done\n no check if network exists\n\n \n a DPUWorkload\n \n\n float the NN raw output, not filtered\n\nC++: VPUNN::VPUCostModel::run_NN(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("workload"));
		cl.def("run_NN", (const class std::vector<float, class std::allocator<float> > & (VPUNN::VPUCostModel::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &)) &VPUNN::VPUCostModel::run_NN, "Compute the NN Output of multiple DPUWorkloads\n NOT taking in consideration the cache\n no sanitation is done\n no check if network exists\n\n \n a std::vector of DPUWorkloads\n \n\n a reference to the results vector. Do not store, will be invalid after next call to this method.\n\nC++: VPUNN::VPUCostModel::run_NN(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &) --> const class std::vector<float, class std::allocator<float> > &", pybind11::return_value_policy::automatic, pybind11::arg("workloads"));
		cl.def("nn_initialized", (bool (VPUNN::VPUCostModel::*)() const) &VPUNN::VPUCostModel::nn_initialized, "Check if the internal VPUNN is initialized\n\n \n true the VPUNN neural network is initialized\n \n\n false the VPUNN neural network is not initialized\n\nC++: VPUNN::VPUCostModel::nn_initialized() const --> bool");
		cl.def("DPU", (unsigned int (VPUNN::VPUCostModel::*)(struct VPUNN::DPUWorkload)) &VPUNN::VPUCostModel::DPU, "Return the number of cycles needed to compute a workload\n\n Important: If no NN is available it will return Theoretical cycles for the workload. Check if NN is loaded with\n nn_initialized()\n\n A sanity check will be performed on the workload and in case it is not suitable the method will return an error\n code without running the inference on the NN. \n\n DPU_OperationSanitizer::check_and_sanitize() explanations.\n Some checks examples:\n - if the device is supported\n - if workload fits in CMX memory\n - if the operation is supported\n\n List of Error codes is available in CyclesInterfaceType doc.\n\n A sanity check will be performed also on the NN output, in case the NN  raw value is not reliable it will not be\n returned but an error code will be given, e.g. ERROR_INVALID_OUTPUT_RANGE\n\n To see the limits of valid NN values interval , use \n get_NN_Valid_interval().  Zero is a value that will NOT\n be filtered out.\n\n Behind the DPU computation is a trained Neural Network that might give unexpected results in case is asked about\n a workload that is odd/(not well formed) or was not trained in that area or workloads.\n The workload passed as parameter for inference should be a valid one, a one that makes sense, we are checking\n some sanity, but ,for now, not a strict/extensive sanity check is performed. A workload with unrealistic\n combinations of  parameters (eg DW_CONV with 7 input/output channels ) will not be detected.\n\n In case the wl configuration is unrealistic the network will give undefined(aberrant) results (it was not trained\n on invalid data). The NN raw output is filtered for  generic valid interval (no negatives, no huge , e.g. 4bilion\n cycles) but the user can also be aware of this behavior and use its own narrower ranges\n\n e.g.  Depending on the wl a cycle values of 10 might be unrealistic, also a value of 100milion cycles (@1Ghz is\n ~100ms),  The user should be aware that not all aberrant/unrealistic NN outputs are handled inside.\n\n \n a DPUWorkload to be evaluated.\n \n\n unsigned int DPUWorkload execution cycles or an error code.\n\n \n out_of_range : cache problems, cannot pre-process data , generate the NN descriptor due to data unknown\n \n\n runtime_error: cannot generate the NN descriptor, e.g expected sizes do not match\n\n     \n\nC++: VPUNN::VPUCostModel::DPU(struct VPUNN::DPUWorkload) --> unsigned int", pybind11::arg("wl"));
		cl.def("DPUMsg", (class std::tuple<unsigned int, std::string > (VPUNN::VPUCostModel::*)(struct VPUNN::DPUWorkload)) &VPUNN::VPUCostModel::DPUMsg, "same like  \n DPU(DPUWorkload wl) , the extra param is to have as output the textual errors/findings\n discovered when handling the workload\n \n\n the workload to infer on\n \n\n will collect error info regarding wl checking.\n\nC++: VPUNN::VPUCostModel::DPUMsg(struct VPUNN::DPUWorkload) --> class std::tuple<unsigned int, std::string >", pybind11::arg("wl"));
		cl.def("DPU", (unsigned int (VPUNN::VPUCostModel::*)(struct VPUNN::DPUWorkload, std::string &)) &VPUNN::VPUCostModel::DPU, "same like  \n DPU(DPUWorkload wl) , the extra param is to have as output the textual errors/findings\n discovered when handling the workload\n \n\n the workload to infer on\n \n\n [out] will collect error info regarding wl checking.\n\nC++: VPUNN::VPUCostModel::DPU(struct VPUNN::DPUWorkload, std::string &) --> unsigned int", pybind11::arg("wl"), pybind11::arg("info"));
		cl.def("DPU", (class std::vector<unsigned int, class std::allocator<unsigned int> > (VPUNN::VPUCostModel::*)(class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> >)) &VPUNN::VPUCostModel::DPU, "Return the number of cycles needed to compute multiple workloads\n\n \n a std::vector of DPUWorkload\n \n\n std::vector<CyclesInterfaceType> the DPUWorklaods execution cycles, \n DPU for single wl for more\n explanations\n\nC++: VPUNN::VPUCostModel::DPU(class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> >) --> class std::vector<unsigned int, class std::allocator<unsigned int> >", pybind11::arg("workloads"));
		cl.def("hw_utilization", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::hw_utilization, "Compute DPUWorkload hw utilization based on ideal cycles considering also HW/sparsity.\n This is in the context of the operation's datatype. (do not compare float with int values)\n Represents the percentage [0,1+] of ideal resources(MAC based) used by this workload.\n 1 = 100% of MACs are used\n The value is calculated using the Estimated Runtime (cycles) by VPUNN.\n If VPUNN is missing the TheoreticalCycles are used\n\n \n a DPUWorkload\n \n\n  DPUWorkload hardware utilization (zero signals problems)\n\nC++: VPUNN::VPUCostModel::hw_utilization(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("power_mac_hw_utilization", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::power_mac_hw_utilization, "Compute DPUWorkload hw utilization based on ideal cycles considering also HW/sparsity.\n This is in the context of the operation's datatype. (do not compare float with int values)\n Represents the percentage [0,1+] of ideal resources(MAC based) used by this workload.\n 1 = 100% of MACs are used\n The value is calculated using the Estimated Runtime (cycles) by VPUNN.\n If VPUNN is missing the TheoreticalCycles are used\n\n \n a DPUWorkload\n \n\n  DPUWorkload hardware utilization (zero signals problems)\n\nC++: VPUNN::VPUCostModel::power_mac_hw_utilization(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("efficiency_mac_hw_utilization", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::efficiency_mac_hw_utilization, "utilization without sparsity, can be larger than one \n\nC++: VPUNN::VPUCostModel::efficiency_mac_hw_utilization(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("DMA", [](VPUNN::VPUCostModel const &o, enum VPUNN::VPUDevice const & a0, const class VPUNN::VPUTensor & a1, const class VPUNN::VPUTensor & a2) -> unsigned int { return o.DMA(a0, a1, a2); }, "", pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output"));
		cl.def("DMA", [](VPUNN::VPUCostModel const &o, enum VPUNN::VPUDevice const & a0, const class VPUNN::VPUTensor & a1, const class VPUNN::VPUTensor & a2, enum VPUNN::MemoryLocation const & a3) -> unsigned int { return o.DMA(a0, a1, a2, a3); }, "", pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("input_location"));
		cl.def("DMA", [](VPUNN::VPUCostModel const &o, enum VPUNN::VPUDevice const & a0, const class VPUNN::VPUTensor & a1, const class VPUNN::VPUTensor & a2, enum VPUNN::MemoryLocation const & a3, enum VPUNN::MemoryLocation const & a4) -> unsigned int { return o.DMA(a0, a1, a2, a3, a4); }, "", pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("input_location"), pybind11::arg("output_location"));
		cl.def("DMA", (unsigned int (VPUNN::VPUCostModel::*)(enum VPUNN::VPUDevice, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &, enum VPUNN::MemoryLocation, enum VPUNN::MemoryLocation, unsigned int) const) &VPUNN::VPUCostModel::DMA, "Return the number of cycles needed to compute a DMA transfer\n\n \n DMA VPUDevice\n \n\n DMA input Tensor\n \n\n DMA output Tensor\n \n\n where is the source memory\n \n\n where is the destination memory\n \n\n how many CMX tiles the DMA broadcast\n \n\n unsigned int the number of cycles of the DMA transfer\n\nC++: VPUNN::VPUCostModel::DMA(enum VPUNN::VPUDevice, const class VPUNN::VPUTensor &, const class VPUNN::VPUTensor &, enum VPUNN::MemoryLocation, enum VPUNN::MemoryLocation, unsigned int) const --> unsigned int", pybind11::arg("device"), pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("input_location"), pybind11::arg("output_location"), pybind11::arg("output_write_tiles"));
		cl.def("DMA", (unsigned int (VPUNN::VPUCostModel::*)(const struct VPUNN::DMAWorkload &) const) &VPUNN::VPUCostModel::DMA, "Return the number of cycles needed to compute a DMA transfer\n\n \n a DMAWorkload\n \n\n unsigned int the number of cycles of the DMA transfer\n\nC++: VPUNN::VPUCostModel::DMA(const struct VPUNN::DMAWorkload &) const --> unsigned int", pybind11::arg("wl"));
		cl.def("SHAVE", (unsigned int (VPUNN::VPUCostModel::*)(const struct VPUNN::SWOperation &)) &VPUNN::VPUCostModel::SHAVE, "Return the number of cycles needed to compute a Shave kernel\n\n \n a Shave Kernel\n \n\n unsigned int the number of cycles of the Shave kernel\n\nC++: VPUNN::VPUCostModel::SHAVE(const struct VPUNN::SWOperation &) --> unsigned int", pybind11::arg("swl"));
		cl.def("SHAVE_2", (unsigned int (VPUNN::VPUCostModel::*)(const class VPUNN::SHAVEWorkload &, std::string &)) &VPUNN::VPUCostModel::SHAVE_2, "Return the number of cycles needed to compute a Shave kernel\n\n \n a Shave Kernel\n \n\n the number of cycles of the Shave kernel, in DPU cycles. Are the DPUcycles the ones of the device (yes)?\n\nC++: VPUNN::VPUCostModel::SHAVE_2(const class VPUNN::SHAVEWorkload &, std::string &) --> unsigned int", pybind11::arg("swl"), pybind11::arg("infoOut"));
		cl.def("getShaveSupportedOperations", (class std::vector<std::string, class std::allocator<std::string > > (VPUNN::VPUCostModel::*)(enum VPUNN::VPUDevice) const) &VPUNN::VPUCostModel::getShaveSupportedOperations, "C++: VPUNN::VPUCostModel::getShaveSupportedOperations(enum VPUNN::VPUDevice) const --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg("device"));
		cl.def("DPUActivityFactor", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::DPUActivityFactor, "proxy for DPU_RelativeActivityFactor_hw\n\nC++: VPUNN::VPUCostModel::DPUActivityFactor(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("DPU_PowerActivityFactor", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::DPU_PowerActivityFactor, "Compute the activity factor of a DPUWorkload.\n \n\n Activity factor is an estimation of the dynamic power of the DPUWorkload\n relative to the worst case (reference dynamic power) DPUWorkload.\n Interval [0, 1 or more], where 1 means the power virus activity factor\n reference dynamic power is considered for INT8 operations\n It can be more than 1 in case the PowerViruschosen for reference is not the fact the highest (like if reference\n is power virus INT8,  the float operations can have the AF >1).\n\n \n a DPUWorkload\n \n\n float the DPUWorkload activity factor relative to reference PowerVirus  (now is INT8)\n\nC++: VPUNN::VPUCostModel::DPU_PowerActivityFactor(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("DPU_EfficiencyActivityFactor", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::DPU_EfficiencyActivityFactor, "C++: VPUNN::VPUCostModel::DPU_EfficiencyActivityFactor(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("DPUEnergy", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::DPUEnergy, "Compute the energy of a DPUWorkload.\n \n\n This is a relative energy metric with a time base in DPU clock cyles. Energy of\n 1000 would mean energy of worst case power for 1000 DPU clock cycles at reference dynamic power (power virus\n for INT08 operations). measured in PowerVirusJoules = PowerVirus*cycle\n \n\n a DPUWorkload\n \n\n float the DPUWorkload energy, measured  PowerVirus*cycle\n\nC++: VPUNN::VPUCostModel::DPUEnergy(const struct VPUNN::DPUWorkload &) --> float", pybind11::arg("wl"));
		cl.def("SHAVEEnergy", (float (VPUNN::VPUCostModel::*)(const struct VPUNN::SWOperation &)) &VPUNN::VPUCostModel::SHAVEEnergy, "Compute the energy of a SHAVE SWOperation.\n \n\n Energy here is a relative metric, but the activity factor of the SWOperation multiplied by\n          its cost (number of clock cycles). We assume a constant activity factor of 0.5 for all and a max\n          power of 5% of the DPU max power.\n\n \n a SWOperation\n \n\n float the SWOperation energy , in units relative to DPU PowerVirus16\n\nC++: VPUNN::VPUCostModel::SHAVEEnergy(const struct VPUNN::SWOperation &) --> float", pybind11::arg("swl"));
		cl.def("DPUInfo", (struct VPUNN::DPUInfoPack (VPUNN::VPUCostModel::*)(const struct VPUNN::DPUWorkload &)) &VPUNN::VPUCostModel::DPUInfo, "same like  \n DPU(DPUWorkload wl) but return a Pack of information regarding the workload\n The purpose of this Method is to replace several separate calls to individual informations about the same\n workload.\n For example , estimated  cycle-times, errors, energy, activity factor, all can be obtained in one call.\n This method has the potential to be more efficient that the collection of individual ones.\n \n\n the workload to infer on\n \n\n a Structure with all info that L1 APi can provide about this Workload\n\nC++: VPUNN::VPUCostModel::DPUInfo(const struct VPUNN::DPUWorkload &) --> struct VPUNN::DPUInfoPack", pybind11::arg("workload"));
	}
}


// File: VPUNN_38.cpp
#include <array> // std::array
#include <ios> // std::_Ios_Seekdir
#include <iterator> // __gnu_cxx::__normal_iterator
#include <iterator> // std::reverse_iterator
#include <locale> // std::locale
#include <memory> // std::allocator
#include <memory> // std::default_delete
#include <memory> // std::unique_ptr
#include <ostream> // std::basic_ostream
#include <sstream> // __str__
#include <streambuf> // std::basic_streambuf
#include <string> // std::basic_string
#include <string> // std::char_traits
#include <tuple> // std::tuple
#include <utility> // std::pair
#include <vector> // std::_Bit_const_iterator
#include <vector> // std::_Bit_iterator
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

// VPUNN::DPUTiler file: line:64
struct PyCallBack_VPUNN_DPUTiler : public VPUNN::DPUTiler {
	using VPUNN::DPUTiler::DPUTiler;

	using _binder_ret_0 = struct std::pair<unsigned int, class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > >;
	_binder_ret_0 intraTileSplit(const struct VPUNN::DPULayer & a0, const struct VPUNN::SplitOptions & a1) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DPUTiler *>(this), "intraTileSplit");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"DPUTiler::intraTileSplit\"");
	}
	struct VPUNN::PnPEstimates getLayerPerformance(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, const unsigned int a1, const bool a2) override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DPUTiler *>(this), "getLayerPerformance");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<struct VPUNN::PnPEstimates>::value) {
				static pybind11::detail::override_caster_t<struct VPUNN::PnPEstimates> caster;
				return pybind11::detail::cast_ref<struct VPUNN::PnPEstimates>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<struct VPUNN::PnPEstimates>(std::move(o));
		}
		pybind11::pybind11_fail("Tried to call pure virtual function \"DPUTiler::getLayerPerformance\"");
	}
};

// VPUNN::CONVOLUTION_Constraints_Layer file: line:20
struct PyCallBack_VPUNN_CONVOLUTION_Constraints_Layer : public VPUNN::CONVOLUTION_Constraints_Layer {
	using VPUNN::CONVOLUTION_Constraints_Layer::CONVOLUTION_Constraints_Layer;

	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return CONVOLUTION_Constraints_Layer::check_sparsity_rules(a0, a1, a2);
	}
	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CONVOLUTION_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return CONVOLUTION_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CONVOLUTION_Constraints::generate_sparsity(a0, a1, a2);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CONVOLUTION_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CONVOLUTION_Constraints_Layer *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
};

// VPUNN::DW_CONVOLUTION_Constraints_Layer file: line:38
struct PyCallBack_VPUNN_DW_CONVOLUTION_Constraints_Layer : public VPUNN::DW_CONVOLUTION_Constraints_Layer {
	using VPUNN::DW_CONVOLUTION_Constraints_Layer::DW_CONVOLUTION_Constraints_Layer;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DW_CONVOLUTION_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DW_CONVOLUTION_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return DW_CONVOLUTION_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::generate_sparsity(a0, a1, a2);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::DW_CONVOLUTION_Constraints_Layer *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
};

// VPUNN::CM_CONVOLUTION_Constraints_Layer file: line:42
struct PyCallBack_VPUNN_CM_CONVOLUTION_Constraints_Layer : public VPUNN::CM_CONVOLUTION_Constraints_Layer {
	using VPUNN::CM_CONVOLUTION_Constraints_Layer::CM_CONVOLUTION_Constraints_Layer;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::input_0_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return CM_CONVOLUTION_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::generate_sparsity(a0, a1, a2);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return GenericConvolution_Constraints::limit_sparsity(a0, a1);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_volume(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::CM_CONVOLUTION_Constraints_Layer *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
};

// VPUNN::ELTWISE_Constraints_Layer file: line:46
struct PyCallBack_VPUNN_ELTWISE_Constraints_Layer : public VPUNN::ELTWISE_Constraints_Layer {
	using VPUNN::ELTWISE_Constraints_Layer::ELTWISE_Constraints_Layer;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ELTWISE_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return ELTWISE_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ELTWISE_Constraints::generate_sparsity(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return ELTWISE_Constraints::input_1_volume(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return ELTWISE_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ELTWISE_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return ELTWISE_Constraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return ELTWISE_Constraints::filter_output_write_tile_Options(a0);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return ELTWISE_Constraints::get_weight_table_size(a0);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::ELTWISE_Constraints_Layer *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOperationDynamicConstraints::limit_sparsity(a0, a1);
	}
};

// VPUNN::MAXPOOL_Constraints_Layer file: line:50
struct PyCallBack_VPUNN_MAXPOOL_Constraints_Layer : public VPUNN::MAXPOOL_Constraints_Layer {
	using VPUNN::MAXPOOL_Constraints_Layer::MAXPOOL_Constraints_Layer;

	void generate_operation_dependent_tensors(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "generate_operation_dependent_tensors");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MAXPOOL_Constraints::generate_operation_dependent_tensors(a0, a1, a2);
	}
	bool check_input_output_tensor_corelation(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "check_input_output_tensor_corelation");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return MAXPOOL_Constraints::check_input_output_tensor_corelation(a0, a1, a2);
	}
	void generate_sparsity(class VPUNN::Sampler & a0, const class VPUNN::IDeviceValidValues & a1, struct VPUNN::DPUOperation & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "generate_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MAXPOOL_Constraints::generate_sparsity(a0, a1, a2);
	}
	long long input_1_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "input_1_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return MAXPOOL_Constraints::input_1_volume(a0);
	}
	void deduce_input_1(const struct VPUNN::TensorInfo & a0, const struct VPUNN::TensorInfo & a1, const class VPUNN::IDeviceValidValues & a2, const struct VPUNN::KernelInfo & a3, struct VPUNN::TensorInfo & a4) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "deduce_input_1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MAXPOOL_Constraints::deduce_input_1(a0, a1, a2, a3, a4);
	}
	long long get_weight_table_size(const long long a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "get_weight_table_size");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::get_weight_table_size(a0);
	}
	bool check_sparsity_rules(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1, std::string & a2) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "check_sparsity_rules");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return Base_Constraints::check_sparsity_rules(a0, a1, a2);
	}
	long long input_1_aligned_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "input_1_aligned_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_aligned_size_bytes(a0, a1);
	}
	long long input_1_contiguous_size_bytes(const class VPUNN::IDeviceValidValues & a0, const struct VPUNN::DPUOperation & a1) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "input_1_contiguous_size_bytes");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return Base_Constraints::input_1_contiguous_size_bytes(a0, a1);
	}
	long long input_0_volume(const struct VPUNN::TensorInfo & a0) const throw() override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "input_0_volume");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<long long>::value) {
				static pybind11::detail::override_caster_t<long long> caster;
				return pybind11::detail::cast_ref<long long>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<long long>(std::move(o));
		}
		return IOperationDynamicConstraints::input_0_volume(a0);
	}
	using _binder_ret_0 = class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> >;
	_binder_ret_0 filter_ISI_Strategy_Options(const class std::vector<enum VPUNN::ISIStrategy, class std::allocator<enum VPUNN::ISIStrategy> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "filter_ISI_Strategy_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_ISI_Strategy_Options(a0);
	}
	using _binder_ret_1 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_1 filter_output_write_tile_Options(const class std::vector<int, class std::allocator<int> > & a0) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "filter_output_write_tile_Options");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::override_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return IOperationDynamicConstraints::filter_output_write_tile_Options(a0);
	}
	bool normalize_kernel_dimension(const enum VPUNN::ISIStrategy & a0, struct VPUNN::KernelInfo & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "normalize_kernel_dimension");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::override_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return IOperationDynamicConstraints::normalize_kernel_dimension(a0, a1);
	}
	void limit_sparsity(const class VPUNN::IDeviceValidValues & a0, struct VPUNN::DPUOperation & a1) const override {
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const VPUNN::MAXPOOL_Constraints_Layer *>(this), "limit_sparsity");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::override_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return IOperationDynamicConstraints::limit_sparsity(a0, a1);
	}
};

void bind_VPUNN_38(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// VPUNN::VPUOptimizationTarget file: line:27
	pybind11::enum_<VPUNN::VPUOptimizationTarget>(M("VPUNN"), "VPUOptimizationTarget", "Available VPU workload generation optimization targets\n\n ")
		.value("LATENCY", VPUNN::VPUOptimizationTarget::LATENCY)
		.value("POWER", VPUNN::VPUOptimizationTarget::POWER)
		.value("EFFICIENCY", VPUNN::VPUOptimizationTarget::EFFICIENCY);

;

	// VPUNN::VPUSplitStrategy file: line:32
	pybind11::enum_<VPUNN::VPUSplitStrategy>(M("VPUNN"), "VPUSplitStrategy", "Available VPU splitting strategies\n\n ")
		.value("HW_TILING", VPUNN::VPUSplitStrategy::HW_TILING)
		.value("Z_TILING", VPUNN::VPUSplitStrategy::Z_TILING)
		.value("H_TILING", VPUNN::VPUSplitStrategy::H_TILING)
		.value("W_TILING", VPUNN::VPUSplitStrategy::W_TILING);

;

	{ // VPUNN::SplitOptions file: line:38
		pybind11::class_<VPUNN::SplitOptions, std::shared_ptr<VPUNN::SplitOptions>> cl(M("VPUNN"), "SplitOptions", "VPU splitting optimization configuration options\n Used to guide the splitting of a Layer to 1 or more DPUs");
		cl.def( pybind11::init( [](){ return new VPUNN::SplitOptions(); } ) );
		cl.def( pybind11::init( [](VPUNN::SplitOptions const &o){ return new VPUNN::SplitOptions(o); } ) );
		cl.def_readwrite("maxWorkloads", &VPUNN::SplitOptions::maxWorkloads);
		cl.def_readwrite("maxLatencyUs", &VPUNN::SplitOptions::maxLatencyUs);
		cl.def_readwrite("nDPU", &VPUNN::SplitOptions::nDPU);
		cl.def_readwrite("runtimeOverhead", &VPUNN::SplitOptions::runtimeOverhead);
		cl.def_readwrite("target", &VPUNN::SplitOptions::target);
		cl.def_readwrite("availableStrategies", &VPUNN::SplitOptions::availableStrategies);
	}
	{ // VPUNN::PnPEstimates file: line:56
		pybind11::class_<VPUNN::PnPEstimates, std::shared_ptr<VPUNN::PnPEstimates>> cl(M("VPUNN"), "PnPEstimates", "VPU Power and Performance estimates (cycles and power)");
		cl.def( pybind11::init( [](){ return new VPUNN::PnPEstimates(); } ) );
		cl.def_readwrite("cycles", &VPUNN::PnPEstimates::cycles);
		cl.def_readwrite("power", &VPUNN::PnPEstimates::power);
	}
	{ // VPUNN::DPUTiler file: line:64
		pybind11::class_<VPUNN::DPUTiler, std::shared_ptr<VPUNN::DPUTiler>, PyCallBack_VPUNN_DPUTiler> cl(M("VPUNN"), "DPUTiler", "DPU Tiler interface");
		cl.def( pybind11::init( [](){ return new PyCallBack_VPUNN_DPUTiler(); } ) );
		cl.def("intraTileSplit", (struct std::pair<unsigned int, class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > > (VPUNN::DPUTiler::*)(const struct VPUNN::DPULayer &, const struct VPUNN::SplitOptions &)) &VPUNN::DPUTiler::intraTileSplit, "Generate the optimal intra-tile split for a specific DPULayer\n \n\n This function takes the model, the layer to optimize and the nDPU as a parameter and returns the optimal\n workloads split. The information about the device, sparsity are encoded in the DPULayer type. The mode is part of\n the DPUWorkload structure\n\n \n DPULayer to optimize\n \n\n workload splits algorithm configuration options\n \n\n DPUWorkloadsCost the optimal workloads split\n\nC++: VPUNN::DPUTiler::intraTileSplit(const struct VPUNN::DPULayer &, const struct VPUNN::SplitOptions &) --> struct std::pair<unsigned int, class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > >", pybind11::arg("layer"), pybind11::arg("options"));
		cl.def("getLayerPerformance", [](VPUNN::DPUTiler &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0) -> VPUNN::PnPEstimates { return o.getLayerPerformance(a0); }, "", pybind11::arg("workloads"));
		cl.def("getLayerPerformance", [](VPUNN::DPUTiler &o, const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > & a0, const unsigned int & a1) -> VPUNN::PnPEstimates { return o.getLayerPerformance(a0, a1); }, "", pybind11::arg("workloads"), pybind11::arg("runtimeOverhead"));
		cl.def("getLayerPerformance", (struct VPUNN::PnPEstimates (VPUNN::DPUTiler::*)(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, const unsigned int, const bool)) &VPUNN::DPUTiler::getLayerPerformance, "Get the cycles and power estimate for a list of workloads.\n \n\n This function does not optimize any workloads\n but simply calculate the cost of that configuration. It is possible to pass an optional runtime overhead in\n cycles\n\n \n a vector of DPUWorkload\n \n\n execution runtime overhead in cycles (per workload)\n \n\n if true power will be zero, otherwise is calculated\n \n\n PnPEstimates power and performance estimate for the workloads\n\n \n exceptions from inner dependencies. like DPU invocation\n\nC++: VPUNN::DPUTiler::getLayerPerformance(const class std::vector<struct VPUNN::DPUWorkload, class std::allocator<struct VPUNN::DPUWorkload> > &, const unsigned int, const bool) --> struct VPUNN::PnPEstimates", pybind11::arg("workloads"), pybind11::arg("runtimeOverhead"), pybind11::arg("skip_power"));
		cl.def("assign", (class VPUNN::DPUTiler & (VPUNN::DPUTiler::*)(const class VPUNN::DPUTiler &)) &VPUNN::DPUTiler::operator=, "C++: VPUNN::DPUTiler::operator=(const class VPUNN::DPUTiler &) --> class VPUNN::DPUTiler &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// VPUNN::getDPUTiler(class VPUNN::VPUCostModel &) file: line:106
	M("VPUNN").def("getDPUTiler", (class std::unique_ptr<class VPUNN::DPUTiler, struct std::default_delete<class VPUNN::DPUTiler> > (*)(class VPUNN::VPUCostModel &)) &VPUNN::getDPUTiler, "Factory function that generates a DPUTiler instance\n\n \n a reference to a VPUCostModel object\n \n\n std::unique_ptr<DPUTiler>\n\nC++: VPUNN::getDPUTiler(class VPUNN::VPUCostModel &) --> class std::unique_ptr<class VPUNN::DPUTiler, struct std::default_delete<class VPUNN::DPUTiler> >", pybind11::arg("_model"));

	{ // VPUNN::CONVOLUTION_Constraints_Layer file: line:20
		pybind11::class_<VPUNN::CONVOLUTION_Constraints_Layer, std::shared_ptr<VPUNN::CONVOLUTION_Constraints_Layer>, PyCallBack_VPUNN_CONVOLUTION_Constraints_Layer, VPUNN::CONVOLUTION_Constraints> cl(M("VPUNN"), "CONVOLUTION_Constraints_Layer", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_CONVOLUTION_Constraints_Layer const &o){ return new PyCallBack_VPUNN_CONVOLUTION_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](VPUNN::CONVOLUTION_Constraints_Layer const &o){ return new VPUNN::CONVOLUTION_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::CONVOLUTION_Constraints_Layer(); }, [](){ return new PyCallBack_VPUNN_CONVOLUTION_Constraints_Layer(); } ) );
		cl.def("assign", (class VPUNN::CONVOLUTION_Constraints_Layer & (VPUNN::CONVOLUTION_Constraints_Layer::*)(const class VPUNN::CONVOLUTION_Constraints_Layer &)) &VPUNN::CONVOLUTION_Constraints_Layer::operator=, "C++: VPUNN::CONVOLUTION_Constraints_Layer::operator=(const class VPUNN::CONVOLUTION_Constraints_Layer &) --> class VPUNN::CONVOLUTION_Constraints_Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::DW_CONVOLUTION_Constraints_Layer file: line:38
		pybind11::class_<VPUNN::DW_CONVOLUTION_Constraints_Layer, std::shared_ptr<VPUNN::DW_CONVOLUTION_Constraints_Layer>, PyCallBack_VPUNN_DW_CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints> cl(M("VPUNN"), "DW_CONVOLUTION_Constraints_Layer", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_DW_CONVOLUTION_Constraints_Layer const &o){ return new PyCallBack_VPUNN_DW_CONVOLUTION_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](VPUNN::DW_CONVOLUTION_Constraints_Layer const &o){ return new VPUNN::DW_CONVOLUTION_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::DW_CONVOLUTION_Constraints_Layer(); }, [](){ return new PyCallBack_VPUNN_DW_CONVOLUTION_Constraints_Layer(); } ) );
		cl.def("assign", (class VPUNN::DW_CONVOLUTION_Constraints_Layer & (VPUNN::DW_CONVOLUTION_Constraints_Layer::*)(const class VPUNN::DW_CONVOLUTION_Constraints_Layer &)) &VPUNN::DW_CONVOLUTION_Constraints_Layer::operator=, "C++: VPUNN::DW_CONVOLUTION_Constraints_Layer::operator=(const class VPUNN::DW_CONVOLUTION_Constraints_Layer &) --> class VPUNN::DW_CONVOLUTION_Constraints_Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::CM_CONVOLUTION_Constraints_Layer file: line:42
		pybind11::class_<VPUNN::CM_CONVOLUTION_Constraints_Layer, std::shared_ptr<VPUNN::CM_CONVOLUTION_Constraints_Layer>, PyCallBack_VPUNN_CM_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints> cl(M("VPUNN"), "CM_CONVOLUTION_Constraints_Layer", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_CM_CONVOLUTION_Constraints_Layer const &o){ return new PyCallBack_VPUNN_CM_CONVOLUTION_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](VPUNN::CM_CONVOLUTION_Constraints_Layer const &o){ return new VPUNN::CM_CONVOLUTION_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::CM_CONVOLUTION_Constraints_Layer(); }, [](){ return new PyCallBack_VPUNN_CM_CONVOLUTION_Constraints_Layer(); } ) );
		cl.def("assign", (class VPUNN::CM_CONVOLUTION_Constraints_Layer & (VPUNN::CM_CONVOLUTION_Constraints_Layer::*)(const class VPUNN::CM_CONVOLUTION_Constraints_Layer &)) &VPUNN::CM_CONVOLUTION_Constraints_Layer::operator=, "C++: VPUNN::CM_CONVOLUTION_Constraints_Layer::operator=(const class VPUNN::CM_CONVOLUTION_Constraints_Layer &) --> class VPUNN::CM_CONVOLUTION_Constraints_Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::ELTWISE_Constraints_Layer file: line:46
		pybind11::class_<VPUNN::ELTWISE_Constraints_Layer, std::shared_ptr<VPUNN::ELTWISE_Constraints_Layer>, PyCallBack_VPUNN_ELTWISE_Constraints_Layer, VPUNN::ELTWISE_Constraints> cl(M("VPUNN"), "ELTWISE_Constraints_Layer", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_ELTWISE_Constraints_Layer const &o){ return new PyCallBack_VPUNN_ELTWISE_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](VPUNN::ELTWISE_Constraints_Layer const &o){ return new VPUNN::ELTWISE_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::ELTWISE_Constraints_Layer(); }, [](){ return new PyCallBack_VPUNN_ELTWISE_Constraints_Layer(); } ) );
		cl.def("assign", (class VPUNN::ELTWISE_Constraints_Layer & (VPUNN::ELTWISE_Constraints_Layer::*)(const class VPUNN::ELTWISE_Constraints_Layer &)) &VPUNN::ELTWISE_Constraints_Layer::operator=, "C++: VPUNN::ELTWISE_Constraints_Layer::operator=(const class VPUNN::ELTWISE_Constraints_Layer &) --> class VPUNN::ELTWISE_Constraints_Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::MAXPOOL_Constraints_Layer file: line:50
		pybind11::class_<VPUNN::MAXPOOL_Constraints_Layer, std::shared_ptr<VPUNN::MAXPOOL_Constraints_Layer>, PyCallBack_VPUNN_MAXPOOL_Constraints_Layer, VPUNN::MAXPOOL_Constraints> cl(M("VPUNN"), "MAXPOOL_Constraints_Layer", "");
		cl.def( pybind11::init( [](PyCallBack_VPUNN_MAXPOOL_Constraints_Layer const &o){ return new PyCallBack_VPUNN_MAXPOOL_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](VPUNN::MAXPOOL_Constraints_Layer const &o){ return new VPUNN::MAXPOOL_Constraints_Layer(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::MAXPOOL_Constraints_Layer(); }, [](){ return new PyCallBack_VPUNN_MAXPOOL_Constraints_Layer(); } ) );
		cl.def("assign", (class VPUNN::MAXPOOL_Constraints_Layer & (VPUNN::MAXPOOL_Constraints_Layer::*)(const class VPUNN::MAXPOOL_Constraints_Layer &)) &VPUNN::MAXPOOL_Constraints_Layer::operator=, "C++: VPUNN::MAXPOOL_Constraints_Layer::operator=(const class VPUNN::MAXPOOL_Constraints_Layer &) --> class VPUNN::MAXPOOL_Constraints_Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::VPU_LayerValidator file: line:38
		pybind11::class_<VPUNN::VPU_LayerValidator, std::shared_ptr<VPUNN::VPU_LayerValidator>, VPUNN::Behavior_Device_Mapping<VPUNN::Behaviours<VPUNN::CONVOLUTION_Constraints_Layer, VPUNN::DW_CONVOLUTION_Constraints_Layer, VPUNN::CM_CONVOLUTION_Constraints_Layer, VPUNN::ELTWISE_Constraints_Layer, VPUNN::MAXPOOL_Constraints_Layer>,VPUNN::VPU2_0_LayerValidValues, VPUNN::VPU2_7_LayerValidValues, VPUNN::VPU4_0_LayerValidValues>> cl(M("VPUNN"), "VPU_LayerValidator", "services for Layer validation");
		cl.def( pybind11::init( [](VPUNN::VPU_LayerValidator const &o){ return new VPUNN::VPU_LayerValidator(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::VPU_LayerValidator(); } ) );
		cl.def("check_layer_consistency", (void (VPUNN::VPU_LayerValidator::*)(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &, const class VPUNN::IOperationDynamicConstraints &, struct VPUNN::SanityReport &) const) &VPUNN::VPU_LayerValidator::check_layer_consistency, "C++: VPUNN::VPU_LayerValidator::check_layer_consistency(const struct VPUNN::DPUOperation &, const class VPUNN::IDeviceValidValues &, const class VPUNN::IOperationDynamicConstraints &, struct VPUNN::SanityReport &) const --> void", pybind11::arg("w"), pybind11::arg("config"), pybind11::arg("operation_behaviour"), pybind11::arg("result"));
	}
	{ // VPUNN::LayersValidation file: line:29
		pybind11::class_<VPUNN::LayersValidation, std::shared_ptr<VPUNN::LayersValidation>> cl(M("VPUNN"), "LayersValidation", "Layer validation mechanisms for split and un-split layers");
		cl.def( pybind11::init( [](VPUNN::LayersValidation const &o){ return new VPUNN::LayersValidation(o); } ) );
		cl.def( pybind11::init( [](){ return new VPUNN::LayersValidation(); } ) );
		cl.def("getDeviceConfiguratorForTiles", (const class VPUNN::IDeviceValidValues & (VPUNN::LayersValidation::*)(enum VPUNN::VPUDevice) const) &VPUNN::LayersValidation::getDeviceConfiguratorForTiles, "C++: VPUNN::LayersValidation::getDeviceConfiguratorForTiles(enum VPUNN::VPUDevice) const --> const class VPUNN::IDeviceValidValues &", pybind11::return_value_policy::automatic, pybind11::arg("device"));
		cl.def("check_completeLayer_consistency", (void (VPUNN::LayersValidation::*)(const struct VPUNN::DPULayer &, struct VPUNN::SanityReport &, enum VPUNN::ISIStrategy, unsigned int) const) &VPUNN::LayersValidation::check_completeLayer_consistency, "checks the layer validity against the rules of an unsplit Layer\n\nC++: VPUNN::LayersValidation::check_completeLayer_consistency(const struct VPUNN::DPULayer &, struct VPUNN::SanityReport &, enum VPUNN::ISIStrategy, unsigned int) const --> void", pybind11::arg("layer"), pybind11::arg("result"), pybind11::arg("strategy"), pybind11::arg("nTiles"));
		cl.def("check_splitLayer_consistency", (void (VPUNN::LayersValidation::*)(const struct VPUNN::DPULayer &, struct VPUNN::SanityReport &) const) &VPUNN::LayersValidation::check_splitLayer_consistency, "C++: VPUNN::LayersValidation::check_splitLayer_consistency(const struct VPUNN::DPULayer &, struct VPUNN::SanityReport &) const --> void", pybind11::arg("layer"), pybind11::arg("result"));
		cl.def("sanitize_preconditions", (void (VPUNN::LayersValidation::*)(struct VPUNN::DPULayer &) const) &VPUNN::LayersValidation::sanitize_preconditions, "C++: VPUNN::LayersValidation::sanitize_preconditions(struct VPUNN::DPULayer &) const --> void", pybind11::arg("layer"));
	}
	{ // VPUNN::VPULayerStrategy file: line:29
		pybind11::class_<VPUNN::VPULayerStrategy, std::shared_ptr<VPUNN::VPULayerStrategy>> cl(M("VPUNN"), "VPULayerStrategy", "A VPU layer strategy");
		cl.def( pybind11::init( [](){ return new VPUNN::VPULayerStrategy(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPULayerStrategy const &o){ return new VPUNN::VPULayerStrategy(o); } ) );
		cl.def_readwrite("nDPUs", &VPUNN::VPULayerStrategy::nDPUs);
		cl.def_readwrite("nSHVs", &VPUNN::VPULayerStrategy::nSHVs);
		cl.def_readwrite("nTiles", &VPUNN::VPULayerStrategy::nTiles);
		cl.def_readwrite("tiling_strategy", &VPUNN::VPULayerStrategy::tiling_strategy);
		cl.def_readwrite("input_fetching", &VPUNN::VPULayerStrategy::input_fetching);
		cl.def_readwrite("output_spilling", &VPUNN::VPULayerStrategy::output_spilling);
		cl.def_readwrite("prefetching", &VPUNN::VPULayerStrategy::prefetching);
		cl.def("assign", (struct VPUNN::VPULayerStrategy & (VPUNN::VPULayerStrategy::*)(const struct VPUNN::VPULayerStrategy &)) &VPUNN::VPULayerStrategy::operator=, "C++: VPUNN::VPULayerStrategy::operator=(const struct VPUNN::VPULayerStrategy &) --> struct VPUNN::VPULayerStrategy &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		cl.def("__str__", [](VPUNN::VPULayerStrategy const &o) -> std::string { std::ostringstream s; VPUNN::operator<<(s, o); return s.str(); } );
	}
}


// File: VPUNN_39.cpp
#include <array> // std::array
#include <iterator> // __gnu_cxx::__normal_iterator
#include <memory> // std::allocator
#include <memory> // std::shared_ptr
#include <sstream> // __str__
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_39(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::VPULayerCostModel file: line:60
		pybind11::class_<VPUNN::VPULayerCostModel, std::shared_ptr<VPUNN::VPULayerCostModel>, VPUNN::VPUCostModel> cl(M("VPUNN"), "VPULayerCostModel", "The VPUNN layer cost model (also called VPUNN Level2 API)");
		cl.def( pybind11::init( [](){ return new VPUNN::VPULayerCostModel(); } ) );
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2){ return new VPUNN::VPULayerCostModel(a0, a1, a2); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, bool const & a3){ return new VPUNN::VPULayerCostModel(a0, a1, a2, a3); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, bool const & a3, const unsigned int & a4){ return new VPUNN::VPULayerCostModel(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"), pybind11::arg("cache_size"));
		cl.def( pybind11::init<const char *, unsigned long, bool, bool, const unsigned int, const unsigned int>(), pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"), pybind11::arg("cache_size"), pybind11::arg("batch_size") );

		cl.def( pybind11::init( [](){ return new VPUNN::VPULayerCostModel(); } ), "doc" );
		cl.def( pybind11::init( [](const std::string & a0){ return new VPUNN::VPULayerCostModel(a0); } ), "doc" , pybind11::arg("filename"));
		cl.def( pybind11::init( [](const std::string & a0, bool const & a1){ return new VPUNN::VPULayerCostModel(a0, a1); } ), "doc" , pybind11::arg("filename"), pybind11::arg("profile"));
		cl.def( pybind11::init( [](const std::string & a0, bool const & a1, const unsigned int & a2){ return new VPUNN::VPULayerCostModel(a0, a1, a2); } ), "doc" , pybind11::arg("filename"), pybind11::arg("profile"), pybind11::arg("cache_size"));
		cl.def( pybind11::init<const std::string &, bool, const unsigned int, const unsigned int>(), pybind11::arg("filename"), pybind11::arg("profile"), pybind11::arg("cache_size"), pybind11::arg("batch_size") );

		cl.def("set_maxWorkloadsPerIntraTileSplit", (void (VPUNN::VPULayerCostModel::*)(unsigned int)) &VPUNN::VPULayerCostModel::set_maxWorkloadsPerIntraTileSplit, "limits the split of a tile (intra-tile split) to this number of individual workloads\n\nC++: VPUNN::VPULayerCostModel::set_maxWorkloadsPerIntraTileSplit(unsigned int) --> void", pybind11::arg("new_value"));
		cl.def("get_maxWorkloadsPerIntraTileSplit", (unsigned int (VPUNN::VPULayerCostModel::*)() const) &VPUNN::VPULayerCostModel::get_maxWorkloadsPerIntraTileSplit, "C++: VPUNN::VPULayerCostModel::get_maxWorkloadsPerIntraTileSplit() const --> unsigned int");
		cl.def("Layer", (unsigned int (VPUNN::VPULayerCostModel::*)(struct VPUNN::DPULayer &, struct VPUNN::VPULayerStrategy)) &VPUNN::VPULayerCostModel::Layer, "Compute the optimal cost of a DPULayer given a strategy and context\n\n \n the DPULayer\n \n\n the layer strategy, shaves do not matter\n \n\n  measured best cycles or error code . \n Cycles for error codes\n\nC++: VPUNN::VPULayerCostModel::Layer(struct VPUNN::DPULayer &, struct VPUNN::VPULayerStrategy) --> unsigned int", pybind11::arg("layer"), pybind11::arg("strategy"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, enum VPUNN::VPUTilingStrategy const & a1) -> unsigned int { return o.Layer(a0, a1); }, "", pybind11::arg("layer"), pybind11::arg("strategy"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, enum VPUNN::VPUTilingStrategy const & a1, unsigned int const & a2) -> unsigned int { return o.Layer(a0, a1, a2); }, "", pybind11::arg("layer"), pybind11::arg("strategy"), pybind11::arg("nDPU"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, enum VPUNN::VPUTilingStrategy const & a1, unsigned int const & a2, unsigned int const & a3) -> unsigned int { return o.Layer(a0, a1, a2, a3); }, "", pybind11::arg("layer"), pybind11::arg("strategy"), pybind11::arg("nDPU"), pybind11::arg("nTiles"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, enum VPUNN::VPUTilingStrategy const & a1, unsigned int const & a2, unsigned int const & a3, bool const & a4) -> unsigned int { return o.Layer(a0, a1, a2, a3, a4); }, "", pybind11::arg("layer"), pybind11::arg("strategy"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, enum VPUNN::VPUTilingStrategy const & a1, unsigned int const & a2, unsigned int const & a3, bool const & a4, bool const & a5) -> unsigned int { return o.Layer(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("layer"), pybind11::arg("strategy"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"));
		cl.def("Layer", (unsigned int (VPUNN::VPULayerCostModel::*)(struct VPUNN::DPULayer &, enum VPUNN::VPUTilingStrategy, unsigned int, unsigned int, bool, bool, bool)) &VPUNN::VPULayerCostModel::Layer, "Compute the optimal cost of a DPULayer using a specific strategy and context\n\n It splits on tiles(between tiles, using the strategy), then, for each tile , makes the intra-tile split on\n workloads and choses the best one\n\n \n the DPULayer\n \n\n the inter-tile tiling strategy to use\n \n\n the number of DPU (for each tile)\n \n\n the number of CMX tiles\n \n\n enable/disable input in DDR (require extra DMA to fetch data in CMX)\n \n\n enable/disable output in DDR (require extra DMA to spill data in CMX)\n \n\n If true it considers the weights are prefetched, if false\n will fetch the weights considering also sparsity\n takes in consideration the sparsity(enabled and value)\n \n\n measured best cycles or error code . \n Cycles for error codes\n\nC++: VPUNN::VPULayerCostModel::Layer(struct VPUNN::DPULayer &, enum VPUNN::VPUTilingStrategy, unsigned int, unsigned int, bool, bool, bool) --> unsigned int", pybind11::arg("layer"), pybind11::arg("strategy"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"), pybind11::arg("prefetching"));
		cl.def("Layer", (unsigned int (VPUNN::VPULayerCostModel::*)(struct VPUNN::DPULayer &, enum VPUNN::VPUTilingStrategy, unsigned int, unsigned int, bool, bool, bool, class std::vector<struct VPUNN::OneTileLayerInfo, class std::allocator<struct VPUNN::OneTileLayerInfo> > &)) &VPUNN::VPULayerCostModel::Layer, "Compute the optimal cost of a DPULayer using a specific strategy and execution mode\n\n It splits on tiles(between tiles, using the strategy), then, for each tile , makes the intra-tile split on\n workloads and choses the best one\n\n \n the DPULayer\n \n\n the inter-tile tiling strategy to use\n \n\n the number of DPU (for each tile)\n \n\n the number of CMX tiles\n \n\n enable/disable input in DDR (require extra DMA to fetch data in CMX). Data fetch time is\n computed considering the full layer input tensor not he split ones\n \n\n enable/disable output in DDR (require extra DMA to spill data in CMX). Data fetch time is\n computed considering the full layer output tensor not he split ones\n \n\n  If true it considers the weights are prefetched, if false\n will fetch the weights considering also sparsity. Data fetch time is computed considering the split layers\n weights tensors, that are pipelined on all available DMA channels.\n \n\n [out] gives as output the information on how was split this layer and what is the best\n split on workloads\n \n\n measured best cycles or error code . \n Cycles for error codes\n\nC++: VPUNN::VPULayerCostModel::Layer(struct VPUNN::DPULayer &, enum VPUNN::VPUTilingStrategy, unsigned int, unsigned int, bool, bool, bool, class std::vector<struct VPUNN::OneTileLayerInfo, class std::allocator<struct VPUNN::OneTileLayerInfo> > &) --> unsigned int", pybind11::arg("layer"), pybind11::arg("strategy"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"), pybind11::arg("prefetching"), pybind11::arg("detailed_split"));
		cl.def("LayersPreSplit", (unsigned int (VPUNN::VPULayerCostModel::*)(const class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > &, unsigned int, bool, bool, bool, class std::vector<struct VPUNN::OneTileLayerInfo, class std::allocator<struct VPUNN::OneTileLayerInfo> > &)) &VPUNN::VPULayerCostModel::LayersPreSplit, "Compute the optimal cost of a pre split layer. Layer is already split on tiles, only the intratile split\n si performed.\n\n For each tile , makes the intra-tile split on workloads and choses the best one\n\n \n the list of layers split on tiles, their number indicates the tiles. Full info has to be\n specified, as it is for a DPUWorkload\n \n\n the number of DPU (for each tile)\n\n \n enable/disable input in DDR (require extra DMA to fetch data in CMX). Data fetch time is\n computed considering the split layers input tensors, that are summed up.\n \n\n enable/disable output in DDR (require extra DMA to spill data in CMX). Data fetch time is\n computed considering the split layers output tensors, that are summed up.\n \n\n  If true it considers the weights are prefetched, if false\n will fetch the weights considering also sparsity. Data fetch time is computed considering the split layers\n weights tensors, that are pipelined on all available DMA channels.\n\n \n [out] gives as output the information on how was split this layer and what is the best\n split on workloads\n\n \n measured best cycles for the overall vector of layers or error code . \n Cycles for error codes\n\nC++: VPUNN::VPULayerCostModel::LayersPreSplit(const class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > &, unsigned int, bool, bool, bool, class std::vector<struct VPUNN::OneTileLayerInfo, class std::allocator<struct VPUNN::OneTileLayerInfo> > &) --> unsigned int", pybind11::arg("layers_pre_split"), pybind11::arg("nDPU"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"), pybind11::arg("prefetching"), pybind11::arg("detailed_split"));
		cl.def("LayersPreSplit", (unsigned int (VPUNN::VPULayerCostModel::*)(const class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > &, unsigned int, bool, bool, bool)) &VPUNN::VPULayerCostModel::LayersPreSplit, "version without detailed split output parameter.\n\nC++: VPUNN::VPULayerCostModel::LayersPreSplit(const class std::vector<struct VPUNN::DPULayer, class std::allocator<struct VPUNN::DPULayer> > &, unsigned int, bool, bool, bool) --> unsigned int", pybind11::arg("layers_pre_split"), pybind11::arg("nDPU"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"), pybind11::arg("prefetching"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0) -> unsigned int { return o.Layer(a0); }, "", pybind11::arg("layer"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, unsigned int const & a1) -> unsigned int { return o.Layer(a0, a1); }, "", pybind11::arg("layer"), pybind11::arg("nDPU"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, unsigned int const & a1, unsigned int const & a2) -> unsigned int { return o.Layer(a0, a1, a2); }, "", pybind11::arg("layer"), pybind11::arg("nDPU"), pybind11::arg("nTiles"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, unsigned int const & a1, unsigned int const & a2, bool const & a3) -> unsigned int { return o.Layer(a0, a1, a2, a3); }, "", pybind11::arg("layer"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::DPULayer & a0, unsigned int const & a1, unsigned int const & a2, bool const & a3, bool const & a4) -> unsigned int { return o.Layer(a0, a1, a2, a3, a4); }, "", pybind11::arg("layer"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"));
		cl.def("Layer", (unsigned int (VPUNN::VPULayerCostModel::*)(struct VPUNN::DPULayer &, unsigned int, unsigned int, bool, bool, bool)) &VPUNN::VPULayerCostModel::Layer, "Compute the optimal cost of a DPULayer, given a context but no strategy\n\n Analyses all strategies and selects the time o the fastest one\n\n \n the DPULayer\n \n\n the number of DPU\n \n\n the number of CMX tiles\n \n\n enable/disable input in DDR (require extra DMA to fetch data in CMX)\n \n\n enable/disable output in DDR (require extra DMA to spill data in CMX)\n \n\n enable/disable weight prefetching\n \n\n measured best cycles or error code . \n Cycles for error codes\n\nC++: VPUNN::VPULayerCostModel::Layer(struct VPUNN::DPULayer &, unsigned int, unsigned int, bool, bool, bool) --> unsigned int", pybind11::arg("layer"), pybind11::arg("nDPU"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"), pybind11::arg("prefetching"));
		cl.def("Layer", (unsigned long (VPUNN::VPULayerCostModel::*)(struct VPUNN::SWOperation &, const struct VPUNN::VPULayerStrategy &)) &VPUNN::VPULayerCostModel::Layer, "Compute the optimal cost of a SWOperation\n\n \n the SHV kernel\n \n\n the layer strategy\n \n\n unsigned long int\n\nC++: VPUNN::VPULayerCostModel::Layer(struct VPUNN::SWOperation &, const struct VPUNN::VPULayerStrategy &) --> unsigned long", pybind11::arg("layer"), pybind11::arg("strategy"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::SWOperation & a0) -> unsigned long { return o.Layer(a0); }, "", pybind11::arg("layer"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::SWOperation & a0, unsigned int const & a1) -> unsigned long { return o.Layer(a0, a1); }, "", pybind11::arg("layer"), pybind11::arg("nSHV"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::SWOperation & a0, unsigned int const & a1, unsigned int const & a2) -> unsigned long { return o.Layer(a0, a1, a2); }, "", pybind11::arg("layer"), pybind11::arg("nSHV"), pybind11::arg("nTiles"));
		cl.def("Layer", [](VPUNN::VPULayerCostModel &o, struct VPUNN::SWOperation & a0, unsigned int const & a1, unsigned int const & a2, bool const & a3) -> unsigned long { return o.Layer(a0, a1, a2, a3); }, "", pybind11::arg("layer"), pybind11::arg("nSHV"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"));
		cl.def("Layer", (unsigned long (VPUNN::VPULayerCostModel::*)(struct VPUNN::SWOperation &, unsigned int, unsigned int, bool, bool)) &VPUNN::VPULayerCostModel::Layer, "Compute the optimal cost of a SHV kernel\n\n \n the SHV kernel\n \n\n the number of SHV/tile\n \n\n the number of CMX tiles\n \n\n enable/disable input in DDR (require extra DMA to fetch data in CMX)\n \n\n enable/disable output in DDR (require extra DMA to spill data in CMX)\n \n\n unsigned long int\n\nC++: VPUNN::VPULayerCostModel::Layer(struct VPUNN::SWOperation &, unsigned int, unsigned int, bool, bool) --> unsigned long", pybind11::arg("layer"), pybind11::arg("nSHV"), pybind11::arg("nTiles"), pybind11::arg("input_in_ddr"), pybind11::arg("output_in_ddr"));
		cl.def_static("getValidTilingStrategies", (class std::vector<enum VPUNN::VPUTilingStrategy, class std::allocator<enum VPUNN::VPUTilingStrategy> > (*)(const enum VPUNN::VPUDevice &)) &VPUNN::VPULayerCostModel::getValidTilingStrategies, "Get the valid tiling strategy for a device\n\n \n the VPUDevice\n \n\n std::vector<VPUTilingStrategy>\n\nC++: VPUNN::VPULayerCostModel::getValidTilingStrategies(const enum VPUNN::VPUDevice &) --> class std::vector<enum VPUNN::VPUTilingStrategy, class std::allocator<enum VPUNN::VPUTilingStrategy> >", pybind11::arg("device"));
	}
	{ // VPUNN::VPUComputeNode file: line:25
		pybind11::class_<VPUNN::VPUComputeNode, std::shared_ptr<VPUNN::VPUComputeNode>> cl(M("VPUNN"), "VPUComputeNode", "Base class that represent\n\n ");
		cl.def( pybind11::init<const class std::shared_ptr<struct VPUNN::DPULayer>>(), pybind11::arg("dpu_op") );

		cl.def( pybind11::init<const class std::shared_ptr<struct VPUNN::SWOperation>>(), pybind11::arg("shv_op") );

		cl.def( pybind11::init( [](VPUNN::VPUComputeNode const &o){ return new VPUNN::VPUComputeNode(o); } ) );

		pybind11::enum_<VPUNN::VPUComputeNode::OpType>(cl, "OpType", "Node operation type enum\n\n     ")
			.value("DPU_COMPUTE_NODE", VPUNN::VPUComputeNode::OpType::DPU_COMPUTE_NODE)
			.value("SHV_COMPUTE_NODE", VPUNN::VPUComputeNode::OpType::SHV_COMPUTE_NODE);

		cl.def_readwrite("type", &VPUNN::VPUComputeNode::type);
		cl.def("cycles", (unsigned int (VPUNN::VPUComputeNode::*)(class VPUNN::VPULayerCostModel &, struct VPUNN::VPULayerStrategy &) const) &VPUNN::VPUComputeNode::cycles, "Compute the cycles of the VPUComputeNode\n\n \n a reference to a VPULayerCostModel object\n \n\n the strategy to be used.\n \n\n unsigned int execution cycles\n\nC++: VPUNN::VPUComputeNode::cycles(class VPUNN::VPULayerCostModel &, struct VPUNN::VPULayerStrategy &) const --> unsigned int", pybind11::arg("cost_model"), pybind11::arg("strategy"));
		cl.def("__eq__", (bool (VPUNN::VPUComputeNode::*)(const class VPUNN::VPUComputeNode &) const) &VPUNN::VPUComputeNode::operator==, "Operator == : compare this with rhs\n\n \n\n \n\n true\n \n\n false\n\nC++: VPUNN::VPUComputeNode::operator==(const class VPUNN::VPUComputeNode &) const --> bool", pybind11::arg("rhs"));
		cl.def("hash", (unsigned long (VPUNN::VPUComputeNode::*)() const) &VPUNN::VPUComputeNode::hash, "Generate a has for the node\n\n \n size_t\n\nC++: VPUNN::VPUComputeNode::hash() const --> unsigned long");
	}
}


// File: VPUNN_40.cpp
#include <iterator> // __gnu_cxx::__normal_iterator
#include <list> // std::list
#include <memory> // std::allocator
#include <memory> // std::shared_ptr
#include <sstream> // __str__
#include <vector> // std::vector

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_40(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::VPUComputeHash file: line:105
		pybind11::class_<VPUNN::VPUComputeHash, std::shared_ptr<VPUNN::VPUComputeHash>> cl(M("VPUNN"), "VPUComputeHash", "An helper class to generate has for VPUComputeNode objects\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUComputeHash(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUComputeHash const &o){ return new VPUNN::VPUComputeHash(o); } ) );
		cl.def("__call__", (unsigned long (VPUNN::VPUComputeHash::*)(class std::shared_ptr<class VPUNN::VPUComputeNode>) const) &VPUNN::VPUComputeHash::operator(), "a VPUComputeNode object\n \n\n size_t\n\nC++: VPUNN::VPUComputeHash::operator()(class std::shared_ptr<class VPUNN::VPUComputeNode>) const --> unsigned long", pybind11::arg("op"));
	}
	{ // VPUNN::VPUComputeNodeMap file: line:123
		pybind11::class_<VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>, std::shared_ptr<VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>>> cl(M("VPUNN"), "VPUComputeNodeMap_std_vector_std_shared_ptr_VPUNN_VPUComputeNode_std_allocator_std_shared_ptr_VPUNN_VPUComputeNode_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >> const &o){ return new VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>(o); } ) );
		cl.def("__getitem__", (class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > & (VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > > >::operator[], "C++: VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > > >::operator[](const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > &", pybind11::return_value_policy::automatic, pybind11::arg("_key"));
		cl.def("exists", (bool (VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > > >::exists, "C++: VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > > >::exists(const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> bool", pybind11::arg("_key"));
		cl.def("assign", (class VPUNN::VPUComputeNodeMap<class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > > & (VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > >>::*)(const class VPUNN::VPUComputeNodeMap<class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > > &)) &VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > > >::operator=, "C++: VPUNN::VPUComputeNodeMap<std::vector<std::shared_ptr<VPUNN::VPUComputeNode>, std::allocator<std::shared_ptr<VPUNN::VPUComputeNode> > > >::operator=(const class VPUNN::VPUComputeNodeMap<class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > > &) --> class VPUNN::VPUComputeNodeMap<class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > > &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::VPUComputeNodeMap file: line:123
		pybind11::class_<VPUNN::VPUComputeNodeMap<unsigned int>, std::shared_ptr<VPUNN::VPUComputeNodeMap<unsigned int>>> cl(M("VPUNN"), "VPUComputeNodeMap_unsigned_int_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUComputeNodeMap<unsigned int>(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUComputeNodeMap<unsigned int> const &o){ return new VPUNN::VPUComputeNodeMap<unsigned int>(o); } ) );
		cl.def("__getitem__", (unsigned int & (VPUNN::VPUComputeNodeMap<unsigned int>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<unsigned int>::operator[], "C++: VPUNN::VPUComputeNodeMap<unsigned int>::operator[](const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> unsigned int &", pybind11::return_value_policy::automatic, pybind11::arg("_key"));
		cl.def("exists", (bool (VPUNN::VPUComputeNodeMap<unsigned int>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<unsigned int>::exists, "C++: VPUNN::VPUComputeNodeMap<unsigned int>::exists(const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> bool", pybind11::arg("_key"));
		cl.def("assign", (class VPUNN::VPUComputeNodeMap<unsigned int> & (VPUNN::VPUComputeNodeMap<unsigned int>::*)(const class VPUNN::VPUComputeNodeMap<unsigned int> &)) &VPUNN::VPUComputeNodeMap<unsigned int>::operator=, "C++: VPUNN::VPUComputeNodeMap<unsigned int>::operator=(const class VPUNN::VPUComputeNodeMap<unsigned int> &) --> class VPUNN::VPUComputeNodeMap<unsigned int> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::VPUComputeNodeMap file: line:123
		pybind11::class_<VPUNN::VPUComputeNodeMap<bool>, std::shared_ptr<VPUNN::VPUComputeNodeMap<bool>>> cl(M("VPUNN"), "VPUComputeNodeMap_bool_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUComputeNodeMap<bool>(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUComputeNodeMap<bool> const &o){ return new VPUNN::VPUComputeNodeMap<bool>(o); } ) );
		cl.def("__getitem__", (bool & (VPUNN::VPUComputeNodeMap<bool>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<bool>::operator[], "C++: VPUNN::VPUComputeNodeMap<bool>::operator[](const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> bool &", pybind11::return_value_policy::automatic, pybind11::arg("_key"));
		cl.def("exists", (bool (VPUNN::VPUComputeNodeMap<bool>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<bool>::exists, "C++: VPUNN::VPUComputeNodeMap<bool>::exists(const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> bool", pybind11::arg("_key"));
		cl.def("assign", (class VPUNN::VPUComputeNodeMap<bool> & (VPUNN::VPUComputeNodeMap<bool>::*)(const class VPUNN::VPUComputeNodeMap<bool> &)) &VPUNN::VPUComputeNodeMap<bool>::operator=, "C++: VPUNN::VPUComputeNodeMap<bool>::operator=(const class VPUNN::VPUComputeNodeMap<bool> &) --> class VPUNN::VPUComputeNodeMap<bool> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::VPUComputeNodeMap file: line:123
		pybind11::class_<VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>, std::shared_ptr<VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>>> cl(M("VPUNN"), "VPUComputeNodeMap_VPUNN_VPULayerStrategy_t", "");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy> const &o){ return new VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>(o); } ) );
		cl.def("__getitem__", (struct VPUNN::VPULayerStrategy & (VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::operator[], "C++: VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::operator[](const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> struct VPUNN::VPULayerStrategy &", pybind11::return_value_policy::automatic, pybind11::arg("_key"));
		cl.def("exists", (bool (VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::exists, "C++: VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::exists(const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> bool", pybind11::arg("_key"));
		cl.def("assign", (class VPUNN::VPUComputeNodeMap<struct VPUNN::VPULayerStrategy> & (VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::*)(const class VPUNN::VPUComputeNodeMap<struct VPUNN::VPULayerStrategy> &)) &VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::operator=, "C++: VPUNN::VPUComputeNodeMap<VPUNN::VPULayerStrategy>::operator=(const class VPUNN::VPUComputeNodeMap<struct VPUNN::VPULayerStrategy> &) --> class VPUNN::VPUComputeNodeMap<struct VPUNN::VPULayerStrategy> &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::VPUComputationDAG file: line:144
		pybind11::class_<VPUNN::VPUComputationDAG, std::shared_ptr<VPUNN::VPUComputationDAG>> cl(M("VPUNN"), "VPUComputationDAG", "Represent the Computation DAG in a VPU device\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUComputationDAG(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUComputationDAG const &o){ return new VPUNN::VPUComputationDAG(o); } ) );
		cl.def("addNode", (class VPUNN::VPUComputationDAG & (VPUNN::VPUComputationDAG::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode>)) &VPUNN::VPUComputationDAG::addNode, "Add a node to a VPUComputationDAG\n\n \n\n \n\n VPUComputationDAG&\n\nC++: VPUNN::VPUComputationDAG::addNode(const class std::shared_ptr<class VPUNN::VPUComputeNode>) --> class VPUNN::VPUComputationDAG &", pybind11::return_value_policy::automatic, pybind11::arg("layer"));
		cl.def("has", (bool (VPUNN::VPUComputationDAG::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode>) const) &VPUNN::VPUComputationDAG::has, "Return true if a VPUComputeNode is in the VPUComputationDAG\n\n \n layer VPUComputeNode\n \n\n true\n \n\n false\n\nC++: VPUNN::VPUComputationDAG::has(const class std::shared_ptr<class VPUNN::VPUComputeNode>) const --> bool", pybind11::arg("layer"));
		cl.def("addEdge", (class VPUNN::VPUComputationDAG & (VPUNN::VPUComputationDAG::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode>, const class std::shared_ptr<class VPUNN::VPUComputeNode>)) &VPUNN::VPUComputationDAG::addEdge, "add and edge to a VPUComputationDAG\n\n \n the edge predecessor\n \n\n the edge successor\n \n\n VPUComputationDAG&\n\nC++: VPUNN::VPUComputationDAG::addEdge(const class std::shared_ptr<class VPUNN::VPUComputeNode>, const class std::shared_ptr<class VPUNN::VPUComputeNode>) --> class VPUNN::VPUComputationDAG &", pybind11::return_value_policy::automatic, pybind11::arg("source"), pybind11::arg("sink"));
		cl.def("nodes", (unsigned long (VPUNN::VPUComputationDAG::*)() const) &VPUNN::VPUComputationDAG::nodes, "Returns the number of nodes\n\n \n size_t\n\nC++: VPUNN::VPUComputationDAG::nodes() const --> unsigned long");
		cl.def("edges", (unsigned long (VPUNN::VPUComputationDAG::*)()) &VPUNN::VPUComputationDAG::edges, "Returns the number of edges\n\n \n size_t\n\nC++: VPUNN::VPUComputationDAG::edges() --> unsigned long");
		cl.def("sources", (class std::list<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > (VPUNN::VPUComputationDAG::*)()) &VPUNN::VPUComputationDAG::sources, "Returns the list of DAG sources\n\n \n std::list<std::shared_ptr<VPUComputeNode>>\n\nC++: VPUNN::VPUComputationDAG::sources() --> class std::list<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > >");
		cl.def("get_layers", (class std::list<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > (VPUNN::VPUComputationDAG::*)()) &VPUNN::VPUComputationDAG::get_layers, "Return a reference to layers\n\n \n std::list<VPUComputeNode>\n\nC++: VPUNN::VPUComputationDAG::get_layers() --> class std::list<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > >");
		cl.def("get_successors", (class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > (VPUNN::VPUComputationDAG::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode>)) &VPUNN::VPUComputationDAG::get_successors, "Return a list of successors of a layer\n\n \n a pointer to a VPUComputeNode\n \n\n std::vector<std::shared_ptr<VPUComputeNode>>\n\nC++: VPUNN::VPUComputationDAG::get_successors(const class std::shared_ptr<class VPUNN::VPUComputeNode>) --> class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > >", pybind11::arg("layer"));
		cl.def("get_predecessors", (class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > > (VPUNN::VPUComputationDAG::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode>)) &VPUNN::VPUComputationDAG::get_predecessors, "Return a list of predecessors of a layer\n\n \n layer a pointer to a VPUComputeNode\n \n\n std::vector<std::shared_ptr<VPUComputeNode>>\n\nC++: VPUNN::VPUComputationDAG::get_predecessors(const class std::shared_ptr<class VPUNN::VPUComputeNode>) --> class std::vector<class std::shared_ptr<class VPUNN::VPUComputeNode>, class std::allocator<class std::shared_ptr<class VPUNN::VPUComputeNode> > >", pybind11::arg("layer"));
		cl.def("begin", (struct VPUNN::VPUComputationDAG::Iterator (VPUNN::VPUComputationDAG::*)()) &VPUNN::VPUComputationDAG::begin, "DAG Iterator begin\n\n \n Iterator\n\nC++: VPUNN::VPUComputationDAG::begin() --> struct VPUNN::VPUComputationDAG::Iterator");
		cl.def("end", (struct VPUNN::VPUComputationDAG::Iterator (VPUNN::VPUComputationDAG::*)()) &VPUNN::VPUComputationDAG::end, "DAG Iterator end\n\n \n Iterator\n\nC++: VPUNN::VPUComputationDAG::end() --> struct VPUNN::VPUComputationDAG::Iterator");
		cl.def("assign", (class VPUNN::VPUComputationDAG & (VPUNN::VPUComputationDAG::*)(const class VPUNN::VPUComputationDAG &)) &VPUNN::VPUComputationDAG::operator=, "C++: VPUNN::VPUComputationDAG::operator=(const class VPUNN::VPUComputationDAG &) --> class VPUNN::VPUComputationDAG &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		{ // VPUNN::VPUComputationDAG::Iterator file: line:274
			auto & enclosing_class = cl;
			pybind11::class_<VPUNN::VPUComputationDAG::Iterator, std::shared_ptr<VPUNN::VPUComputationDAG::Iterator>> cl(enclosing_class, "Iterator", "A DAG iterator\n\n     ");
			cl.def( pybind11::init( [](class VPUNN::VPUComputationDAG & a0){ return new VPUNN::VPUComputationDAG::Iterator(a0); } ), "doc" , pybind11::arg("dag"));
			cl.def( pybind11::init<class VPUNN::VPUComputationDAG &, bool>(), pybind11::arg("dag"), pybind11::arg("all_visited") );

			cl.def( pybind11::init( [](VPUNN::VPUComputationDAG::Iterator const &o){ return new VPUNN::VPUComputationDAG::Iterator(o); } ) );
			cl.def("dereference", (class std::shared_ptr<class VPUNN::VPUComputeNode> (VPUNN::VPUComputationDAG::Iterator::*)() const) &VPUNN::VPUComputationDAG::Iterator::operator*, "Dereference operator\n\n \n std::shared_ptr<VPUComputeNode>\n\nC++: VPUNN::VPUComputationDAG::Iterator::operator*() const --> class std::shared_ptr<class VPUNN::VPUComputeNode>");
			cl.def("arrow", (class std::shared_ptr<class VPUNN::VPUComputeNode> (VPUNN::VPUComputationDAG::Iterator::*)()) &VPUNN::VPUComputationDAG::Iterator::operator->, "Arrow operator\n\n \n std::shared_ptr<VPUComputeNode>\n\nC++: VPUNN::VPUComputationDAG::Iterator::operator->() --> class std::shared_ptr<class VPUNN::VPUComputeNode>");
			cl.def("pre_increment", (struct VPUNN::VPUComputationDAG::Iterator & (VPUNN::VPUComputationDAG::Iterator::*)()) &VPUNN::VPUComputationDAG::Iterator::operator++, "Prefix increment operator\n\n \n Iterator&\n\nC++: VPUNN::VPUComputationDAG::Iterator::operator++() --> struct VPUNN::VPUComputationDAG::Iterator &", pybind11::return_value_policy::automatic);
			cl.def("post_increment", (struct VPUNN::VPUComputationDAG::Iterator (VPUNN::VPUComputationDAG::Iterator::*)(int)) &VPUNN::VPUComputationDAG::Iterator::operator++, "Postfix increment operator\n\n \n Iterator\n\nC++: VPUNN::VPUComputationDAG::Iterator::operator++(int) --> struct VPUNN::VPUComputationDAG::Iterator", pybind11::arg(""));
		}

	}
}


// File: VPUNN_41.cpp
#include <list> // std::list
#include <memory> // std::allocator
#include <memory> // std::shared_ptr
#include <sstream> // __str__
#include <vector> // std::vector
#include <vpu_network_cost_model.h> // VPUNN::VPUNetworkCostModel
#include <vpu_network_cost_model.h> // VPUNN::VPUNetworkStrategy

#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vpu_cost_model.h>
#include <vpu_network_cost_model.h>
#include <vpu/shave/layers.h>
#include <pybind11/stl.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*)
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>)
#endif

void bind_VPUNN_41(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // VPUNN::VPUNetworkStrategy file:vpu_network_cost_model.h line:22
		pybind11::class_<VPUNN::VPUNetworkStrategy, std::shared_ptr<VPUNN::VPUNetworkStrategy>> cl(M("VPUNN"), "VPUNetworkStrategy", "VPU Network strategy type\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUNetworkStrategy(); } ) );
		cl.def( pybind11::init( [](VPUNN::VPUNetworkStrategy const &o){ return new VPUNN::VPUNetworkStrategy(o); } ) );
		cl.def("__getitem__", (struct VPUNN::VPULayerStrategy & (VPUNN::VPUNetworkStrategy::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUNetworkStrategy::operator[], "C++: VPUNN::VPUNetworkStrategy::operator[](const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> struct VPUNN::VPULayerStrategy &", pybind11::return_value_policy::automatic, pybind11::arg("_key"));
		cl.def("exists", (bool (VPUNN::VPUNetworkStrategy::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &)) &VPUNN::VPUNetworkStrategy::exists, "C++: VPUNN::VPUNetworkStrategy::exists(const class std::shared_ptr<class VPUNN::VPUComputeNode> &) --> bool", pybind11::arg("_key"));
		cl.def("set", (class VPUNN::VPUNetworkStrategy & (VPUNN::VPUNetworkStrategy::*)(const class std::shared_ptr<class VPUNN::VPUComputeNode> &, const struct VPUNN::VPULayerStrategy &)) &VPUNN::VPUNetworkStrategy::set, "C++: VPUNN::VPUNetworkStrategy::set(const class std::shared_ptr<class VPUNN::VPUComputeNode> &, const struct VPUNN::VPULayerStrategy &) --> class VPUNN::VPUNetworkStrategy &", pybind11::return_value_policy::automatic, pybind11::arg("_key"), pybind11::arg("val"));
		cl.def("assign", (class VPUNN::VPUNetworkStrategy & (VPUNN::VPUNetworkStrategy::*)(const class VPUNN::VPUNetworkStrategy &)) &VPUNN::VPUNetworkStrategy::operator=, "C++: VPUNN::VPUNetworkStrategy::operator=(const class VPUNN::VPUNetworkStrategy &) --> class VPUNN::VPUNetworkStrategy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // VPUNN::VPUNetworkCostModel file: line:47
		pybind11::class_<VPUNN::VPUNetworkCostModel, std::shared_ptr<VPUNN::VPUNetworkCostModel>, VPUNN::VPULayerCostModel> cl(M("VPUNN"), "VPUNetworkCostModel", "The VPUNN network cost model (also called VPUNN Level3 API)\n\n ");
		cl.def( pybind11::init( [](){ return new VPUNN::VPUNetworkCostModel(); } ) );
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2){ return new VPUNN::VPUNetworkCostModel(a0, a1, a2); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, bool const & a3){ return new VPUNN::VPUNetworkCostModel(a0, a1, a2, a3); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"));
		cl.def( pybind11::init( [](const char * a0, unsigned long const & a1, bool const & a2, bool const & a3, const unsigned int & a4){ return new VPUNN::VPUNetworkCostModel(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"), pybind11::arg("cache_size"));
		cl.def( pybind11::init<const char *, unsigned long, bool, bool, const unsigned int, const unsigned int>(), pybind11::arg("model_data"), pybind11::arg("model_data_length"), pybind11::arg("copy_model_data"), pybind11::arg("profile"), pybind11::arg("cache_size"), pybind11::arg("batch_size") );

		cl.def( pybind11::init( [](){ return new VPUNN::VPUNetworkCostModel(); } ), "doc" );
		cl.def( pybind11::init( [](const std::string & a0){ return new VPUNN::VPUNetworkCostModel(a0); } ), "doc" , pybind11::arg("filename"));
		cl.def( pybind11::init( [](const std::string & a0, bool const & a1){ return new VPUNN::VPUNetworkCostModel(a0, a1); } ), "doc" , pybind11::arg("filename"), pybind11::arg("profile"));
		cl.def( pybind11::init( [](const std::string & a0, bool const & a1, const unsigned int & a2){ return new VPUNN::VPUNetworkCostModel(a0, a1, a2); } ), "doc" , pybind11::arg("filename"), pybind11::arg("profile"), pybind11::arg("cache_size"));
		cl.def( pybind11::init<const std::string &, bool, const unsigned int, const unsigned int>(), pybind11::arg("filename"), pybind11::arg("profile"), pybind11::arg("cache_size"), pybind11::arg("batch_size") );

		cl.def("Network", (unsigned long (VPUNN::VPUNetworkCostModel::*)(class VPUNN::VPUComputationDAG &, class VPUNN::VPUNetworkStrategy &)) &VPUNN::VPUNetworkCostModel::Network, "Compute the cost of executing a network with a specific per-layer strategy\n\n \n a VPUComputationDAG representing the network to estimate\n \n\n a per-layer strategy\n \n\n unsigned long int\n\nC++: VPUNN::VPUNetworkCostModel::Network(class VPUNN::VPUComputationDAG &, class VPUNN::VPUNetworkStrategy &) --> unsigned long", pybind11::arg("dag"), pybind11::arg("strategy"));
	}
}


#include <map>
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_VPUNN_0(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_2(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_3(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_4(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_5(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_6(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_7(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_8(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_9(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_10(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_11(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_12(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_13(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_14(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_15(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_16(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_17(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_18(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_19(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_20(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_21(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_22(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_23(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_24(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_25(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_26(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_27(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_28(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_29(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_30(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_31(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_32(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_33(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_34(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_35(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_36(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_37(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_38(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_39(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_40(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_VPUNN_41(std::function< pybind11::module &(std::string const &namespace_) > &M);


PYBIND11_MODULE(VPUNN, root_module) {
	root_module.doc() = "VPUNN module";

	std::map <std::string, pybind11::module> modules;
	ModuleGetter M = [&](std::string const &namespace_) -> pybind11::module & {
		auto it = modules.find(namespace_);
		if( it == modules.end() ) throw std::runtime_error("Attempt to access pybind11::module for namespace " + namespace_ + " before it was created!!!");
		return it->second;
	};

	modules[""] = root_module;

	static std::vector<std::string> const reserved_python_words {"nonlocal", "global", };

	auto mangle_namespace_name(
		[](std::string const &ns) -> std::string {
			if ( std::find(reserved_python_words.begin(), reserved_python_words.end(), ns) == reserved_python_words.end() ) return ns;
			else return ns+'_';
		}
	);

	std::vector< std::pair<std::string, std::string> > sub_modules {
		{"", "VPUNN"},
		{"VPUNN", "Dim"},
		{"VPUNN", "intf_01"},
		{"VPUNN", "intf_11"},
		{"", "std"},
		{"std", "chrono"},
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule( mangle_namespace_name(p.second).c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_VPUNN_0(M);
	bind_VPUNN_1(M);
	bind_VPUNN_2(M);
	bind_VPUNN_3(M);
	bind_VPUNN_4(M);
	bind_VPUNN_5(M);
	bind_VPUNN_6(M);
	bind_VPUNN_7(M);
	bind_VPUNN_8(M);
	bind_VPUNN_9(M);
	bind_VPUNN_10(M);
	bind_VPUNN_11(M);
	bind_VPUNN_12(M);
	bind_VPUNN_13(M);
	bind_VPUNN_14(M);
	bind_VPUNN_15(M);
	bind_VPUNN_16(M);
	bind_VPUNN_17(M);
	bind_VPUNN_18(M);
	bind_VPUNN_19(M);
	bind_VPUNN_20(M);
	bind_VPUNN_21(M);
	bind_VPUNN_22(M);
	bind_VPUNN_23(M);
	bind_VPUNN_24(M);
	bind_VPUNN_25(M);
	bind_VPUNN_26(M);
	bind_VPUNN_27(M);
	bind_VPUNN_28(M);
	bind_VPUNN_29(M);
	bind_VPUNN_30(M);
	bind_VPUNN_31(M);
	bind_VPUNN_32(M);
	bind_VPUNN_33(M);
	bind_VPUNN_34(M);
	bind_VPUNN_35(M);
	bind_VPUNN_36(M);
	bind_VPUNN_37(M);
	bind_VPUNN_38(M);
	bind_VPUNN_39(M);
	bind_VPUNN_40(M);
	bind_VPUNN_41(M);

}

// Source list file: /home/fistoc/gitwrk/libraries.performance.modeling.vpu.nn-cost-model/src/python/VPUNN.sources
// VPUNN.cpp
// VPUNN_0.cpp
// VPUNN_1.cpp
// VPUNN_2.cpp
// VPUNN_3.cpp
// VPUNN_4.cpp
// VPUNN_5.cpp
// VPUNN_6.cpp
// VPUNN_7.cpp
// VPUNN_8.cpp
// VPUNN_9.cpp
// VPUNN_10.cpp
// VPUNN_11.cpp
// VPUNN_12.cpp
// VPUNN_13.cpp
// VPUNN_14.cpp
// VPUNN_15.cpp
// VPUNN_16.cpp
// VPUNN_17.cpp
// VPUNN_18.cpp
// VPUNN_19.cpp
// VPUNN_20.cpp
// VPUNN_21.cpp
// VPUNN_22.cpp
// VPUNN_23.cpp
// VPUNN_24.cpp
// VPUNN_25.cpp
// VPUNN_26.cpp
// VPUNN_27.cpp
// VPUNN_28.cpp
// VPUNN_29.cpp
// VPUNN_30.cpp
// VPUNN_31.cpp
// VPUNN_32.cpp
// VPUNN_33.cpp
// VPUNN_34.cpp
// VPUNN_35.cpp
// VPUNN_36.cpp
// VPUNN_37.cpp
// VPUNN_38.cpp
// VPUNN_39.cpp
// VPUNN_40.cpp
// VPUNN_41.cpp

// Modules list file: /home/fistoc/gitwrk/libraries.performance.modeling.vpu.nn-cost-model/src/python/VPUNN.modules
// VPUNN VPUNN.Dim VPUNN.intf_01 VPUNN.intf_11 std std.chrono 
