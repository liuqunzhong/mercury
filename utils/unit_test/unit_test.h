#ifndef UTILS_TEST_AKTEST_H
#define UTILS_TEST_AKTEST_H

#include "engine_test.h"
#include "test_base.h"


#define TEST(test_class, test_function)	\
	class test_class##_##test_function:public test_class{\
	public:\
		friend class ::mercury::test::EnginResOp;\
		void test_function();\
	};\
	const test_class##_##test_function _##test_class##_##test_function;\
	std::function<void(void)> func_##test_class##_##test_function =std::bind(&test_class##_##test_function::test_function,_##test_class##_##test_function); \
	::mercury::test::EnginResOp op_test_class##_##test_function = (::mercury::test::EnginResOp(#test_class,#test_function)\
	>>test_class::get_instance<test_class>() & func_##test_class##_##test_function);\
	void test_class##_##test_function::test_function()


#define InitTest()\
	::mercury::test::config::initial()

#define RUN_ALL_TESTS(argv_0) \
	::mercury::test::EngineTest::get_instance().run_all(argv_0)



#endif 
