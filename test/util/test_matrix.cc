/*********************************************************************************
*     File Name           :     test_matrix.cc
*     Created By          :     yuewu
*     Description         :     test use of matrix
**********************************************************************************/

#include <iostream>
#include <vector>

#include <lsol/math/matrix.h>
#include <lsol/math/vector.h>

using namespace lsol::math;
using namespace lsol::math::expr;

using namespace std;

int main() {
	// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(tmpFlag);
	//_CrtSetBreakAlloc(231);
#endif
	cout << "Test matrix" << endl;
	Matrix<float> m({ 2, 2 });
	m.assign(2);
	cout << "assign 2:" << endl << m << endl;
	cout << "m(0,0):" << m(0,0) << endl;
	m += 3;
	cout << "plus 3:" << endl << m << endl;
	m = 1;
	cout << "assign 1:" << endl << m << endl;

	Matrix<float> m2;
	cout << "constructor: " << endl << m2 << endl;
	m2 = m;
	cout << "assignment: " << endl << m2 << endl;
	m2 = m2 + 2.f;
	cout << "plus 2: " << endl << m2 << endl;
	m2 = (3.f + m2) * 2.f;
	cout << "plus 3 mul 2: " << endl << m2 << endl;

	m2 = m2 / 2.f;
	cout << "div 2: " << endl << m2 << endl;
	m2 = m2 - 2.f;
	cout << "minux 2: " << endl << m2 << endl;
	m2 = m2 * 2.f;
	cout << "mul 2: " << endl << m2 << endl;

	cout << "original matrix:" << endl << m << endl;

	cout << "Test vector" << endl;
	Vector<float> v(3);
	v.push_back(1);
	v.push_back(2);
	v.push_back(3);
	cout << "push: "<<endl<<v<<endl;
	v.pop_back();
	cout << "pop: "<<endl<<v<<endl;
	cout << "front: " <<v.front()<<endl;
	cout << "back: " <<v.back()<<endl;
	cout << "dimsension: "<<v.dim() << endl;

	Vector<float> v2 = v;
	Vector<int> v3;
	cout << "copy constructor: " << endl << v2 << endl;
	v2 = v;
	cout << "assignment: " << endl << v2 << endl;
	v2 += 2.f;
	cout << "plus 2: " << endl << v2 << endl;
	v2 = v2 + 2.f;
	cout << "plus 2: " << endl << v2 << endl;
	v2 = v2 /  2.f;
	cout << "div 2: " << endl << v2 << endl;
	v2 = v2 -  2.f;
	cout << "minux 2: " << endl << v2 << endl;
	v2 = v2 * 2.f;
	cout << "mul 2: " << endl << v2 << endl;

	cout << "original vector:" << endl << v2 << endl;

	/*
	cout << "Test sparse vector" << endl;
	SparseVector<uint32_t, float> sv(3);
	sv.push_back(0, 1);
	sv.push_back(1, 2);
	sv.push_back(2, 3);
	cout << "push: "<<endl<<sv<<endl;
	sv.pop_back();
	cout << "pop: "<<endl<<sv<<endl;
	cout << "front: " <<sv.front()<<endl;
	cout << "back: " <<sv.back()<<endl;
	cout << "dimsension: "<<sv.dim() << endl;

	SparseVector<uint32_t, float> sv2 = sv;
	cout << "copy constructor: " << endl << sv2 << endl;
	sv2 = sv;
	cout << "assignment: " << endl << sv2 << endl;
	//sv2 += SparseItem<uint32_t, float>(2.f);
	sv2 += 2.1f;
	cout << "plus 2.1: " << endl << sv2 << endl;
	sv2 = sv2 + 2.f;
	cout << "plus 2: " << endl << sv2 << endl;
	sv2 = sv2 /  2.f;
	cout << "div 2: " << endl << sv2 << endl;
	sv2 = sv2 -  2.f;
	cout << "minux 2: " << endl << sv2 << endl;
	sv2 = sv2 * 2.f;
	cout << "mul 2: " << endl << sv2 << endl;
	cout << "v2 : " << endl << v2 << endl;
	cout << "sv2 : " << endl << sv2 << endl;
	v2 *= sv2;
	cout << "v2 * sv2 : " << endl << v2 << endl;
	cout << "original vector:" << endl << sv2 << endl;
	*/
	return 0;
}
