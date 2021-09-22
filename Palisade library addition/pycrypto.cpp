// @file pycrypto.cpp This code provide a simple python wrapper to PALISADE
// @author TPOC: contact@palisade-crypto.org
//
// @copyright Copyright (c) 2020, New Jersey Institute of Technology (NJIT)
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. THIS SOFTWARE IS
// PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <vector>
#include <complex>

#include <boost/python.hpp>
#include "ckks_wrapper.h"
#include <palisade/pke/palisade.h>
#include <iostream>

using namespace boost::python;

#define make_tuple boost::python::make_tuple

class cppVectorToPythonList {
 public:
  /**
   * Convert from vector<std::complex<double>> to python list.
   * Only real parts of vector<std::complex<double>> are stored
   */
  static PyObject* convert(const std::vector<std::complex<double>>& vector) {
    boost::python::list* pythonList = new boost::python::list();
    for (unsigned int i = 0; i < vector.size(); i++) {
      pythonList->append(vector[i].real());
    }
    return pythonList->ptr();
  }
};

struct CKKS_pickle_suite : boost::python::pickle_suite
{
    static
    boost::python::tuple
    getinitargs(const pycrypto::CKKSwrapper& wrapper)
    {
        using namespace boost::python;
        return make_tuple();
    }

    static
    boost::python::tuple
    getstate(boost::python::object wrapper_obj)
    {
        using namespace boost::python;
        pycrypto::CKKSwrapper const& wrapper = extract<pycrypto::CKKSwrapper const&>(wrapper_obj)();
        return make_tuple(wrapper_obj.attr("__dict__"),
                              boost::ref(wrapper.get_m_cc()),
                              boost::ref(wrapper.get_m_keys()));
    }

    static
    void
    setstate(boost::python::object w_obj, boost::python::tuple state)
    {
        using namespace boost::python;
        std::cout << "Start extract in setstate" << std::endl;
        pycrypto::CKKSwrapper& w = extract<pycrypto::CKKSwrapper&>(w_obj)();
        
        // restore the object's __dict__

        std::cout << "Extract dict" << std::endl;
        dict d = extract<dict>(w_obj.attr("__dict__"))();

        std::cout << "Update dict" << std::endl;
        d.update(state[0]);
        
        // restore the internal state of the C++ object

        std::cout << "Set context" << std::endl;
        lbcrypto::CryptoContext<lbcrypto::DCRTPoly> context = extract<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(state[1]);

        std::cout << "Set keys" << std::endl;
        lbcrypto::LPKeyPair<lbcrypto::DCRTPoly> keys = extract<lbcrypto::LPKeyPair<lbcrypto::DCRTPoly>>(state[2]);


        w.set_m_cc(context);
        w.set_m_keys(keys);
    }
};

BOOST_PYTHON_MODULE(pycrypto) {
  /*
   * Whenever a vector<std::complex<double> is returned by a function,
   * it will automatically be converted to a Python list
   * with real parts of complex values in vector<std::complex<double>
   */
  to_python_converter<std::vector<std::complex<double>>,
                      cppVectorToPythonList>();

  class_<pycrypto::CiphertextInterfaceType>("Ciphertext");

  class_<pycrypto::CKKSwrapper>("CKKSwrapper")
      .def("KeyGen", &pycrypto::CKKSwrapper::KeyGen)
      .def("Encrypt", &pycrypto::CKKSwrapper::Encrypt,
           return_value_policy<manage_new_object>())
      .def("Decrypt", &pycrypto::CKKSwrapper::Decrypt)
      .def("EvalAdd", &pycrypto::CKKSwrapper::EvalAdd,
           return_value_policy<manage_new_object>())
      .def("EvalSub", &pycrypto::CKKSwrapper::EvalSub,
           return_value_policy<manage_new_object>())
      .def("EvalInnerProduct", &pycrypto::CKKSwrapper::EvalInnerProduct,
           return_value_policy<manage_new_object>())
      .def("EvalMult", &pycrypto::CKKSwrapper::EvalMult,
           return_value_policy<manage_new_object>())
      .def("EvalMultAndRelinearize", &pycrypto::CKKSwrapper::EvalMultAndRelinearize,
           return_value_policy<manage_new_object>())
      .def("EvalMultConst", &pycrypto::CKKSwrapper::EvalMultConst,
           return_value_policy<manage_new_object>())
      .def("EvalSum", &pycrypto::CKKSwrapper::EvalSum,
           return_value_policy<manage_new_object>())
      .def_pickle(CKKS_pickle_suite())
      ;
}
