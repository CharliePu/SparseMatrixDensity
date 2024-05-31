#include <boost/python.hpp>
#include "MatrixGenerator.h"
#include "EntryGenerator.h"

BOOST_PYTHON_MODULE(MatrixGenerator)
{
    using namespace boost::python;
    def("generate_entry", generate_entry_rectangle_matrices);
    def("generate_entry_rectangle_matrices", generate_entry_rectangle_matrices);
    def("generate_entry_square_matrices", generate_entry_square_matrices);
    def("generate_entry_inner_product", generate_entry_inner_product);
    def("generate_entry_outer_product", generate_entry_outer_product);
    def("generate_entry_extreme_cases", generate_entry_extreme_cases);
}
