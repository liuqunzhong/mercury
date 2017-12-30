#ifndef ANAKIN2_SABER_LAYOUT_H
#define ANAKIN2_SABER_LAYOUT_H

#include "shape.h"

namespace anakin {

namespace saber {

class Layout {
public:
    Layout(){}
    Layout(LayoutType layout):_layout(layout){}
    ~Layout(){}
    inline LayoutType getLayout(){ return _layout; }
    // converter shape 4D to targetLayout, need to be specialized.
    template<LayoutType targetLayout>
    void convert(Shape4D& shape);
    // converter shape 3D to targetLayout
    template<LayoutType targetLayout>
    void convert(Shape3D& shape){
    }
    // converter shape 2D to targetLayout
    template<LayoutType targetLayout>
    void convert(Shape2D& shape){
    }
    // converter shape 1D to targetLayout
    template<LayoutType targetLayout>
    void convert(Shape1D& shape){
    }
private:
    LayoutType _layout;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN2_SABER_LAYOUT_H
