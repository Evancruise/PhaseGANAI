//vector.h

#ifndef VECTOR_H
#define VECTOR_H
template <class T1>
class CVector
{
    private:
    T1 x;
    T1 y;
    T1 z;
    public:
    CVector(T1 x1=0,T1 y1=0,T1 z1=0)
    {
        x=x1;
        y=y1;
        z=z1;    
    }

    //copy constructor            
    CVector(const CVector<T1> &p)
    {
        this->x=p.x;
        this->y=p.y;
        this->z=p.z;
    }

    T1 GetX(){return x;}
    T1 GetY(){return y;}
    T1 GetZ(){return z;}
    void SetX(T1 x1){x=x1;}
    void SetY(T1 y1){y=y1;}
    void SetZ(T1 z1){z=z1;}

    T1 length(){return sqrt(x*x+y*y+z*z);}
    
    template<class T1>
    friend CVector<T1> operator + (const CVector<T1>& p1,const CVector<T1> &p2);
    
    template<class T1>
    friend CVector<T1> operator - (const CVector<T1>& p1,const CVector<T1> &p2);
    
    template<class T1>
    friend CVector<T1> operator * (T1 t, const CVector<T1> &p);
    
    template<class T1>
    friend CVector<T1> operator * (const CVector<T1> &p, T1 t);
    
    template<class T1>
    friend CVector<T1> operator / (const CVector<T1> &p, T1 t);
    
    template<class T1>
    friend T1 dot(const CVector<T1> &p1,const CVector<T1> &p2);
    
    template<class T1>
    friend CVector<T1> transform(T1 (*A)[3],const CVector<T1> &temp);
};

template<class T1>
CVector<T1> operator +(const CVector<T1> &p1, const CVector<T1> &p2)
{
    return CVector<T1>(p1.x+p2.x,p1.y+p2.y,p1.z+p2.z);
}

template<class T1>
CVector<T1> operator -(const CVector<T1> &p1, const CVector<T1> &p2)
{
    return CVector<T1>(p1.x-p2.x,p1.y-p2.y,p1.z-p2.z);
}

template<class T1>
CVector<T1> operator *(T1 t,const CVector<T1> &p)
{
    return CVector<T1>(p.x*t,p.y*t,p.z*t);
}

template<class T1>
CVector<T1> operator *(CVector<T1>& p, T1 t)
{
    return t*p;
}

template<class T1>
CVector<T1> operator /(const CVector<T1>& p, T1 t)
{
    return T1 (1.0/t) * p;
}

template<class T1>
T1 dot(const CVector<T1> &p1,const CVector<T1> &p2)
{
    return (p1.x*p2.x+p1.y*p2.y+p1.z*p2.z);
}

template<class T1>
CVector<T1> transform(T1 (*A)[3],const CVector<T1> &temp)
{
    //A is a 3*3-element array
    T1 x=A[0][0]*temp.x+A[0][1]*temp.y+A[0][2]*temp.z;
    T1 y=A[1][0]*temp.x+A[1][1]*temp.y+A[1][2]*temp.z;
    T1 z=A[2][0]*temp.x+A[2][1]*temp.y+A[2][2]*temp.z;

    return CVector<T1>(x,y,z);
}

#endif
