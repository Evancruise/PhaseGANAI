//vector.h

#ifndef VECTOR_H
#define VECTOR_H

class CVector
{
    private:
    double x;
    double y;
    double z;
    public:
    CVector(double x1=0,double y1=0,double z1=0)
    {
        x=x1;
        y=y1;
        z=z1;    
    }

    //copy constructor            
    CVector(const CVector &p)
    {
        this->x=p.x;
        this->y=p.y;
        this->z=p.z;
    }

    double GetX(){return x;}
    double GetY(){return y;}
    double GetZ(){return z;}
    void SetX(double x1){x=x1;}
    void SetY(double y1){y=y1;}
    void SetZ(double z1){z=z1;}

    double length(){return sqrt(x*x+y*y+z*z);}
    
    friend CVector operator + (const CVector& p1,const CVector &p2);
    friend CVector operator - (const CVector& p1,const CVector &p2);
    friend CVector operator * (double t, const CVector &p);
    friend CVector operator * (const CVector &p, double t);
    friend CVector operator / (const CVector &p, double t);
    
    friend double dot(const CVector &p1,const CVector &p2);
    friend CVector transform(double (*A)[3],const CVector &temp);
};

CVector operator +(const CVector &p1, const CVector &p2)
{
    return CVector(p1.x+p2.x,p1.y+p2.y,p1.z+p2.z);
}

CVector operator -(const CVector &p1, const CVector &p2)
{
    return CVector(p1.x-p2.x,p1.y-p2.y,p1.z-p2.z);
}

CVector operator *(double t,const CVector &p)
{
    return CVector(p.x*t,p.y*t,p.z*t);
}

CVector operator *(CVector& p, double t)
{
    return t*p;
}

CVector operator /(const CVector& p, double t)
{
    return (1.0/t)*p;
}

double dot(const CVector &p1,const CVector &p2)
{
    return (p1.x*p2.x+p1.y*p2.y+p1.z*p2.z);
}

CVector transform(double (*A)[3],const CVector &temp)
{
    //A is a 3*3-element array
    double x=A[0][0]*temp.x+A[0][1]*temp.y+A[0][2]*temp.z;
    double y=A[1][0]*temp.x+A[1][1]*temp.y+A[1][2]*temp.z;
    double z=A[2][0]*temp.x+A[2][1]*temp.y+A[2][2]*temp.z;

    return CVector(x,y,z);
}

#endif
