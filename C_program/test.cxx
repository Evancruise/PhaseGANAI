/*********************************************************/
/**                       Fresnel.cxx                   **/
/** This routine is to calculate the forward projection **/
/** of Bronnikov method.                                **/
/**                                Aug. 08 2006         **/
/*********************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>


#include "vector.h"
#include "fftw.h"


#ifndef PI
#define PI   (4.0*atan(1.0))
#endif 

#define DO               (0.025)
#define LAMBDA           (1e-10)
#define VIEWS            (180)
#define RAYS             (1024)
#define ZETAS            (RAYS)

#define K                (2.0*PI/LAMBDA)
#define DELTA_X          (1e-6)


#define VOXEL_NO       (256)





#define N     (10)

#define FACTOR (100*DELTA_X)
#define ZMAX  (1.0*FACTOR)
#define ZMIN  (-ZMAX)
#define XMAX  (1.0*FACTOR)
#define XMIN  (-XMAX)
#define YMAX  (XMAX)
#define YMIN  (-YMAX)






template <class T1>
void Calculate_Theta_Y(const CVector<T1> &p1,const CVector<T1> & p2,
        CVector<T1> &theta, CVector<T1> &y)
{

//calculate the vector P1P2
    CVector<T1> p=p2-p1;
    

    theta=p/p.length();

    y=p1;

}

template<class T1>
T1 Projection(T1 (*A)[3],CVector<T1> theta,CVector<T1> y)
{
    CVector<T1> A_theta=transform(A,theta);
    CVector<T1> A_Y=transform(A,y);
    
    //calculate E_Atheta/||Athea||(AY)
    
    T1 length1=A_Y.length();
    T1 length2=A_theta.length();
    T1 AY_dot_ATheta=dot(A_Y,A_theta);
    
    T1 t=length1*length1-AY_dot_ATheta*AY_dot_ATheta/ \
             (length2*length2);
    if(t>1.0)
        return 0.0;
    else
        return 2.0*sqrt(1.0-t)/length2;
}


template<class T1>
void radon(T1 a)
{
    int i,j,k,m;
    static double semiaxes[][3]={   0.69  ,0.92  ,0.9  ,//1
                                    0.6624,0.874 ,0.88 ,//2
                                    0.41  ,0.16  ,0.21 ,//3
                                    0.31  ,0.11  ,0.22 ,//4
                                    0.21  ,0.25  ,0.5  ,//5
                                    0.046 ,0.046 ,0.046,//6
                                    0.046 ,0.023 ,0.02 ,//7
                                    0.046 ,0.023 ,0.02 ,//8
                                    0.056 ,0.04  ,0.1  ,//9
                                    0.056 ,0.04  ,0.1   //10
                                    };
    static double Center[][3]={ 0.0,        0.0,       0.0,       //ellipsoid 1
                                0.0,        0.0,       0.0,       //ellipsoid 2
                               -0.22,       0.0,      -0.25,      //ellipsoid 3
                                0.22,       0.0,      -0.25,      //ellipsoid 4
                                0.0,        0.35,     -0.25,      //ellipsoid 5
                                0.0,        0.1,      -0.25,      //ellipsoid 6
                               -0.08,      -0.605,    -0.25,      //ellipsoid 7
                                0.06,      -0.605,    -0.25,      //ellipsoid 8
                                0.06,      -0.605,     0.625,     //ellipsoid 9
                                0.0,        0.1,       0.625      //ellipsoid 10
                                };
    T1 rho[N]={
                   2.0e-8,    -0.98e-8,
                  -0.2e-8,    -0.2e-8,
                   0.2e-8,     0.2e-8,
                   0.2e-8,     0.2e-8,
                   0.2e-8,    -0.2e-8
               }; 
    T1 mu[N]={
                   2.0e-8,    -0.98e-8,
                  -0.2e-8,    -0.2e-8,
                   0.2e-8,     0.2e-8,
                   0.2e-8,     0.2e-8,
                   0.2e-8,    -0.2e-8
               }; 

    double alpha[N]={0, 0, 108, 72, 0, 
                     0, 0, 90, 90, 0};
    
    for(i=0;i<N;i++)
    {
        for(j=0;j<3;j++)
        {
            semiaxes[i][j] *=FACTOR;
            Center[i][j] *=FACTOR;
        }
    }
                                    
    static double D[N][3][3];
    for(i=0;i<N;i++)
    {
        for(j=0;j<3;j++)
        {
            for(k=0;k<3;k++)
            {
                if(k==j)
                    D[i][j][k]=1.0/semiaxes[i][k];
                else
                    D[i][j][k]=0.0;
            }
        }
    }
    static double OrgV[N][3][3];
    for(k=0;k<N;k++)
    {
        OrgV[k][0][0]=cos(alpha[k]/180.0*PI);
        OrgV[k][0][1]=sin(alpha[k]/180.0*PI);
        OrgV[k][0][2]=0.0;

        OrgV[k][1][0]=-sin(alpha[k]/180.0*PI);
        OrgV[k][1][1]=cos(alpha[k]/180.0*PI);
        OrgV[k][1][2]=0.0;

        OrgV[k][2][0]=0.0;
        OrgV[k][2][1]=0.0;
        OrgV[k][2][2]=1.0;
    }
                           
    double A[N][3][3];
    
    for(i=0;i<N;i++)
    {
        for(j=0;j<3;j++)
        {
            for(k=0;k<3;k++)
            {
                A[i][j][k]=0.0;
                for(m=0;m<3;m++)
                    A[i][j][k] +=D[i][j][m]*OrgV[i][m][k];
                    
            }
        }
    }

    //zeta goes from ZMAX to ZMIN 
    //s    goes from YMIN to YMAX 
    //Dso=DSO;
    //the original point P1=(DSO,0,0)
    //the z component of P2 varies via zeta
    //the y component of P2 varies via s
    //the x component of P2 is 0
    //when the source rotates, P2 also rotates 

    
    double zMax=ZMAX;
    double zMin=ZMIN;
    double z_intvl=(zMax-zMin)/ZETAS;
    
    double sMin=YMIN;
    double sMax=YMAX;
    double s_intvl=(sMax-sMin)/RAYS;

    CVector P1,P2;
    double x,y,z;
    double x1,y1,z1;
    T1 (*proj_rho)[RAYS]=new T1 [ZETAS][RAYS];
    T1 (*proj_mu)[RAYS]=new T1 [ZETAS][RAYS];
    
    CVector Theta,Y;
    
    double phi_intvl=2.0*PI/VIEWS;
	m=0;
   // for(m=0;m<VIEWS;m++)
    {
        double phi=m*phi_intvl;
        
        x=DSO;
        y=0.0;
        z=0.0;

        x1=x*cos(phi)-y*sin(phi);
        y1=x*sin(phi)+y*cos(phi);
        z1=z;

        P1=CVector(x1,y1,z1);

        for(i=0;i<ZETAS;i++)
        {
            z=zMax-i*z_intvl;
            for(j=0;j<RAYS;j++)
            {
                y=sMin+j*s_intvl;
                x=0.0;
                //rotate P2
                x1=x*cos(phi)-y*sin(phi);
                y1=x*sin(phi)+y*cos(phi);
                z1=z;
                P2=CVector(x1,y1,z1);
                
                Calculate_Theta_Y(P1,P2,Theta,Y);
                double sum_rho=0.0;
                double sum_mu=0.0;
                for(k=0;k<N;k++)
                {
                    /*******************************************/
                    /** BE CAREFUL HERE, Xm IS THE VECTOR IN  **/
                    /** THE OLD COORDINATE SYSTEM WHICH MEANS **/
                    /** THAT IT DOES NOT NEED TO BE ROTATED AS**/
                    /** I DID WITH P1 AND P2. SINCE THETA IS  **/
                    /** IS CHANGED, E_THETA_Xm WILL BE CHANGED**/
                    /*******************************************/
                    CVector Xm=CVector(Center[k][0],Center[k][1],Center[k][2]);
                    CVector E_theta_Xm=Xm-dot(Xm,Theta)*Theta;
                    
                    double t=Projection(A[k],Theta,Y-E_theta_Xm);
                    sum_rho += t*rho[k];
                    sum_mu += t*mu[k];
                }
                proj_rho[i][j]=sum_rho;
                proj_mu[i][j]=sum_mu;
            }
        }
        printf("m=%d\n",m+1);
        char fn[80];
        sprintf(fn,"phi_%03d.prj",m+1);
        FILE *fp=fopen(fn,"wb");
        fwrite(proj_rho,sizeof(double)*ZETAS*RAYS,1,fp);
        fclose(fp);

        sprintf(fn,"mu_%03d.prj",m+1);
        FILE *fp=fopen(fn,"wb");
        fwrite(proj_mu,sizeof(double)*ZETAS*RAYS,1,fp);
        fclose(fp);
    }
}

int main()
{
    fftw_real a=1.0;
    radon(a);
//    DrawPhantom(a);
    return 0;
}
