///////////////////////////////////////////////////////////////////////////
//         ConeBeamCT forward projection subroutine.
//This subroutine calculate the X-ray transform of a mathematical phantom
//that contains N=10 ellipsoids and spheres. 
//rho: the refractive index data 
//semiaxis: the semiaxis of the ellipsoids
//Center[]: center of the ellipsoids
//alpha[] : rotated angle of the ellipsoids
//DSO     : the distance between the source and the object
//ZETAS   : # of rows of the projection image
//RAYS    : # of columns of the projection image
//ZMAX    : the maximum size of the phantom along the z-axis
//ZMIN    : the minimum size of the phantom along the z-axis
//XMAX    : the maximum size of the phantom along the x-axis  
//XMIN    : the minimum size of the phantom along the x-axis  
//YMAX    : the maximum size of the phantom along the y-axis  
//YMIN    : the minimum size of the phantom along the y-axis  
//VIEWS   : # of tomographic views and since this is a full scan problem, 
//          the views spans the 2 pi region
//The output projection data is named with the views as Shepp_001.prj 
//
//                                                   March 10 2004
///////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "vector_Phan.h"
//DELTA_X is not the resolution on the detector plane
//The pixel size on the detector plane should be computed as
//
// 2.0*FACTOR/RAYS

#define DELTA_X          (1.0e-6)
#define VIEWS (1)
    
    
#define ZETAS (256)
#define RAYS  (256)

#define FACTOR (128*DELTA_X)

#define DSO   (0.025)
#define N     (10)
#define ZMAX  (1.0*FACTOR)
#define ZMIN  (-ZMAX)
#define XMAX  (1.0*FACTOR)
#define XMIN  (-XMAX)
#define YMAX  (XMAX)
#define YMIN  (-YMAX)

#ifndef PI
#define PI    (4.0*atan(1.0))
#endif

//Given line passing through p1,p2,
//calculate the vector theta and point y
//in forward projection

void Calculate_Theta_Y(const CVector &p1,const CVector & p2,CVector &theta,
                       CVector &y)
{

//calculate the vector P1P2
    CVector p=p2-p1;
    
//now the vector p is the direction vector and I need to define a unit 
//vector which can be done as p/norm(p);

    theta=p/p.length();

//now I need to find the vector y which sit on the theta purb. In fact
//both the vector p1 and p2 are sitting on the theta purb. I only need
//to pick either one of them.
    y=p1;

}

double Projection(double (*A)[3],CVector theta,CVector y)
//A is a 3*3-element array
//y lies in theta purb
{
    CVector A_theta=transform(A,theta);
    CVector A_Y=transform(A,y);
    
    //calculate E_Atheta/||Athea||(AY)
    
    double length1=A_Y.length();
    double length2=A_theta.length();
    double AY_dot_ATheta=dot(A_Y,A_theta);
    
    double t=length1*length1-AY_dot_ATheta*AY_dot_ATheta/ \
             (length2*length2);
    if(t>1.0)
        return 0.0;
    else
        return 2.0*sqrt(1.0-t)/length2;
}

// [optical axis, x in matlab, y in matlab] 
int main(int argc,char *argv[])
{
    int i,j,k,m;
//    static double semiaxes[][3]={   0.69  ,0.92  ,0.8  ,//1
      static double semiaxes[][3]={ 0.8,0.8,0.8,
                                    0.62, 0.85 ,0.75 ,//2
                                    0.38  ,0.16  ,0.21 ,//3
                                    0.31  ,0.11  ,0.22 ,//4
                                    0.26  ,0.16  ,0.40  ,//5
                                    0.046 ,0.046 ,0.046,//6
                                    0.046 ,0.023 ,0.02 ,//7
                                    0.046 ,0.023 ,0.02 ,//8
                                    0.064 ,0.08  ,0.1  ,//9
                                    0.22 ,0.22  ,0.1   //10
                                    };
    static double Center[][3]={ 0.0,        0.0,       0.0,       //ellipsoid 1
                                0.0,        0.0,       0.0,       //ellipsoid 2
                                0.22,      -0.18,      -0.2,      //ellipsoid 3
                               -0.28,      -0.1,      -0.2,      //ellipsoid 4
                                0.0,        0.47,     -0.16,      //ellipsoid 5
                                0.0,        0.1,      -0.25,      //ellipsoid 6
                               -0.08,      -0.605,    -0.25,      //ellipsoid 7
                                0.06,      -0.605,    -0.25,      //ellipsoid 8
                                0.06,      -0.45,     0.625,     //ellipsoid 9
                                0.08,      -0.3,      -0.2      //ellipsoid 10
                                };
//    double rho[N]={0.5e-8,    -0.6e-8,
//                 0.e-8,    1.2e-8,
//                   1.e-8,     0.e-8,
//                   0.e-8,     0.e-8,
//                   0.e-8,    -1.e-8
//                   }; //densities of the ellipsoids
//    double mu[N]={ 0.5e-9,    -0.5e-9,
//                   .6e-9,     0.e-9,
//                   .5e-9,     0.e-9,
//                   0.e-9,     0.e-9,
//                   0.e-9,     .3e-9
//                  }; //densities of the ellipsoids

	double rho[N]={0.5e-8,0,
			0,0,
			0,0,
			0,0,
			0,0
			};
	double mu[N]={0.5e-9,0,
		      0,0,
		      0,0,
		      0,0,
		      0,0
		      };

    double alpha[N]={0, 0, 108, 72, 0, 
                     0, 0, 90, 90, 0};
    for(i=0;i<N;i++)
    {
        for(j=0;j<3;j++)
        {
            semiaxes[i][j] *=FACTOR;
            Center[i][j] *=FACTOR;
//We can scale the object as follows,
              semiaxes[i][j] *=0.3;
              Center[i][j] *=0.3;
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
    double (*proj_rho)[RAYS]=new double [ZETAS][RAYS];
    double (*proj_mu)[RAYS]=new double [ZETAS][RAYS];
    
    CVector Theta,Y;
    
    double phi_intvl=PI/VIEWS;
    for(m=0;m<VIEWS;m++)
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
        fp=fopen(fn,"wb");
        fwrite(proj_mu,sizeof(double)*ZETAS*RAYS,1,fp);
        fclose(fp);
    }
                         
    return 0;
}
