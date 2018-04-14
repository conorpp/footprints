from libc.stdint cimport uint32_t

cdef extern from "slvs.h":
    ctypedef uint32_t Slvs_hParam;
    ctypedef uint32_t Slvs_hEntity;
    ctypedef uint32_t Slvs_hConstraint;
    ctypedef uint32_t Slvs_hGroup;

    ctypedef struct Slvs_Param:
        Slvs_hParam     h;
        Slvs_hGroup     group;
        double          val;

    ctypedef struct Slvs_Entity:
        Slvs_hEntity    h;
        Slvs_hGroup     group;

        int             type;

        Slvs_hEntity    wrkpl;
        Slvs_hEntity    point[4];
        Slvs_hEntity    normal;
        Slvs_hEntity    distance;

        Slvs_hParam     param[4];

    ctypedef struct Slvs_Constraint:
        Slvs_hConstraint    h;
        Slvs_hGroup         group;

        int                 type;

        Slvs_hEntity        wrkpl;

        double              valA;
        Slvs_hEntity        ptA;
        Slvs_hEntity        ptB;
        Slvs_hEntity        entityA;
        Slvs_hEntity        entityB;
        Slvs_hEntity        entityC;
        Slvs_hEntity        entityD;

        int                 other;
        int                 other2;

    ctypedef struct Slvs_System:
        Slvs_Param          *param;
        int                 params;
        Slvs_Entity         *entity;
        int                 entities;
        Slvs_Constraint     *constraint;
        int                 constraints;

        Slvs_hParam         dragged[4];

        int                 calculateFaileds;

        Slvs_hConstraint    *failed;
        int                 faileds;

        int                 dof;

        int                 result;


    void Slvs_Solve(Slvs_System *sys, Slvs_hGroup hg)

    void Slvs_QuaternionU(double qw, double qx, double qy, double qz,
                                 double *x, double *y, double *z)
    void Slvs_QuaternionV(double qw, double qx, double qy, double qz,
                                 double *x, double *y, double *z)
    void Slvs_QuaternionN(double qw, double qx, double qy, double qz,
                                 double *x, double *y, double *z)

    void Slvs_MakeQuaternion(double ux, double uy, double uz,
                             double vx, double vy, double vz,
                             double *qw, double *qx, double *qy, double *qz)


