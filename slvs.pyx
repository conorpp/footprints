cimport slvs
from libc.string cimport memset
from libc.stdlib cimport malloc, free

#/* To obtain the 3d (not projected into a workplane) of a constraint or
# * an entity, specify this instead of the workplane. */
SLVS_FREE_IN_3D         = 0

SLVS_E_POINT_IN_3D          = 50000
SLVS_E_POINT_IN_2D          = 50001

SLVS_E_NORMAL_IN_3D         = 60000
SLVS_E_NORMAL_IN_2D         = 60001

SLVS_E_DISTANCE             = 70000

SLVS_E_WORKPLANE            = 80000
SLVS_E_LINE_SEGMENT         = 80001
SLVS_E_CUBIC                = 80002
SLVS_E_CIRCLE               = 80003
SLVS_E_ARC_OF_CIRCLE        = 80004

SLVS_C_POINTS_COINCIDENT        = 100000
SLVS_C_PT_PT_DISTANCE           = 100001
SLVS_C_PT_PLANE_DISTANCE        = 100002
SLVS_C_PT_LINE_DISTANCE         = 100003
SLVS_C_PT_FACE_DISTANCE         = 100004
SLVS_C_PT_IN_PLANE              = 100005
SLVS_C_PT_ON_LINE               = 100006
SLVS_C_PT_ON_FACE               = 100007
SLVS_C_EQUAL_LENGTH_LINES       = 100008
SLVS_C_LENGTH_RATIO             = 100009
SLVS_C_EQ_LEN_PT_LINE_D         = 100010
SLVS_C_EQ_PT_LN_DISTANCES       = 100011
SLVS_C_EQUAL_ANGLE              = 100012
SLVS_C_EQUAL_LINE_ARC_LEN       = 100013
SLVS_C_SYMMETRIC                = 100014
SLVS_C_SYMMETRIC_HORIZ          = 100015
SLVS_C_SYMMETRIC_VERT           = 100016
SLVS_C_SYMMETRIC_LINE           = 100017
SLVS_C_AT_MIDPOINT              = 100018
SLVS_C_HORIZONTAL               = 100019
SLVS_C_VERTICAL                 = 100020
SLVS_C_DIAMETER                 = 100021
SLVS_C_PT_ON_CIRCLE             = 100022
SLVS_C_SAME_ORIENTATION         = 100023
SLVS_C_ANGLE                    = 100024
SLVS_C_PARALLEL                 = 100025
SLVS_C_PERPENDICULAR            = 100026
SLVS_C_ARC_LINE_TANGENT         = 100027
SLVS_C_CUBIC_LINE_TANGENT       = 100028
SLVS_C_EQUAL_RADIUS             = 100029
SLVS_C_PROJ_PT_DISTANCE         = 100030
SLVS_C_WHERE_DRAGGED            = 100031
SLVS_C_CURVE_CURVE_TANGENT      = 100032
SLVS_C_LENGTH_DIFFERENCE        = 100033

SLVS_RESULT_OKAY                = 0
SLVS_RESULT_INCONSISTENT        = 1
SLVS_RESULT_DIDNT_CONVERGE      = 2
SLVS_RESULT_TOO_MANY_UNKNOWNS   = 3


cdef void check_malloc(void * mem):
    if mem is NULL:
        raise MemoryError()

cdef void check_free(void * mem):
    if mem is not NULL:
        free(mem)

class Param:
    def __init__(self,h,val):
        self.h = h
        self.val = val

cdef class SlvsResult:

    cdef public int status
    cdef public int dof
    param  = []
    failed = []

    def __init__(self, ):
        pass

    cdef assign(self, Slvs_System * sys):

        self.dof = sys[0].dof
        self.status = sys[0].result

        cdef Slvs_Param par
        for i in range(0,sys[0].params):
            par = sys[0].param[i]
            self.param.append(Param(par.h,par.val))

        for i in range(0,sys[0].faileds):
            self.failed.append(sys[0].failed[i])


cdef class Slvs:
    cdef Slvs_System sys

    param = []
    entity = []
    constraint = []

    cdef public uint32_t paramid
    cdef public uint32_t entityid
    cdef public uint32_t constraintid

    cdef public Slvs_hEntity origin
    cdef public Slvs_hEntity normal
    cdef public Slvs_hEntity workplane

    cdef public Slvs_hGroup group

    def __cinit__(self,):
        memset(&self.sys, 0, sizeof(Slvs_System))
        self.paramid = 0x8000
        self.entityid = 0x8000
        self.constraintid = 0x8000
        self.group = 1


        # First, we create our workplane. Its origin corresponds to the origin
        # of our base frame (x y z) = (0 0 0)
        self.origin = self.MakePoint3d(self.P(0.0), self.P(0.0), self.P(0.0));
        # and it is parallel to the xy plane, so it has basis vectors (1 0 0)
        # and (0 1 0).
        (qw,qx,qy,qz) = self.MakeQuaternion(1, 0, 0,
                                         0, 1, 0);

        self.normal = self.MakeNormal3d(self.P(qw), self.P(qx), self.P(qy), self.P(qz));

        self.workplane = self.MakeWorkplane(self.origin, self.normal);

        self.group = 2


    def __dealloc__(self):
        self.free_sys()

    cdef free_sys(self):
        check_free(self.sys.param)
        check_free(self.sys.entity)
        check_free(self.sys.constraint)
        check_free(self.sys.failed)
        memset(&self.sys, 0, sizeof(Slvs_System))

    cdef alloc_sys(self, items):
        memset(&self.sys, 0, sizeof(Slvs_System))

        self.sys.param = <Slvs_Param *> malloc(items * sizeof(Slvs_Param))
        check_malloc(self.sys.param)
        self.sys.entity = <Slvs_Entity *> malloc(items * sizeof(Slvs_Entity))
        check_malloc(self.sys.entity)
        self.sys.constraint = <Slvs_Constraint*> malloc(items * sizeof(Slvs_Constraint))
        check_malloc(self.sys.constraint)
        self.sys.failed = <Slvs_hConstraint*> malloc(items * sizeof(Slvs_Constraint))
        check_malloc(self.sys.failed)
        self.sys.faileds = items

        self.sys.calculateFaileds = 1

    cpdef set_group(self, Slvs_hGroup group):
        self.group = group


    def solve(self,group = None):
        cdef Slvs_Param par

        if group is None:
            group = self.group

        items = max(len(self.param), len(self.entity), len(self.constraint))

        self.free_sys()
        self.alloc_sys(items)
        
        # copy over records
        for i in range(0,len(self.param)):
            self.sys.param[i] = self.param[i]
            self.sys.params += 1
        for i in range(0,len(self.entity)):
            self.sys.entity[i] = self.entity[i]
            self.sys.entities += 1
        for i in range(0,len(self.constraint)):
            self.sys.constraint[i] = self.constraint[i]
            self.sys.constraints += 1


        slvs.Slvs_Solve(&self.sys, group)
        res = self.sys.result
        dof = self.sys.dof

        sr = SlvsResult()
        sr.assign(&self.sys)

        return sr


    cpdef MakeQuaternion(self,double ux, double uy, double uz, double vx, double vy, double vz):
        cdef double qw, qx, qy, qz
        slvs.Slvs_MakeQuaternion(ux,uy,uz,vx,vy,vz, &qw, &qx, &qy, &qz)
        return (qw,qx,qy,qz)


    cdef add_param(self, Slvs_Param par):
        self.param.append(par)

    cdef add_entity(self, Slvs_Entity ent):
        self.entity.append(ent)

    cdef add_constraint(self, Slvs_Constraint con):
        self.constraint.append(con)


    def MakeParam(self, double val, ** kwargs):
        #par = slvs.Slvs_MakeParam(ref,g,num)
        cdef Slvs_Param par;
        par.h = self.paramId(kwargs)
        par.group = self.group
        par.val = val
        self.add_param(par)
        return par.h

    def P(self, double val, ** kwargs):
        return self.MakeParam(val,**kwargs)


    def MakePoint2d(self,
                   Slvs_hParam u, Slvs_hParam v, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_POINT_IN_2D;
        r.wrkpl = self.workplane;
        r.param[0] = u;
        r.param[1] = v;
        self.add_entity(r)
        return r.h

    def MakePoint3d(self,
            Slvs_hParam x, Slvs_hParam y, Slvs_hParam z, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_POINT_IN_3D;
        r.wrkpl = SLVS_FREE_IN_3D;
        r.param[0] = x;
        r.param[1] = y;
        r.param[2] = z;
        self.add_entity(r)
        return r.h

    def MakeNormal3d(self,
                    Slvs_hParam qw, Slvs_hParam qx,
                    Slvs_hParam qy, Slvs_hParam qz, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_NORMAL_IN_3D;
        r.wrkpl = SLVS_FREE_IN_3D;
        r.param[0] = qw;
        r.param[1] = qx;
        r.param[2] = qy;
        r.param[3] = qz;
        self.add_entity(r)
        return r.h

    def MakeNormal2d(self,  **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_NORMAL_IN_2D;
        r.wrkpl = self.workplane;
        self.add_entity(r)
        return r.h

    def MakeDistance(self,
            Slvs_hParam d, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_DISTANCE;
        r.wrkpl = self.workplane;
        r.param[0] = d;
        self.add_entity(r)
        return r.h

    def MakeLineSegment(self,
                       Slvs_hEntity ptA, Slvs_hEntity ptB, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_LINE_SEGMENT;
        r.wrkpl = self.workplane;
        r.point[0] = ptA;
        r.point[1] = ptB;
        self.add_entity(r)
        return r.h

    def MakeCubic(self,
                 Slvs_hEntity pt0, Slvs_hEntity pt1,
                 Slvs_hEntity pt2, Slvs_hEntity pt3, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_CUBIC;
        r.wrkpl = self.workplane;
        r.point[0] = pt0;
        r.point[1] = pt1;
        r.point[2] = pt2;
        r.point[3] = pt3;
        self.add_entity(r)
        return r.h

    def MakeArcOfCircle(self,
                       Slvs_hEntity center,
                       Slvs_hEntity start, Slvs_hEntity end, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_ARC_OF_CIRCLE;
        r.wrkpl = self.workplane;
        r.normal = self.normal;
        r.point[0] = center;
        r.point[1] = start;
        r.point[2] = end;
        self.add_entity(r)
        return r.h

    def MakeCircle(self, 
                  Slvs_hEntity center,
                  Slvs_hEntity radius, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_CIRCLE;
        r.wrkpl = self.workplane;
        r.point[0] = center;
        r.normal = self.normal;
        r.distance = radius;
        self.add_entity(r)
        return r.h

    def MakeWorkplane(self,
            Slvs_hEntity origin, Slvs_hEntity normal, **kwargs):
        cdef Slvs_Entity r;
        memset(&r, 0, sizeof(r));
        r.h = self.entityId(kwargs)
        r.group = self.group;
        r.type = SLVS_E_WORKPLANE;
        r.wrkpl = SLVS_FREE_IN_3D;
        r.point[0] = origin;
        r.normal = normal;
        self.add_entity(r)
        return r.h

    def MakeConstraint(self,
                          int _type,
                          double valA,
                          Slvs_hEntity ptA,
                          Slvs_hEntity ptB,
                          Slvs_hEntity entityA,
                          Slvs_hEntity entityB, **kwargs):
        cdef Slvs_Constraint r;
        memset(&r, 0, sizeof(r));
        r.h = self.constraintId(kwargs)
        r.group = self.group;
        r.type = _type;
        r.wrkpl = self.workplane;
        r.valA = valA;
        r.ptA = ptA;
        r.ptB = ptB;
        r.entityA = entityA;
        r.entityB = entityB;
        self.add_constraint(r)
        return r.h

    cdef Slvs_hParam paramId(self,dic):
        hid = dic.get('id',None)
        if hid is None:
            self.paramid += 1
            return <Slvs_hParam>self.paramid
        return <Slvs_hParam>hid

    cdef Slvs_hEntity entityId(self,dic):
        hid = dic.get('id',None)
        if hid is None:
            self.entityid += 1
            return <Slvs_hEntity>self.entityid
        return <Slvs_hEntity>hid

    cdef Slvs_hConstraint constraintId(self,dic):
        hid = dic.get('id',None)
        if hid is None:
            self.constraintid += 1
            return <Slvs_hConstraint>self.constraintid
        return <Slvs_hConstraint>hid






