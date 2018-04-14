import slvs
from slvs import *

s = slvs.Slvs()
s.set_group(1)

# First, we create our workplane. Its origin corresponds to the origin
# of our base frame (x y z) = (0 0 0)
s.MakePoint3d(s.P(0.0), s.P(0.0), s.P(0.0), id = 101);
# and it is parallel to the xy plane, so it has basis vectors (1 0 0)
# and (0 1 0).
(qw,qx,qy,qz) = s.MakeQuaternion(1, 0, 0,
                                0, 1, 0);
s.MakeParam(qw, id = 4);
s.MakeParam(qx, id = 5);
s.MakeParam(qy, id = 6);
s.MakeParam(qz, id = 7);
s.MakeNormal3d(4, 5, 6, 7, id = 102);

s.MakeWorkplane(101, 102, id = 200);

# Now create a second group. We'll solve group 2, while leaving group 1
# constant; so the workplane that we've created will be locked down,
# and the solver can't move it.

s.set_group(2)

# These points are represented by their coordinates (u v) within the
# workplane, so they need only two parameters each.

s.MakeParam(10.0, id = 11);
s.MakeParam(20.0, id = 12);
s.MakePoint2d(200, 11, 12, id = 301);

s.MakeParam(20.0, id = 13);
s.MakeParam(10.0, id = 14);
s.MakePoint2d(200, 13, 14, id = 302);

# And we create a line segment with those endpoints.
s.MakeLineSegment(200, 301, 302, id = 400);

# Now three more points.
s.MakeParam(100.0, id = 15);
s.MakeParam(120.0, id = 16);
s.MakePoint2d(200, 15, 16, id = 303);

s.MakeParam(120.0, id = 17);
s.MakeParam(110.0, id = 18);
s.MakePoint2d(200, 17, 18, id = 304);

s.MakeParam(115.0, id = 19);
s.MakeParam(115.0, id = 20);
s.MakePoint2d(200, 19, 20, id = 305);

# And arc, centered at point 303, starting at point 304, ending at
# point 305.
s.MakeArcOfCircle(200, 102, 303, 304, 305, id = 401);

# Now one more point, and a distance
s.MakeParam(200.0, id = 21);
s.MakeParam(200.0, id = 22);
s.MakePoint2d(200, 21, 22, id = 306);

s.MakeParam(30.0, id = 23);
s.MakeDistance(200, 23, id = 307);

# And a complete circle, centered at point 306 with radius equal to
# distance 307. The normal is 102, the same as our workplane.
s.MakeCircle(200, 306, 102, 307, id = 402);


# The length of our line segment is 30.0 units.
s.MakeConstraint(SLVS_C_PT_PT_DISTANCE,200,30.0,301, 302, 0, 0, id = 1);

# And the distance from our line segment to the origin is 10.0 units.
s.MakeConstraint(SLVS_C_PT_LINE_DISTANCE,200,10.0,101, 0, 400, 0, id = 2);

# And the line segment is vertical.
s.MakeConstraint(SLVS_C_VERTICAL,200,0.0,0, 0, 400, 0, id = 3);

# And the distance from one endpoint to the origin is 15.0 units.
s.MakeConstraint(SLVS_C_PT_PT_DISTANCE,200,15.0,301, 101, 0, 0, id = 4);

# And same for the other endpoint; so if you add this constraint then
# the sketch is overconstrained and will signal an error.
#s.MakeConstraint(5, g,SLVS_C_PT_PT_DISTANCE,200,18.0,302, 101, 0, 0);

# The arc and the circle have equal radius.
s.MakeConstraint(SLVS_C_EQUAL_RADIUS, 200, 0.0, 0, 0, 401, 402, id =  6);

# The arc has radius 17.0 units.
s.MakeConstraint(SLVS_C_DIAMETER, 200, 17.0*2, 0, 0, 401, 0, id =  7);

# If the solver fails, then ask it to report which constraints caused
# the problem.

# And solve.
result = s.solve(2)
param = result.param

if result.status == SLVS_RESULT_OKAY:
    print("solved okay");
    print("line from (%.3f %.3f) to (%.3f %.3f)\n" % (
            param[7].val, param[8].val,
            param[9].val, param[10].val));

    print("arc center (%.3f %.3f) start (%.3f %.3f) finish (%.3f %.3f)\n" % (
            param[11].val, param[12].val,
            param[13].val, param[14].val,
            param[15].val, param[16].val));

    print("circle center (%.3f %.3f) radius %.3f\n" % (
            param[17].val, param[18].val,
            param[19].val));
    print("%d DOF\n" % result.dof);
else:
    print("solve failed: problematic constraints are:");
    for i in range(0,len(result.failed)):
        print(" %lu" % result.failed[i]);
    print();

    if result.status == SLVS_RESULT_INCONSISTENT:
        print("system inconsistent");
    else:
        print("system nonconvergent");

