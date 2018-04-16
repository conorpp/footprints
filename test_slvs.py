import slvs
from slvs import *
from cli import put_thing
import numpy as np
from  time import time

s = slvs.Slvs()

workplane = s.workplane
origin = s.origin

# Now create a second group. We'll solve group 2, while leaving group 1
# constant; so the workplane that we've created will be locked down,
# and the solver can't move it.


# These points are represented by their coordinates (u v) within the
# workplane, so they need only two parameters each.

p7 = s.MakeParam(10.0);
p8 = s.MakeParam(20.0);
line1_p1 = s.MakePoint2d(p7, p8);

p9 = s.MakeParam(20.0);
p10 = s.MakeParam(10.0);
line1_p2 = s.MakePoint2d(p9, p10);


# And we create a line segment with those endpoints.
line1 = s.MakeLineSegment(line1_p1, line1_p2);

# Now three more points.
p11 = s.MakeParam(100.0);
p12 = s.MakeParam(120.0);
arc_center = s.MakePoint2d(p11, p12);

p13 = s.MakeParam(120.0);
p14 = s.MakeParam(110.0);
arc_start = s.MakePoint2d(p13,p14);

p15 = s.MakeParam(115.0);
p16 = s.MakeParam(115.0);
arc_end = s.MakePoint2d(p15, p16);

# And arc, centered at point 303, starting at point 304, ending at
# point 305.
arc = s.MakeArcOfCircle(arc_center, arc_start, arc_end);

# Now one more point, and a distance
p17 = s.MakeParam(200.0);
p18 = s.MakeParam(200.0);
circle_center = s.MakePoint2d(p17, p18);

p19 = s.MakeParam(30.0);
circle_r = s.MakeDistance(p19);

# And a complete circle, centered at point 306 with radius equal to
# distance 307. 
circle = s.MakeCircle(circle_center, circle_r);


# The length of our line segment is 30.0 units.
s.MakeConstraint(SLVS_C_PT_PT_DISTANCE,30.0,line1_p1, line1_p2, 0, 0);

# And the distance from our line segment to the origin is 10.0 units.
s.MakeConstraint(SLVS_C_PT_LINE_DISTANCE,10.0,origin, 0, line1, 0);

# And the line segment is vertical.
s.MakeConstraint(SLVS_C_VERTICAL,0.0,0, 0, line1, 0, id = 3);

# And the distance from one endpoint to the origin is 15.0 units.
s.MakeConstraint(SLVS_C_PT_PT_DISTANCE,15.0,line1_p1,origin, 0, 0);

# And same for the other endpoint; so if you add this constraint then
# the sketch is overconstrained and will signal an error.
#s.MakeConstraint(SLVS_C_PT_PT_DISTANCE,workplane,18.0,302, 101, 0, 0, id =5 );

# The arc and the circle have equal radius.
s.MakeConstraint(SLVS_C_EQUAL_RADIUS, 0.0, 0, 0, arc, circle);

# The arc has radius 17.0 units.
s.MakeConstraint(SLVS_C_DIAMETER, 17.0*2, 0, 0, arc, 0);

# If the solver fails, then ask it to report which constraints caused
# the problem.

# And solve.
t1 = time() * 1000000
result = s.solve()
t2 = time() * 1000000
print('time: %d us' % (t2-t1))

if result.status == SLVS_RESULT_OKAY:
    print("solved okay");
    print("line from (%.3f %.3f) to (%.3f %.3f)\n" % (
            p7.val, p8.val,
            p9.val, p10.val));

    print("arc center (%.3f %.3f) start (%.3f %.3f) finish (%.3f %.3f)\n" % (
            p11.val, p12.val,
            p13.val, p14.val,
            p15.val, p16.val));

    print("circle center (%.3f %.3f) radius %.3f\n" % (
            p17.val, p18.val,
            p19.val));
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

