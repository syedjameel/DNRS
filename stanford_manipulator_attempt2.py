# Written by Jameel Ahmed Syed
# Email id: j.syed@innopolis.university

# Assignment 1
# Dynamics of Non-Linear Robotic Systems
# Forward and Inverse Kinematics of Stanford Manipulator with the spherical arm

from sympy import Matrix
from numpy import pi
import math


def rotz(t):
    """Rotates the frame about Z Axis"""
    rot = Matrix([[cos(t), -sin(t), 0, 0],
                  [sin(t), cos(t),  0, 0],
                  [0,       0,      1, 0],
                  [0,       0,      0, 1]])
    return rot

def rotx(t):
    """Rotates the frame about X Axis"""
    rot = Matrix([[1,       0,      0, 0],
                  [0, cos(t), -sin(t), 0],
                  [0, sin(t), cos(t),  0],
                  [0,   0,      0,     1]])
    return rot


def roty(t):
    """Rotates the frame about Y Axis"""
    rot = Matrix([[cos(t),  0, sin(t), 0],
                  [0,       1,   0,    0],
                  [-sin(t), 0, cos(t), 0],
                  [0,       0,   0,    1]])
    return rot


def tranx(d):
    """Translates the frame on X Axis"""
    transx = Matrix([[1, 0, 0, d],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    return transx

def trany(d):
    """Translates the frame on Y Axis"""
    transy = Matrix([[1, 0, 0, 0],
                     [0, 1, 0, d],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    return transy

def tranz(d):
    """Translates the frame on Z Axis"""
    transz = Matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, d],
                     [0, 0, 0, 1]])
    return transz

def print_matrix(input_matrix, name_matrix='T'):
    """Prints a Matrix in a clear Matrix format
    Just a little fancy stuff to make the matrix printing readable"""
    a = np.array(input_matrix)
    print(name_matrix,'=')
    s = [[str(e) for e in row] for row in a]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table), '\n')


# noinspection PyShadowingNames
def forward_kinematics(t1, t2, t3, t4, t5, t6, d1, d2, d3, d4, d5, d6):
    """Forward Kinematics Solution For Stanford Manipulator
    with Spherical Wrist in Euler's ZXZ arrangement"""
    t01 = rotz(t1)                          # Rotation around Z axis
    t12 = rotx(t2) * tranx(d2)              # Rotation around Z axis and Translation along X axis
    t23 = tranz(d3)                         # Translation along Z axis
    t34 = rotz(t4)                          # Rotation around Z axis
    t45 = rotx(t5)                          # Rotation around X axis
    t56 = rotz(t6) * tranz(d6)              # Rotation around Z axis and Translation along Z axis
    t03 = t01 * t12 * t23                   # Transformation from 0 to 3
    print_matrix(t03, name_matrix='T03')
    t36 = t34 * t45 * t56                   # Transformation from 3 to 6
    t06 = t03 * t36                         # Transformation from 0 to 6 (Complete Forward Kinematic Solution)
    t06md = t01 * t12 * t23 * t34 * t45 * rotz(t6)  # Transformation from 0 to Wrist by eliminating the d6 translation
    print_matrix(t06, name_matrix='T06')
    X_p = t06[0, 3]
    Y_p = t06[1, 3]
    Z_p = t06[2, 3]
    return t01, t12, t23, t34, t45, t56, t03, t36, t06, t06md, X_p, Y_p, Z_p

def arbitrary_calculations(t03, d6, t36, t06, t06md):
    """Some arbitrary calculations for Inverse Kinematics Transformations
    For Stanford Manipulator with Spherical Wrist in Euler's ZXZ arrangement"""
    p06 = t06[0:3, 3]                       # This is the P0 to wrist position using Pipers Method
    p0w = p06 - d6 * t06[0:3, 2]
    print_matrix(p0w, 'P0w')

    p06_md = t06md[0:3, 3]
    p0w_md = p06_md                         # This is the P0 to wrist without the d6 Translation without using Pipers method
    print_matrix(p0w_md, 'P0w_md')

    p03 = t03[0:3, 3]                       # Translation Matrix(P03) extracted from T03
    print_matrix(p03, 'P03')

    r03 = t03[0:3, 0:3]                     # Rotation Matrix(R03) extracted from T03
    print_matrix(r03, 'R03')

    r06 = t06[0:3, 0:3]                     # Rotation Matrix(R06) extracted from T06
    print_matrix(r06, 'R06')

    r36 = t36[0:3, 0:3]                     # Rotation Matrix(R36) extracted from T36
    print_matrix(r36, 'R36')

    r03_transpose = r03.T                   # Transpose of R03
    r36 = r03_transpose * r06
    print_matrix(r36, 'R36 = R03.T * R06')
    return p0w, p0w_md, p03, r03, r03_transpose, r36

def inverse_kinematics(t06, d3, d6, t06md, t03, r03, t36, t1, t2, t4, t5, t6):
    """Forward Kinematics Solution For Stanford Manipulator
    with Spherical Wrist in Euler's ZXZ arrangement"""
    p06_sym = t06[0:3, 3]                               # Translation Matrix(P06) Symbolic extracted from T06 using Pipers Method
    p0w_sym = p06_sym - d6 * t06[0:3, 2]
    print_matrix(p0w_sym, 'P0w_Symbolic')

    p06_md_sym = t06md[0:3, 3]                          # Translation Matrix(P06) Symbolic extracted from T06 without Pipers Method
    p0w_md_sym = p06_md_sym
    print_matrix(p0w_md_sym, 'P0w_md_Symbolic')

    p03_sym = t03[0:3, 3]                               # Translation Matrix(P03) Symbolic extracted from T03
    print_matrix(p03_sym, 'P03_Symbolic')

    theta2 = solve(l1 + l3 * cos(t2) - p03[2, 0])       # First we solve for theta2
    thet2 = []
    thet2.append([i*(180/pi) for i in theta2])          # thet6 contains the Joint 2 angels in Degrees

    theta1 = []
    for i in theta2:
        th1 = solve(l2*cos(t1) + l3*sin(t1)*sin(i) - p03[0, 0])     # Solve for theta1 for each angle of theta1
        for i in range(len(th1)):
            theta1.append(th1[i])
        th1 = []
    thet1 = []
    thet1.append([i*(180/pi) for i in theta1])          # thet1 contains the Joint 1 angels in Degrees

    d3_length = []
    for i in theta2:                                    # Solve for d3 Length for each angle of theta1 and Theta2
        for j in theta1:
            d3_length.append(solve(l2*cos(j) + d3 * sin(j) * sin(i) - p03[0, 0]))


    r03_sym = t03[0:3, 0:3]                             # Rotation Matrix(R03) Symbolic extracted from T03
    # print_matrix(r03_sym, 'R03_Symbolic')

    r03_transpose = r03.T                               # Rotation Matrix(R03) Transpose

    r36_sym = t36[0:3, 0:3]                             # Rotation Matrix(R36) Symbolic extracted from T36
    print_matrix(r36_sym, 'R36_Symbolic')

    # Till here we get the t1, t2, d3 from inverse kinematics calculations

    theta4 = math.atan2(r36[0, 2], -r36[1, 2])          # Theta4 value in radians


    theta5 = math.atan2(math.sqrt(1-r36[2,2]**2), r36[2,2])

    theta6 = math.atan2(r36[2, 0], r36[2, 1])


    theta5_2 = math.acos(r36[2, 2])

    theta6_2 = solve(-sin(theta4)*sin(t6)*cos(theta5) + cos(theta4)*cos(t6) - r36[0, 0])
    thet6 = []
    thet6 = [i*(180/pi) for i in theta6_2]              # thet6 contains the Joint 6 angels in Degrees

    theta5_3 = solve(-sin(theta4)*cos(t5)*cos(theta6) - sin(theta6)*cos(theta4) - r36[0, 1])
    thet5 = []
    thet5 = [i*(180/pi) for i in theta5_3]              # thet5 contains the Joint 5 angels in Degrees

    theta4_3 = solve(-sin(theta5)*cos(t4) - r36[1, 2])
    #print(theta4_3)
    thet4 = []
    thet4 = [i*(180/pi) for i in theta4_3]              # thet4 contains the Joint 4 angels in Degrees
    return thet1, thet2, d3_length, thet4, thet5, thet6


def plot_start(fk):
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('X')  # axis label
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.scatter(t06[0,3], t06[1,3], t06[2,3])
    for k in fk:
        plot_frame(ax, k)

    xs, ys, zs = extract_plot_points(fk)
    ax.plot(xs, ys, zs, linewidth=1)

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])

    #plt.show()

def extract_plot_points(serial):
    xs, ys, zs = [], [], []
    for trans in serial:
        x, y, z = trans[0:3, 3]
        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs, ys, zs


def extract_vectors_from_trans(trans):
    x, y, z = trans[0:3, 3]
    p = [x, y, z]
    v1 = trans[0:3, 0]
    v2 = trans[0:3, 1]
    v3 = trans[0:3, 2]

    return p, [v1, v2, v3]


def plot_arrow(ax, p, v, color):
    x, y, z = p
    u, v, w = v
    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True, color=color)


def plot_frame(ax, trans):
    p, vs = extract_vectors_from_trans(trans)

    colors = ['r', 'g', 'b']

    for i in range(3):
        plot_arrow(ax, p, vs[i], colors[i])


SYMBOLIC = False        # Here we get all the numerical calculations

t1 = pi/4   # Joint Angle 1
t2 = pi/4   # Joint Angle 2
t3 = 0      # Dummy variable for just sequence sake
t4 = pi/4   # Joint Angle 4
t5 = pi/4   # Joint Angle 5
t6 = pi/4   # Joint Angle 6

d1 = 0      # Dummy variable for just sequence sake
d2 = 1      # Link 2 Length
d3 = 1      # Prismatic Joint length + the Link 3 Length
d4 = 0      # Dummy variable for just sequence sake
d5 = 0      # Dummy variable for just sequence sake
d6 = 1      # Wrist to the end effector Length

x = 0       # End-Effector position on X-axis
y = 0       # End-Effector position on Y-axis
z = 0       # End-Effector position on Z-axis


l1, l2, l3, l4, l5, l6 = d1, d2, d3, d4, d5, d6     # Storing the Link lengths for Further Calculations


if SYMBOLIC == False:

    # if SYMBOLIC == False we import numpy, math, matplotlib libraries to do
    # the calculations and get the values for our forward kinematics
    # else we import the sympy libraries to do symbolic expression Matrices

    from numpy import sin, cos, pi
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    t01, t12, t23, t34, t45, t56, t03, t36, t06, t06md, X, Y, Z = \
        forward_kinematics(t1, t2, t3, t4, t5, t6, d1, d2, d3, d4, d5,
                                                                  d6)
    # All the transformations happens in the above function "forward_kinematics()"

    X_pos = X
    Y_pos = Y
    Z_pos = Z

    p0w, p0w_md, p03, r03, r03_transpose, r36 = arbitrary_calculations(t03, d6, t36, t06, t06md)


    fk = [t01, t12, t23, t34, t45, t56]     # All the Frames

    plot_start(fk)



SYMBOLIC = True  # Here we do all the symbolic calculations

if SYMBOLIC == True:
    # Importing the libraries for symbolic calculations
    from sympy import symbols
    from sympy.solvers import solve
    from sympy import symbols, cos, sin
    from numpy import pi

    symbol_names = ['t1', 't2', 't3', 't4', 't5', 't6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6']

    # Assigning the symbols for calculating the Symbolic equations
    for name in symbol_names:
        globals()[name] = symbols(name)

    t01, t12, t23, t34, t45, t56, t03, t36, t06, t06md, X_pos, Y_pos, Z_pos = \
        forward_kinematics(t1, t2, t3, t4, t5, t6, d1, d2, d3, d4, d5,
                                                                  d6)
    # All the Symbolic transformations happens in the above function "forward_kinematics()"


    thet1, thet2, d3_length, thet4, thet5, thet6 = inverse_kinematics(t06, d3, d6, t06md, t03, r03, t36, t1, t2, t4, t5, t6)

    print("Forward Kinematics   :")
    print(f"X Position          : {X}")
    print(f"Y Position          : {Y}")
    print(f"Z Position          : {Z}")
    print("\n")
    print("Inverse Kinematics   :")
    print(f"Theta 1 Angles are  : {thet1}")
    print(f"Theta 2 Angles are  : {thet2}")
    print(f"d3 Lengths are      : {d3_length}")
    print(f"Theta 4 Angles are  : {thet4}")
    print(f"Theta 5 Angles are  : {thet5}")
    print(f"Theta 6 Angles are  : {thet6}")


#   To verify this with the text book (Siciliano 2009, Page - 77, I did the following Rzyz model, Otherwise the above one (t06) is the Rzxz model for my assignment 1 - DNRS)
#   t06zyz = t01 * roty(t2) * trany(d2) * tranz(d3) * t34 * roty(t5) * t56
#   print_matrix(t06zyz)


#t34 = Matrix([[cos(theta4), -sin(theta4), 0, 0], [sin(theta4), cos(theta4), 0, 0], [0, 0, 1, d4], [0, 0, 0, 1]])
#t45 = Matrix([[1, 0, 0, 0],[0, cos(theta5), -sin(theta5), 0],[0, sin(theta5), cos(theta5), d5],[0, 0, 0, 1]])
#t35 = t34*t45
#t56 = Matrix([[cos(theta6), -sin(theta6), 0, 0], [sin(theta6), cos(theta6), 0, 0], [0, 0, 1, d6], [0, 0, 0, 1]])
#t36 = t35*t56
#print(t36)
#rotz(t1)
#translation([d1, 0, 0])



