import math

import numpy as np

import taichi as ti

ti.init(arch=ti.gpu, debug=True)

window_size = (800, 400)
screen_res = (800, 400)
depth_res = 100
window_size_x_recpr = 1.0 / window_size[0]
window_size_y_recpr = 1.0 / window_size[1]
screen_res_x_recpr = 1.0 / screen_res[0]
screen_res_y_recpr = 1.0 / screen_res[1]
depth_res_recpr = 1.0 / depth_res
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio,
            depth_res / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1),
             round_up(boundary[1], 1),
             round_up(boundary[2], 1))

dim = 3
bg_color = 0x112f41
water_color = (116/256, 204/256, 244/256)
particle_color = (0/256, 0/256, 255/256)
boundary_color = 0xebaca2
num_particles_x = 50
num_particles_xy = num_particles_x * 20
num_particles = num_particles_xy * 10
max_num_particles_per_cell = 500
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 2
corr_deltaQ_coeff = 0.3
corrK = 0.001

neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi


### marching cubes
mc_table_list = []
cube_size_x = 0.75
cube_size_y = 0.75
cube_size_z = 0.75
cube_recpr_x = 1.0 / cube_size_x
cube_recpr_y = 1.0 / cube_size_y
cube_recpr_z = 1.0 / cube_size_z
max_num_particles_per_cube = 500
density_threshold = 0.075
max_mc_triangles = 300000
max_mc_vertices = 3 * max_mc_triangles
vertex_neighbor_cube_offset_list = [
    [[0, -1, 0], [0, -1, -1], [-1, -1, 0], [-1, -1, -1], [0, 0, 0], [0, 0, -1], [-1, 0, 0], [-1, 0, -1]],
    [[1, -1, 0], [1, -1, -1], [0, -1, 0], [0, -1, -1], [1, 0, 0], [1, 0, -1], [0, 0, 0], [0, 0, -1]],
    [[1, -1, 1], [1, -1, 0], [0, -1, 0], [0, -1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 1]],
    [[0, -1, 1], [0, -1, 0], [-1, -1, 0], [-1, -1, 1], [0, 0, 1], [0, 0, 0], [-1, 0, 0], [-1, 0, 1]],
    [[0, 0, 0], [0, 0, -1], [-1, 0, 0], [-1, 0, -1], [0, 1, 0], [0, 1, -1], [-1, 1, 0], [-1, 1, -1]],
    [[1, 0, 0], [1, 0, -1], [0, 0, 0], [0, 0, -1], [1, 1, 0], [1, 1, -1], [0, 1, 0], [0, 1, -1]],
    [[1, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
    [[0, 0, 1], [0, 0, 0], [-1, 0, 0], [-1, 0, 1], [0, 1, 1], [0, 1, 0], [-1, 1, 0], [-1, 1, 1]]
]

mc_edge2vertices_list = [[0, 1], [1, 2], [2, 3], [0, 3],
                         [4, 5], [5, 6], [6, 7], [7, 4],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

def round_up_cube(f, s, cube_recpr):
    return (math.floor(f * cube_recpr / s) + 1) * s

cube_num = (round_up_cube(boundary[0], 1, cube_recpr_x),
            round_up_cube(boundary[1], 1, cube_recpr_y),
            round_up_cube(boundary[2], 1, cube_recpr_z))

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
render_positions = ti.Vector.field(dim, float)

velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
density = ti.field(float)

board_states = ti.Vector.field(2, float)

cube_vertices_pos = ti.Vector.field(dim, float)
cube_vertices_density = ti.field(float)
cube_num_particles = ti.field(int)
cube_particles = ti.field(int)
mc_num_vertices = ti.field(int, shape=())
mc_vertices = ti.Vector.field(dim, float)
mc_id = ti.field(int)
mc_table = ti.field(int, shape=(256, 16))
mc_edge2vertices = ti.field(int, shape=(12, 2))
vertex_neighbor_cube_offset = ti.Vector.field(dim, int)



ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities, render_positions)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas, density)
ti.root.place(board_states)



cube_snode = ti.root.dense(ti.ijk, cube_num)
cube_snode.place(mc_id, cube_num_particles)
cube_snode.dense(ti.l, 8).place(cube_vertices_pos, cube_vertices_density)
cube_snode.dense(ti.l, max_num_particles_per_cube).place(cube_particles)
ti.root.dense(ti.i, 8).dense(ti.j, 8).place(vertex_neighbor_cube_offset)
ti.root.dense(ti.i, max_mc_vertices).place(mc_vertices)

@ti.func
def get_cube(pos):
    pos.x *= cube_recpr_x
    pos.y *= cube_recpr_y
    pos.z *= cube_recpr_z
    return int(pos)

@ti.func
def is_in_cube(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < cube_num[0] and 0 <= c[1] and c[
        1] < cube_num[1] and 0 <= c[2] and c[2] < cube_num[2]

###

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_,
                                                     h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]

@ti.func
def confine_position_to_boundary(p, v):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1], boundary[2]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin - epsilon * v[i]
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * v[i]
    return p

@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 6.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b

@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8, 0.0])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos, vel)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0
        density[p_i] = density_constraint

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos, vel = positions[i], velocities[i]
        positions[i] = confine_position_to_boundary(pos, vel)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...

def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()

@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02,
                          (boundary[2] - delta * (num_particles // num_particles_xy)) * 0.5])
        positions[i] = ti.Vector([i % num_particles_x,
                                  (i % num_particles_xy) // num_particles_x,
                                  i // num_particles_xy]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

@ti.func
def normalize_to_render(pos):
    pos.x *= screen_to_world_ratio * screen_res_x_recpr
    pos.y *= screen_to_world_ratio * screen_res_y_recpr
    pos.z *= screen_to_world_ratio * depth_res_recpr

    return pos
#
#
@ti.func
def compute_cube_vertices(cube):
    for i in ti.ndrange(4):
        cube_vertices_pos[cube, i][0] = cube[0] * cube_size_x
        cube_vertices_pos[cube, i][1] = cube[1] * cube_size_y
        cube_vertices_pos[cube, i][2] = cube[2] * cube_size_z
    cube_vertices_pos[cube, 1][0] += cube_size_x
    cube_vertices_pos[cube, 2][0] += cube_size_x
    cube_vertices_pos[cube, 2][2] += cube_size_z
    cube_vertices_pos[cube, 3][2] += cube_size_z
    for i in ti.ndrange((4, 8)):
        cube_vertices_pos[cube, i][0] = cube_vertices_pos[cube, i - 4][0]
        cube_vertices_pos[cube, i][1] = cube_vertices_pos[cube, i - 4][1] + cube_size_y
        cube_vertices_pos[cube, i][2] = cube_vertices_pos[cube, i - 4][2]

@ti.func
def find_particles_in_all_cubes():
    for I in ti.grouped(cube_num_particles):
        cube_num_particles[I] = 0

    for p_i in positions:
        pos = positions[p_i]
        cube = get_cube(pos)
        offs = ti.atomic_add(cube_num_particles[cube], 1)
        cube_particles[cube, offs] = p_i

        # # one particle exists in many cubes around it
        # for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
        #     cube_to_check = cube + offs
        #     if is_in_grid(cube_to_check):
        #         num = ti.atomic_add(cube_num_particles[cube_to_check], 1)
        #         cube_particles[cube, num] = p_i

@ti.kernel
def marching_cube():
    # print("bbb")
    find_particles_in_all_cubes()

    mc_num_vertices[None] = 0
    for cube in ti.grouped(ti.ndrange((0, cube_num[0]), (0, cube_num[1]), (0, cube_num[2]))):
        # pass
        # print(cube)
        # compute positions of cube's 8 vertices
        compute_cube_vertices(cube)

        # compute density of each vertex
        for i in ti.static(range(8)):
            cube_vertices_density[cube, i] = 0
            pos_vertex_i = cube_vertices_pos[cube, i]
            for neighbor in ti.static(ti.ndrange(8)):
            # for offs in ti.static(ti.grouped(ti.ndrange((0, 1), (0, 1), (0, 1)))):
                offs = vertex_neighbor_cube_offset[i, neighbor]
                cube_to_check = cube + offs
                if is_in_cube(cube_to_check):
                    for j in range(cube_num_particles[cube_to_check]):
                        p_j = cube_particles[cube_to_check, j]
                        vertex_i_p_j_norm = (pos_vertex_i - positions[p_j]).norm()
                        if vertex_i_p_j_norm < 0.5 * h_:
                            cube_vertices_density[cube, i] += density[p_j] * poly6_value(vertex_i_p_j_norm, h_)

        mc_id[cube] = 0
        if cube_vertices_density[cube, 0] < density_threshold:
            mc_id[cube] |= 1
        if cube_vertices_density[cube, 1] < density_threshold:
            mc_id[cube] |= 2
        if cube_vertices_density[cube, 2] < density_threshold:
            mc_id[cube] |= 4
        if cube_vertices_density[cube, 3] < density_threshold:
            mc_id[cube] |= 8
        if cube_vertices_density[cube, 4] < density_threshold:
            mc_id[cube] |= 16
        if cube_vertices_density[cube, 5] < density_threshold:
            mc_id[cube] |= 32
        if cube_vertices_density[cube, 6] < density_threshold:
            mc_id[cube] |= 64
        if cube_vertices_density[cube, 7] < density_threshold:
            mc_id[cube] |= 128

        if mc_id[cube] == 0 or mc_id[cube] == 255:
            continue
        # print(mc_id[cube])

        j = 0
        offs = 0
        for k in ti.ndrange(16):
            edge = mc_table[mc_id[cube], k]
            if edge == -1:
                continue
            v0 = mc_edge2vertices[edge, 0]
            v1 = mc_edge2vertices[edge, 1]
            pos_v0 = cube_vertices_pos[cube, v0]
            pos_v1 = cube_vertices_pos[cube, v1]
            density_v0 = cube_vertices_density[cube, v0] - density_threshold
            density_v1 = cube_vertices_density[cube, v1] - density_threshold
            sum_recpr = 1 / (density_v1 - density_v0)
            pos_interp = pos_v0 + (pos_v1 - pos_v0) * sum_recpr * density_v1
            # pos_interp = (pos_v1 + pos_v0) / 2
            if (j == 0):
                offs = ti.atomic_add(mc_num_vertices[None], 3)
            mc_vertices[offs + j] = normalize_to_render(pos_interp)
            j = (j + 1) % 3

def read_mc_table():
    with open('mc_table.txt') as f:
        lines = f.readlines()

    i = 0
    for line in lines:
        line = line.split(',\n')[0][1:-1]
        if (line[-1] == '}'):
            line = line[:-1]
        # print(line)
        l = []
        j = 0
        for s in line.split(','):
            l.append(int(s))
            mc_table[i, j] = int(s)
            j += 1
        mc_table_list.append(l)
        i += 1

    for i in range(12):
        for j in range(2):
            mc_edge2vertices[i, j] = mc_edge2vertices_list[i][j]

def initialize_vertex_neighbor_cube_offset():
    for i in range(8):
        for j in range(8):
            for k in range(3):
                vertex_neighbor_cube_offset[i, j][k] = vertex_neighbor_cube_offset_list[i][j][k]


@ti.kernel
def normalize_positions():
    for i in positions:
        render_positions[i] = normalize_to_render(positions[i])

def main():
    result_dir = "./results"
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    read_mc_table()
    initialize_vertex_neighbor_cube_offset()
    # print(mc_table)
    # print(mc_edge2vertices)

    window = ti.ui.Window('PBF3D', window_size,
                          vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    camera.position(0.4, 0.5, 1.8)
    # camera.position(0.4, 0.5, 1.8)
    camera.lookat(0.4, 0.15, 0.5)
    camera.up(0, 1, 0)
    # camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((1, 1, 1))
    # scene.ambient_light((0, 0, 0))

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.ui.ESCAPE]:
                break
        move_board()
        run_pbf()
        # if gui.frame % 20 == 1:
        #     print_stats()
        # print(render_positions[4])
        normalize_positions()
        particle_radius_render = particle_radius_in_world * screen_to_world_ratio / screen_res[0]
        scene.particles(render_positions, radius=particle_radius_render, color=water_color)
        # print("aaa")
        # marching_cube()
        # scene.point_light(pos=(0.4, 0.5, -1.8), color=(1, 1, 1))
        # scene.point_light(pos=(-0.4, 1.5, 0), color=(1, 1, 1))
        # scene.point_light(pos=(0.8, 10.5, 0), color=(1, 1, 1))
        # scene.mesh(mc_vertices, vertex_count=mc_num_vertices[None], two_sided=True, color=water_color)

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()


