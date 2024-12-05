import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import time
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython import embed

class RectBlock:
    def __init__(self, x, y, side, id = 0, tag = None):
        self.x = x
        self.y = y
        self.side = side
        self.center_x = x + side / 2
        self.center_y = y + side / 2
        self.id = id
        self.tag = tag

    def update_position(self, x, y, id = None):
        self.x = x
        self.y = y
        self.id = id if id is not None else self.id
        self.center_x = x + self.side / 2
        self.center_y = y + self.side / 2

    def get_position(self):
        return self.x, self.y

    def get_center(self):
        return self.center_x, self.center_y

def calculate_hpwl(block1, block2):
    x1, y1 = block1.get_center()
    x2, y2 = block2.get_center()
    return 0.5 * (abs(x2 - x1) + abs(y2 - y1))

def total_hpwl(blocks, connections):
    total_length = 0
    path_lengths = {}  # 创建一个字典来存储每条路径的线长

    for i, j in connections:
        length = calculate_hpwl(blocks[i], blocks[j])
        total_length += length
        path_lengths[(i, j)] = length  # 将每条路径的线长存储在字典中
    return total_length, path_lengths
       
def is_overlapping(block1, block2):
    return not (block1.x + block1.side <= block2.x or
                block1.x >= block2.x + block2.side or
                block1.y + block1.side <= block2.y or
                block1.y >= block2.y + block2.side)

def resolve_overlap(block1, block2):
    overlap_x = min(block1.x + block1.side, block2.x + block2.side) - max(block1.x, block2.x)
    overlap_y = min(block1.y + block1.side, block2.y + block2.side) - max(block1.y, block2.y)
    
    if overlap_x < overlap_y:
        if block1.center_x < block2.center_x:
            block1.update_position(block1.x - overlap_x / 2, block1.y)
            block2.update_position(block2.x + overlap_x / 2, block2.y)
        else:
            block1.update_position(block1.x + overlap_x / 2, block1.y)
            block2.update_position(block2.x - overlap_x / 2, block2.y)
    else:
        if block1.center_y < block2.center_y:
            block1.update_position(block1.x, block1.y - overlap_y / 2)
            block2.update_position(block2.x, block2.y + overlap_y / 2)
        else:
            block1.update_position(block1.x, block1.y + overlap_y / 2)
            block2.update_position(block2.x, block2.y - overlap_y / 2)


def resolve_all_overlaps(blocks, grid_size):
    """
    解决所有块之间的重叠问题，确保每个块在布局中不重叠。
    :param blocks: 所有块的列表
    """
    # 2. 创建网格进行空间分区
    grid = {}
    for i, block in enumerate(blocks):
        # 计算块所属的网格单元
        cell_x = int(block.x // grid_size)
        cell_y = int(block.y // grid_size)
        grid.setdefault((cell_x, cell_y), []).append(i)

    # 3. 只检查同一网格和相邻网格中的块
    for (cell_x, cell_y), indices in grid.items():
        for i in indices:
            for dx in (-1, 0, 1):  # 相邻网格的x方向偏移
                for dy in (-1, 0, 1):  # 相邻网格的y方向偏移
                    neighbor_cell = (cell_x + dx, cell_y + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j and is_overlapping(blocks[i], blocks[j]):
                                resolve_overlap(blocks[i], blocks[j])



def calculate_enclosing_area(blocks):
    """
    计算给定 blocks 布局的外界矩形面积。
    :param blocks: 一个包含所有块的列表，每个块具有 x, y, width, height 属性。
    :return: 外界矩形的面积。
    """
    # 获取所有块的 x 和 y 坐标边界
    min_x = min(block.x for block in blocks)
    max_x = max(block.x + block.side for block in blocks)
    min_y = min(block.y for block in blocks)
    max_y = max(block.y + block.side for block in blocks)

    # 计算外界矩形面积
    return (max_x - min_x) * (max_y - min_y)




# def plot_blocks(blocks, connections, filename):
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal')
    
#     # Plot blocks
#     for block in blocks:
#         rect = plt.Rectangle((block.x, block.y), block.side, block.side, fill=True, edgecolor='black', alpha=0.5)
#         ax.add_patch(rect)
    
#     # Plot connections
#     for (i, j) in connections:
#         x1, y1 = blocks[i].get_center()
#         x2, y2 = blocks[j].get_center()
#         ax.plot([x1, x2], [y1, y2], 'r')
#     plt.savefig(filename)
#     # plt.show()


def generate_color(total_colors, color_id, base_colors='tab10', shades_per_base=10):
    """
    生成高对比度颜色，结合离散化基础颜色和深浅变化。
    
    参数:
        total_colors (int): 需要的总颜色数。
        color_id (int): 当前颜色的索引 (从 0 到 total_colors - 1)。
        base_colors (str): 基础颜色的 colormap 名称（例如 'tab10', 'Set1' 等）。
        shades_per_base (int): 每种基础颜色的深浅变化数。
        
    返回:
        tuple: (R, G, B, A) 的颜色值。
    """
    if total_colors < 1:
        raise ValueError("total_colors must be at least 1.")
    if not (0 <= color_id < total_colors):
        raise ValueError("color_id must be in the range [0, total_colors - 1].")
    if shades_per_base < 1:
        raise ValueError("shades_per_base must be at least 1.")
    
    # 获取基础颜色的调色板
    cmap = plt.get_cmap(base_colors)
    base_color_count = cmap.N  # 调色板中的基础颜色数量
    
    # 每种基础颜色分配的总颜色数
    colors_per_base = (total_colors + base_color_count - 1) // base_color_count
    shades_per_base = min(shades_per_base, colors_per_base)  # 限制深浅变化数
    
    # 计算当前颜色对应的基础颜色和深浅等级
    base_color_id = color_id // shades_per_base
    shade_id = color_id % shades_per_base
    
    # 获取基础颜色
    base_color = cmap(int(base_color_id % base_color_count))
    
    # 调整基础颜色的亮度
    if shades_per_base > 1:
        factor = 1.0 - (shade_id / (shades_per_base - 1)) * 0.5  # 调整亮度，范围为 [1.0, 0.5]
    else:
        factor = 1.0  # 无深浅变化时，直接使用基础颜色
    
    adjusted_color = tuple(factor * channel for channel in base_color[:3]) + (1.0,)  # 保持透明度为 1.0
    
    return adjusted_color



def plot_blocks(blocks, connections, layer_num = 0, filename = "./layout.png"):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # 1. Plot blocks
    for block in blocks:
        color = generate_color(layer_num, block.tag)
        rect = plt.Rectangle((block.x, block.y), block.side, block.side, fill=True, facecolor=color, edgecolor='black', alpha=0.5)
        ax.add_patch(rect)

    # 2. Plot connections with improved appearance
    for (i, j) in connections:
        x1, y1 = blocks[i].get_center()
        x2, y2 = blocks[j].get_center()
        ax.plot(
            [x1, x2], [y1, y2], 
            color='blue', linestyle='-', linewidth=0.6, alpha=0.8 
        )

    # 3. Calculate and draw the minimal bounding rectangle
    all_x = [block.x for block in blocks] + [block.x + block.side for block in blocks]
    all_y = [block.y for block in blocks] + [block.y + block.side for block in blocks]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    bounding_rect = plt.Rectangle(
        (min_x, min_y), max_x - min_x, max_y - min_y, 
        fill=False, edgecolor='darkgreen', linewidth=2, linestyle='-', label='Bounding Box' 
    )
    ax.add_patch(bounding_rect)

    # 4. Set limits to match the bounding rectangle
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # 5. Remove axes and ticks
    ax.axis('off')

    # 6. Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def update_positions_and_resolve_overlap(blocks, forces, grid_size=100):
    """
    更新块位置并使用网格分区优化重叠检测。
    :param blocks: 所有块的列表
    :param forces: 每个块的力数组 (N x 2)
    :param grid_size: 网格的单元大小，用于空间分区
    """
    # 1. 更新块的位置
    for i, block in enumerate(blocks):
        new_x = block.x + forces[i, 0]
        new_y = block.y + forces[i, 1]
        block.update_position(new_x, new_y)

    # 2. 创建网格进行空间分区
    resolve_all_overlaps(blocks, grid_size)

def improved_smooth_boundary_adjustment(blocks, grid_size, spacing=100, iterations=1):
    """
    改进的边界平滑调整函数，通过成组调整块的位置来减少外接矩形的面积。
    """
    prev_area = calculate_enclosing_area(blocks)
    # print(f"Initial enclosing area: {prev_area}")
    
    for iteration in range(iterations):
        blocks_old = copy.deepcopy(blocks)  # 深拷贝保存原有的块状态

        # 获取所有块的 x 和 y 坐标边界
        min_x = min(block.x for block in blocks)
        max_x = max(block.x + block.side for block in blocks)
        min_y = min(block.y for block in blocks)
        max_y = max(block.y + block.side for block in blocks)

        # 获取边界块
        left_boundary_blocks = [block for block in blocks if block.x == min_x]
        right_boundary_blocks = [block for block in blocks if block.x + block.side == max_x]
        top_boundary_blocks = [block for block in blocks if block.y + block.side == max_y]
        bottom_boundary_blocks = [block for block in blocks if block.y == min_y]

        # 对每个边界执行内缩操作
        shrink_amount = spacing * 0.05  # 每次迭代的内缩量

        # 左边界内缩
        for block in left_boundary_blocks:
            block.update_position(block.x + shrink_amount, block.y)
        # 右边界内缩
        for block in right_boundary_blocks:
            block.update_position(block.x - shrink_amount, block.y)
        # 顶部边界内缩
        for block in top_boundary_blocks:
            block.update_position(block.x, block.y - shrink_amount)
        # 底部边界内缩
        for block in bottom_boundary_blocks:
            block.update_position(block.x, block.y + shrink_amount)

        # 解决重叠问题
        resolve_all_overlaps(blocks, grid_size=spacing)

        # 计算新的外接矩形面积
        new_area = calculate_enclosing_area(blocks)
        # print(f"Iteration {iteration + 1}, new enclosing area: {new_area}")

        # 如果面积不再减少，则停止迭代
        if new_area >= prev_area:
            blocks = blocks_old  # 恢复到上一次的状态
            break

        prev_area = new_area

def improved_force_directed_layout(blocks, connections, grid_size, topology="mesh", spacing=100, iterations=1000, initial_learning_rate=0.1, tolerance=1):
    """
    改进的力导向布局算法，增加边界平滑和均匀化调整。
    """
    no_change_counter = 0
    for iteration in range(iterations):
        learning_rate = initial_learning_rate * (1 - iteration / iterations)
        block_centers = [block.get_center() for block in blocks]
        forces = np.zeros((len(blocks), 2), dtype=np.float32)

        # 计算连接块之间的力以减少 HPWL
        current_hpwl = 0
        for (i, j) in connections:
            x1, y1 = block_centers[i]
            x2, y2 = block_centers[j]
            force_x = (x2 - x1) * learning_rate
            force_y = (y2 - y1) * learning_rate
            forces[i, 0] += force_x
            forces[i, 1] += force_y
            forces[j, 0] -= force_x
            forces[j, 1] -= force_y
            current_hpwl += calculate_hpwl(blocks[i], blocks[j])

        # 更新位置并解决重叠
        update_positions_and_resolve_overlap(blocks, forces, grid_size=spacing)

        # 每次迭代后进行边界平滑调整
        improved_smooth_boundary_adjustment(blocks, grid_size, spacing=spacing, iterations=1)
        
        # 再次解决重叠问题
        resolve_all_overlaps(blocks, grid_size=spacing)
        
        # 检查收敛条件
        if iteration > 1 and abs(prev_hpwl - current_hpwl) < tolerance:
            no_change_counter += 1
            if no_change_counter >= 10:  # 如果连续10次变化都很小，则认为收敛
                print(f"Converged at iteration {iteration}")
                break
        else:
            no_change_counter = 0  # 如果有明显变化，重置计数器
        prev_hpwl = current_hpwl

def post_adjustment(blocks, grid_size, spacing=1, iterations=1):
    """
    后置调整函数，将突出块移动至临近边缘的凹陷处，减少布局不均衡现象，并使外接矩形趋于正方形。
    """
    prev_area = calculate_enclosing_area(blocks)
    spacing = spacing / 10

    for iteration in range(iterations):
        blocks_old = copy.deepcopy(blocks)  # 深拷贝保存原有的块状态

        # 找出与外接矩形相接的边界块
        min_x = min(block.x for block in blocks)
        max_x = max(block.x + block.side for block in blocks)
        min_y = min(block.y for block in blocks)
        max_y = max(block.y + block.side for block in blocks)

        left_boundary_blocks = [block for block in blocks if block.x == min_x]
        right_boundary_blocks = [block for block in blocks if block.x + block.side == max_x]
        top_boundary_blocks = [block for block in blocks if block.y + block.side == max_y]
        bottom_boundary_blocks = [block for block in blocks if block.y == min_y]


        # 定义突出块查找函数
        def find_protrusion_blocks(boundary_blocks, boundary_type):
            if boundary_type == 'left':
                return [block for block in boundary_blocks if block.x == min(block.x for block in boundary_blocks)]
            elif boundary_type == 'right':
                return [block for block in boundary_blocks if block.x + block.side == max(block.x + block.side for block in boundary_blocks)]
            elif boundary_type == 'top':
                return [block for block in boundary_blocks if block.y + block.side == max(block.y + block.side for block in boundary_blocks)]
            elif boundary_type == 'bottom':
                return [block for block in boundary_blocks if block.y == min(block.y for block in boundary_blocks)]
            return []

        # 分别对每个边界执行调整操作
        for boundary_blocks, boundary_type in [(left_boundary_blocks, 'left'), (right_boundary_blocks, 'right'), (top_boundary_blocks, 'top'), (bottom_boundary_blocks, 'bottom')]:
            if not boundary_blocks:
                continue

            # 查找突出块
            # protrusion_blocks = find_protrusion_blocks(boundary_blocks, boundary_type)

            # 如果突出块数量少于3块，则尝试滑移并收缩
            if len(boundary_blocks) < 3:
                for protrusion in boundary_blocks:
                    # print(protrusion.side, protrusion.x, protrusion.y)
                    
                    found_position = False
                    original_position = (protrusion.x, protrusion.y)

                    # 尝试沿所在边的两个方向滑移
                    if boundary_type in ['left', 'right']:
                        directions = [(0, 1), (0, -1)]  # 上下滑移方向
                    else:
                        directions = [(1, 0), (-1, 0)]  # 左右滑移方向

                    for dx, dy in directions:
                        for step in range(1, grid_size):
                            new_x = protrusion.x + dx * step * spacing
                            new_y = protrusion.y + dy * step * spacing
                            if (new_x >= min_x and new_x + protrusion.side <= max_x and new_y >= min_y and new_y <= max_y + protrusion.side):
                                protrusion.update_position(new_x, new_y)
                            else:
                                break

                            # 检查是否有重叠
                            if not any(is_overlapping(protrusion, other) for other in blocks if other != protrusion):
                                found_position = True
                                break

                        if found_position:
                            break

                    # 如果找到了可行的位置，则向里收缩直到紧挨其他块
                    if found_position:
                        if boundary_type in ['left', 'right']:
                            while True:
                                new_x = protrusion.x - 1 if boundary_type == 'right' else protrusion.x + 1
                                protrusion.update_position(new_x, protrusion.y)
                                if any(is_overlapping(protrusion, other) for other in blocks if other != protrusion):
                                    # 回到不重叠的位置
                                    protrusion.update_position(protrusion.x + 1 if boundary_type == 'right' else protrusion.x - 1, protrusion.y)
                                    break
                        else:
                            while True:
                                new_y = protrusion.y - 1 if boundary_type == 'top' else protrusion.y + 1
                                protrusion.update_position(protrusion.x, new_y)
                                if any(is_overlapping(protrusion, other) for other in blocks if other != protrusion):
                                    # 回到不重叠的位置
                                    protrusion.update_position(protrusion.x, protrusion.y + 1 if boundary_type == 'top' else protrusion.y - 1)
                                    break
                    else:
                        # 如果找不到可行的位置，回滚到初始位置
                        protrusion.update_position(*original_position)

        # 解决可能的重叠问题
        resolve_all_overlaps(blocks, grid_size=spacing)

        # 计算新的外接矩形面积
        new_area = calculate_enclosing_area(blocks)

        # 如果面积不再减少，则停止迭代
        if new_area >= prev_area:
            blocks = blocks_old  # 恢复到上一次的状态
            break

        prev_area = new_area


def floorplan(grid_size, topology, area, tag, layer_num = 0, iterations=2000, tolerance=1e-4):
    '''
    grid_size: num of topology nodes for one side
    topology: mesh or cmesh
    area: array for nodes
    tag: layer name for each node
    layer_num: total layer num
    iterations: num of iterations
    tolerance: for early termination
    '''
    blocks = []
    sides = np.sqrt(area)
    
    spacing = 1.5 * np.max(sides)

    if topology == "mesh":
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * spacing
                y = i * spacing
                id = i*grid_size + j
                blocks.append(RectBlock(x, y, sides[i * grid_size + j], id=id, tag=tag[i * grid_size + j]))

        # Define 2D mesh connections between blocks
        connections = []
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if j < grid_size - 1:  # Connect to right neighbor
                    connections.append((idx, idx + 1))
                if i < grid_size - 1:  # Connect to bottom neighbor
                    connections.append((idx, idx + grid_size))
    elif topology == "cmesh":
        spacing = 2 * spacing
        # Define regular blocks
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * spacing
                y = i * spacing
                id = i*grid_size + j
                blocks.append(RectBlock(x, y, sides[i * grid_size + j], id=id, tag=tag[i * grid_size + j]))
        
        # Insert IO nodes in the center of quadrants
        io_start_index = len(blocks)  # Index where IO blocks start
        for i in range(grid_size // 2):
            for j in range(grid_size // 2):
                x = (2 * j + 0.5) * spacing
                y = (2 * i + 0.5) * spacing
                id = grid_size*grid_size + i*grid_size//2 + j - 1
                blocks.append(RectBlock(x, y, sides[grid_size * grid_size + i * grid_size // 2 + j - 1], id=id, tag=tag[grid_size * grid_size + i * grid_size // 2 + j - 1]))


        # Define Cmesh connections between blocks and IO nodes
        connections = []
        # Connect IO nodes to their neighboring blocks and other IO nodes
        io_count = grid_size // 2 * grid_size // 2
        for idx in range(io_start_index, io_start_index + io_count):
            # Determine the position of the IO node in the grid
            local_idx = idx - io_start_index
            i = local_idx // (grid_size // 2)
            j = local_idx % (grid_size // 2)

            # Top-left quadrant IO connections (example: adjust accordingly)
            # Connect each IO to the surrounding regular blocks
            connections.append((idx, 2 * i * grid_size + 2 * j))  # Connect to C0
            connections.append((idx, 2 * i * grid_size + 2 * j + 1))  # Connect to C1
            connections.append((idx, 2 * i * grid_size + 2 * j + grid_size))  # Connect to C2
            connections.append((idx, 2 * i * grid_size + 2 * j + grid_size + 1))  # Connect to C3

        # Connect IO nodes to other IO nodes as per the mesh structure
        for i in range(grid_size // 2):
            for j in range(grid_size // 2):
                idx = io_start_index + i * (grid_size // 2) + j
                # Connect to right IO neighbor if it exists
                if j < (grid_size // 2) - 1:
                    connections.append((idx, idx + 1))
                # Connect to bottom IO neighbor if it exists
                if i < (grid_size // 2) - 1:
                    connections.append((idx, idx + (grid_size // 2)))

    # Plot the resulting block layout
    filename = f"./forcedir_{topology}{grid_size}.svg"
    plot_blocks(blocks, connections, layer_num=layer_num, filename=filename)

    initial_hpwl, _ = total_hpwl(blocks, connections)
    initial_area_enclosing_rectangle = calculate_enclosing_area(blocks)
    
    print(f'Initial Total HPWL: {initial_hpwl}')
    print(f"Initial Total Area: {initial_area_enclosing_rectangle}")
    
    improved_force_directed_layout(blocks, connections, grid_size, topology=topology, spacing=spacing, iterations=iterations, tolerance=tolerance)
    #post_adjustment(blocks, grid_size, spacing=spacing, iterations=5)

    opt_hpwl, opt_path_lengths = total_hpwl(blocks, connections)
    Area_enclosing_rectangle = calculate_enclosing_area(blocks)
    util = sum(area)/Area_enclosing_rectangle
    print(f"Area utilization: {util}")
    print(f'Final Total HPWL: {opt_hpwl}')
    print(f'Area_enclosing_rectangle: {Area_enclosing_rectangle}')

    # Plot the resulting block layout
    filename = f"./forcedir_opt_{topology}{grid_size}.svg"
    plot_blocks(blocks, connections, layer_num, filename)

    return Area_enclosing_rectangle, opt_hpwl, opt_path_lengths

