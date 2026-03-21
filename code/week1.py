import json
import numpy as np
# 读取json文件
with open('data(1).json', 'r', encoding='utf-8') as file:
    python_obj = json.load(file)

# 定义坐标系类
class Coordinate:
    # 定义类的属性
    def __init__(self,name,ori_axis,vectors):
        self.name = name
        self.A = np.array(ori_axis).T
        self.vectors = np.array(vectors)
        self.B = self.A.copy()

    # 判断目标坐标系能否构成坐标系
    def is_changeable(self,obj_axis):
        obj_axis = np.array(obj_axis)
        # 检查基向量与空间维度是否相等
        # 检查行列式是否为0
        det = np.linalg.det(obj_axis)
        if np.isclose(det,0):
            return False
        # 若满足条件则添加属性————目标坐标系
        self.B = np.array(obj_axis).T
        return True

    # 变换坐标系
    def change_axis(self):
        trans_matrix = np.linalg.inv(self.B) @ self.A
        self.vectors = (trans_matrix @ self.vectors.T).T

    # 求投影
    def projection(self):
        projections = []
        for v in self.vectors:
            proj = []
            for i in range(self.B.shape[1]):
                axis = self.B[:,i]
                length = np.dot(v,axis) / np.linalg.norm(axis)
                proj.append(length)
            projections.append(proj)
        return np.array(projections, dtype=float).tolist()

    # 计算角度
    def angle(self):
        angles = []
        # 计算基向量数量
        basis_count = self.B.shape[1]
        for v in self.vectors:
            ang = []
            v_norm = np.linalg.norm(v)
            # 验证夹角是否为0
            if v_norm == 0:
                # 令内层列表长度与基向量个数一致
                ang = [0.0] * basis_count
            else :
                for i in range(self.B.shape[1]):
                    axis = self.B[:, i]
                    cos_theta = np.dot(v, axis) / (np.linalg.norm(v) * np.linalg.norm(axis))
                    # 修正不符合区间的值
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta = np.arccos(cos_theta)
                    ang.append(theta)
            angles.append(ang)
        return np.array(angles, dtype=float).tolist()

    # 计算面积或体积
    def area_volume(self):
        return abs(np.linalg.det(self.B))

# 处理data中的数据
for group in python_obj:
    group_name = group['group_name']
    vectors = group['vectors']
    ori_axis = group['ori_axis']
    tasks = group['tasks']

    # 实例化每一个任务组并调用类中方法
    group = Coordinate(group_name,ori_axis,vectors)
    print(f"{group_name}:")
    for task in tasks:
        if task['type'] == 'axis_angle':
            print(f"角度:{group.angle()}")
        elif task['type'] == 'area':
            print(f"面积或体积：{group.area_volume()}")
        elif task['type'] == 'axis_projection':
            print(f"投影：{group.projection()}")
        elif task['type'] == 'change_axis':
            obj_axis = task['obj_axis']
            if group.is_changeable(obj_axis):
                group.change_axis()
            else :
                print('目标坐标系无法构成坐标系')
    print('\n')