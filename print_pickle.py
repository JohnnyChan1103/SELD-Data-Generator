import pickle
import csv
from pathlib import Path
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_vocabulary(vocab_path):
    """读取 vocabulary.csv 并生成 mid 到 label 的映射"""
    mid_to_label = {}
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        print(f"警告: 未找到词汇表文件 {vocab_path}，将无法翻译标签。")
        return mid_to_label
    
    try:
        with open(vocab_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # 假设格式是: index, label, mid
                if len(row) >= 3:
                    index, label, mid = row[0], row[1], row[2]
                    mid_to_label[mid] = label
    except Exception as e:
        print(f"解析词汇表时出错: {e}")
    
    return mid_to_label

def visualize_spatial_layout(srir_data, mixture_data, mix_idx, mid_to_label):
    """
    支持动态轨迹的房间布局可视化：俯视图 (XY) 和 侧视图 (XZ)
    """
    if 'target_classes' not in srir_data:
        return

    # 提取空间数据
    setup = srir_data['target_classes'][mix_idx]
    room_dim = setup['room_size']  # [L, W, H]
    mic_pos_center = setup['mic_pos_center']  # [x, y, z]
    all_mics = np.array(setup['mic_pos'])
    target_srcs = setup['src_pos']  # 可能包含 [x,y,z] 或 (N, 3) 的轨迹
    rt60 = setup.get('rt60', 'N/A')

    # 提取标签数据
    mix_info = mixture_data['target_classes'][mix_idx]
    target_labels = []
    for m_str in mix_info['mid']:
        # 处理可能的列表或逗号分隔字符串
        mid_key = m_str.split(',')[0].strip() if isinstance(m_str, str) else str(m_str)
        label = mid_to_label.get(mid_key, "Unknown")
        target_labels.append(label)

    # 提取干扰源数据
    interf_srcs = []
    if 'interf_classes' in srir_data and len(srir_data['interf_classes']) > mix_idx:
        interf_srcs = srir_data['interf_classes'][mix_idx].get('src_pos', [])

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    title_suffix = " (Moving Sources Enabled)" if any(np.ndim(p) > 1 for p in target_srcs) else ""
    fig.suptitle(f'Spatial Layout - Mix: {mix_idx} | RT60: {rt60}s{title_suffix}', fontsize=16)

    def draw_source(ax, pos, index, label, color, is_target=True):
        marker = 'o' if is_target else '^'
        prefix = 'T' if is_target else 'I'
        
        # 判断是动态轨迹还是静态点
        if np.ndim(pos) > 1:  # 形状为 (N, 3)
            # 1. 绘制完整轨迹线
            ax.plot(pos[:, 0], pos[:, 1 if ax == ax1 else 2], 
                    color=color, alpha=0.4, linestyle='--', linewidth=1.5)
            
            # 2. 标记起点 (用较小的正方形)
            ax.scatter(pos[0, 0], pos[0, 1 if ax == ax1 else 2], 
                       marker='s', s=30, color=color, edgecolors='black')
            
            # 3. 标记终点 (用带箭头的标记)
            ax.scatter(pos[-1, 0], pos[-1, 1 if ax == ax1 else 2], 
                       marker='>', s=80, color=color, edgecolors='black')
            
            # 4. 在起点附近写标签
            text_pos = pos[0]
            ax.text(text_pos[0]+0.1, text_pos[1 if ax == ax1 else 2]+0.1, 
                    f"{prefix}{index}:{label}", fontsize=8, color=color, weight='bold')
        else:  # 静态点 (3,)
            ax.scatter(pos[0], pos[1 if ax == ax1 else 2], marker=marker, s=100, color=color, alpha=0.7)
            ax.text(pos[0]+0.1, pos[1 if ax == ax1 else 2]+0.1, 
                    f"{prefix}{index}:{label}", fontsize=8, color=color)

    # --- 渲染逻辑 ---
    for ax, dims in zip([ax1, ax2], [(0, 1), (0, 2)]): # XY轴 和 XZ轴
        d1, d2 = dims
        axis_names = ("Length (x)", "Width (y)" if d2==1 else "Height (z)")
        
        ax.set_title(f"{'Top' if d2==1 else 'Side'} View ({axis_names[0]} vs {axis_names[1]})")
        
        # 画房间边界
        ax.add_patch(patches.Rectangle((0, 0), room_dim[0], room_dim[d2], fill=False, color='black', lw=2, ls='--'))
        
        # 画麦克风中心和各个阵元
        # ax.scatter(all_mics[:, d1], all_mics[:, d2], marker='.', s=20, color='red', alpha=0.5)
        # ax.scatter(mic_pos_center[d1], mic_pos_center[d2], marker='X', s=150, color='red', label='Mic Center', zorder=10)
        ax.scatter(all_mics[:, d1], all_mics[:, d2], 
                    marker='o', s=40, color='red', edgecolors='black', label='Individual Mics', zorder=5)
        ax.scatter(mic_pos_center[d1], mic_pos_center[d2], marker='X', s=200, color='red', label='Microphone Center')

        # 画目标声源 (蓝色系)
        for i, pos in enumerate(target_srcs):
            draw_source(ax, pos, i, target_labels[i], color='blue', is_target=True)

        # 画干扰声源 (灰色系)
        for i, pos in enumerate(interf_srcs):
            draw_source(ax, pos, i, "Interf", color='gray', is_target=False)

        ax.set_xlabel(f"{axis_names[0]} [m]")
        ax.set_ylabel(f"{axis_names[1]} [m]")
        ax.set_xlim(-0.5, room_dim[0] + 0.5)
        ax.set_ylim(-0.5, room_dim[d2] + 0.5)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')

    # 自定义图例说明
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', marker='X', linestyle='None', markersize=10, label='Mic Center'),
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8, label='Target (Static)'),
        Line2D([0], [0], color='blue', marker='>', linestyle='--', markersize=8, label='Target (Moving Path)'),
        Line2D([0], [0], color='gray', marker='^', linestyle='None', markersize=8, label='Interf'),
    ]
    fig.legend(handles=custom_lines, loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()
    plt.savefig(f"spatial_layout_mix_{mix_idx}.png")

def visualize_mixture(mixture_data, mix_idx, mid_to_label, duration=60):
    """
    可视化指定索引的 Mixture 音轨，标签旋转90度以防重叠
    """
    # 兼容性检查：如果是 target_classes
    if 'target_classes' in mixture_data:
        mix = mixture_data['target_classes'][mix_idx]
    else:
        print("Pickle 格式不符合预期")
        return

    files = mix['audiofile']
    mids = mix['mid']
    start_times = mix['start_time']
    durations = mix['duration']
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(18, 8))
    
    ax.set_xlim(0, duration)
    # 纵轴代表音轨层级，通常 max_polyphony 不会超过 5
    ax.set_ylim(-0.5, 4.5) 
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'Track {i}' for i in range(5)])
    ax.set_title(f'SELD Mixture Timeline Visualization - Mix Index: {mix_idx}', fontsize=15, pad=20)
    
    # 绘制背景网格（每秒一格）
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.set_xticks(range(0, duration + 1, 5)) # 每5秒显示一个刻度

    # 音轨排布逻辑
    current_track_ends = [-1.0] * 10 

    for i in range(len(files)):
        start = start_times[i]
        dur = durations[i]
        end = start + dur
        
        # 处理多标签情况：取第一个MID并查表
        raw_mid = mids[i].split(',')[0].strip()
        label = mid_to_label.get(raw_mid, raw_mid)
        fname = Path(files[i]).name
        
        # 寻找合适的音轨层（不重叠）
        assigned_track = 0
        for t_idx, t_end in enumerate(current_track_ends):
            if start >= t_end:
                assigned_track = t_idx
                current_track_ends[t_idx] = end
                break
        
        # 绘制片段矩形
        color = plt.cm.Set3(assigned_track % 12)
        rect = patches.Rectangle((start, assigned_track - 0.4), dur, 0.8, 
                                 linewidth=1, edgecolor='0.3', facecolor=color, alpha=0.8)
        ax.add_patch(rect)
        
        # --- 核心修改：标签文字旋转 90 度 ---
        # 将 Label 放在矩形中心
        ax.text(start + dur/2, assigned_track, f"T{i}:{label}", 
                fontsize=10, weight='bold', color='black',
                rotation=90,  # 旋转90度
                va='center',  # 垂直居中
                ha='center',  # 水平居中
                clip_on=True) # 如果超出矩形范围会被剪裁

        # 在矩形底部或顶部标注微小的文件名和起始时间（不旋转，保持水平）
        metadata_text = f"{fname[:15]}...\n{start:.1f}s"
        ax.text(start + 0.1, assigned_track - 0.35, metadata_text, 
                fontsize=7, color='0.2', verticalalignment='bottom', alpha=0.7)

    # 绘制 60秒 结束线
    ax.axvline(x=60, color='red', linestyle='--', linewidth=1, label='End of Mixture')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"timeline_mix_{mix_idx}.png")

def inspect_mixture_data(mixture_path, vocab_path, m_idx):
    mixture_path = Path(mixture_path)
    mixtures_file = mixture_path / 'mixtures.obj'
    srir_setup_file = mixture_path / 'srir_setup.obj'

    # 1. 加载标签词典
    mid_to_label = load_vocabulary(vocab_path)

    print(f"--- 正在读取目录: {mixture_path} ---")

    # 2. 读取 mixtures.obj
    if mixtures_file.exists():
        with open(mixtures_file, 'rb') as f:
            mixtures = pickle.load(f)
        
        print("Visualize timeline")
        visualize_mixture(mixtures, m_idx, mid_to_label)
        print(f"\n[mixtures.obj] 包含的主键: {list(mixtures.keys())}")
        
        if 'target_classes' in mixtures and len(mixtures['target_classes']) > 0:
            print(f"\n>>> 第{m_idx}条混合音频的target_classes片段配方:")
            mix = mixtures['target_classes'][m_idx]
            
            for key in mix.keys():
                values = mix[key]
                if key == 'mid':
                    # translated = [mid_to_label.get(m, f"Unknown({m})") for m in values]
                    translated = []
                    for m_str in values:
                        labels = [mid_to_label.get(sub_m.strip(), sub_m) for sub_m in m_str.split(',')]
                        translated.append(" + ".join(labels))
                    print(f"  {key:12}: {values}")
                    print(f"  {'LABEL':12}: {translated}")
                else:
                    print(f"  {key:12}: {values}")

        print(f"{len(mixtures['interf_classes'])=}")
        if 'interf_classes' in mixtures and len(mixtures['interf_classes']) > 0:
            print(f"\n>>> 第{m_idx}条混合音频的interf_classes片段配方:")
            mix = mixtures['interf_classes'][m_idx]
            
            for key in mix.keys():
                values = mix[key]
                if key == 'mid':
                    # translated = [mid_to_label.get(m, f"Unknown({m})") for m in values]
                    translated = []
                    for m_str in values:
                        labels = [mid_to_label.get(sub_m.strip(), sub_m) for sub_m in m_str.split(',')]
                        translated.append(" + ".join(labels))
                    print(f"  {key:12}: {values}")
                    print(f"  {'LABEL':12}: {translated}")
                else:
                    print(f"  {key:12}: {values}")
    else:
        print(f"错误: 找不到 {mixtures_file}")

    # 3. 读取 srir_setup.obj
    if srir_setup_file.exists():
        with open(srir_setup_file, 'rb') as f:
            srir_setup = pickle.load(f)
        
        print(f"\n{'-'*60}")
        print(f"[srir_setup.obj] 包含的主键: {list(srir_setup.keys())}")
        
        if 'target_classes' in srir_setup and len(srir_setup['target_classes']) > 0:
            print(f"\n>>> 第{m_idx}条混合音频的target_classes空间设置:")
            pprint(srir_setup['target_classes'][m_idx])

        if 'interf_classes' in srir_setup and len(srir_setup['interf_classes']) > 0:
            print(f"\n>>> 第{m_idx}条混合音频的interf_classes空间设置:")
            pprint(srir_setup['interf_classes'][m_idx])
    else:
        print(f"错误: 找不到 {srir_setup_file}")
        
    print("Visualize room")
    visualize_spatial_layout(srir_setup, mixtures, m_idx, mid_to_label)

if __name__ == "__main__":
    data_folder = 'database/seld_FSD50K_10_ov4_train' 
    vocabulary_file = 'database/FSD50K/FSD50K.ground_truth/vocabulary.csv' 
    
    for i in range(10):
        inspect_mixture_data(data_folder, vocabulary_file, m_idx=i)