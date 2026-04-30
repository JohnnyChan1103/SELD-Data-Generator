import os, warnings

import numpy as np
import scipy.special as scyspecial
import scipy.signal as scysignal
from itertools import product
from .ambisonics import Ambisonics as amb
import utils


class GenerateSRIR():
    """Spatial Room Impulse Response (SRIR) Generation"""
        
    def __init__(
        self, SH_order=1, fs=24000., src_dir='omni', mic_dir='omni', radius=0.042, 
        c=343., mic_pos=None, coord_type='sph', tools='pyroomacoustics', array_type='open'):
        """
        Parameters
        ----------
        SH_order : int, optional
            Order of sphercial harmony, by default 1
        fs : int, optional
            Sampling rate, by default 24000
        src_dir : str, optional
            Directivity of sources, by default 'omni'
        mic_dir : str, optional
            Directivity of microphones, by default 'omni'
        radius : float, optional
            Radius of spherical array, by default 0.042
        c : float, optional
            Speed of sound, by default 343.
        mic_pos : (num_mic, 2) or (num_mic, 3), array_like, optional
            Spherical coordinate position (azimuth, elevation) of microphones in degree, if coord_type='sph'
            Cartesian coordinate position (x, y, z) of microphones in meters, if coord_type='cart'
            by default None
        coord_type: 'sph' or 'cart', by default 'sph'
            Spherical corrdinate or Cartesian coordinates
        array_type : str, {'open', 'rigid'}, optional
            Type of micophone array, by default 'open'

        """        

        self.fs = fs
        self.src_dir = src_dir
        self.mic_dir = mic_dir
        self.array_type = array_type
        self.SH_order = SH_order
        self.radius = radius
        self.c = c

        assert tools in ['pyroomacoustics', 'gpuRIR', 'smir', 'pygsound', 'collectedRIR'], \
            "tools must be one of 'pyroomacoustics', 'gpuRIR', 'smir', 'pygsound, 'meshGWA'"
        self.tools = tools
        self.rir = None # (num_mic, num_src, num_points)

        if coord_type == 'sph':
            assert mic_pos.shape[1] == 2, f"Spherical coordinate must be (azimuth, elevation) but {mic_pos.shape[1]=}"
            if mic_pos is None:
                if SH_order is None:
                    raise ValueError("One of mic_pos and SH_order must be specified.")
                else:
                    if SH_order == 1:
                        self.mic_pos_sph = np.array([[45, 35], [-45, -35], 
                                                    [135, -35], [-135, 35]])
                        self.mic_pos_cart = utils.sph2cart(
                            self.mic_pos_sph[:,0], self.mic_pos_sph[:,1], radius)
                    else:
                        raise NotImplementedError("Only SH_order=1 is supported.")
            else:
                self.mic_pos_cart = utils.sph2cart(mic_pos[:,0], mic_pos[:,1], radius)
                self.mic_pos_sph = mic_pos
        
        if coord_type == 'cart':
            assert mic_pos.shape[1] == 3, f"Cartesian coordinate must be (x, y, z) but {mic_pos.shape[1]=}"
            self.mic_pos_cart = mic_pos
            self.mic_pos_sph = None


    def compute_srir(self, room_dim, src_pos, rt60, mic_pos_center=None, method='ism', **kwargs):
        if self.tools == 'pyroomacoustics':
            self.compute_srir_pra(room_dim, src_pos, rt60, mic_pos_center, method, **kwargs)
        elif self.tools == 'gpuRIR':
            self.compute_srir_gpuRIR(room_dim, src_pos, rt60, mic_pos_center, **kwargs)
        else: raise NotImplementedError("Only pyroomacoustics and gpuRIR are supported.")
     

    def compute_srir_pra(
        self, room_dim, src_pos, rt60=None, mic_pos_center=None, method='hybrid', **kwargs):
        """Compute an SRIR from a given room parameters
            using pyroomacoustics.

        Parameters
        ----------
        room_dim : (3,) array_like
            Room dimensions in meters
        src_pos : (num_src, 3) array_like
            Source positions in meters
        rt60 : float,
            Desired RT60 in seconds
        mic_pos_center : (3,) array_like, optional
            Position of center of microphone array, 
            The default None, which means the center of room.
        method : str, {'ism', 'hybrid'}, optional
            Method of rir generator, by default 'hybrid', which
            means image source method and ray tracing are used.
        kwargs : dict
            Additional arguments for pyroomacoustics.ShoeBox

        Returns
        -------
        array_like, shape (num_mic, num_src, length)
            Generated SRIR.
        """        

        import pyroomacoustics as pra

        if mic_pos_center is None:
            mic_pos = self.mic_pos_cart.T + np.c_[room_dim]/2
        else:
            mic_pos = self.mic_pos_cart.T + np.c_[mic_pos_center]

        if rt60 is not None:
            # We invert Sabine's formula to obtain the parameters for the ISM simulator
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            max_order = min(max_order, 50)
            if method == "ism":
                room = pra.ShoeBox(
                    p=room_dim, 
                    fs=self.fs, 
                    materials=pra.Material(e_absorption), 
                    max_order=max_order,)
            elif method == "hybrid":
                room = pra.ShoeBox(
                    p=room_dim,
                    fs=self.fs,
                    materials=pra.Material(e_absorption),
                    max_order=max_order,
                    ray_tracing=True,
                    air_absorption=True,)
        else:
            enable = method == 'hybrid'
            room = pra.ShoeBox(
            p=room_dim, fs=self.fs, 
            ray_tracing=enable, 
            air_absorption=enable, 
            **kwargs)
        
        for pos in src_pos:
            room.add_source(pos)
        room.add_microphone_array(mic_pos)
        try:
            room.compute_rir()
        except:
            print(rt60, room_dim, src_pos, mic_pos_center, method, kwargs)
            raise ValueError('RIR computation failed.')

        self.c = room.c
        self.rir = room.rir


    def compute_srir_gpuRIR( 
        self, room_dim, src_pos, rt60, mic_pos_center=None, abs_weights=[0.9]*5+[0.5], 
        att_diff=15.0, att_max=60.0, **kwargs):
        """Compute an SRIR from a given room parameters
            using gpuRIR

        Parameters
        ----------
        room_dim : (3,) array_like
            Room dimensions in meters
        src_pos : (num_src, 3) array_like
            Source positions in meters
        rt60 : float,
            Desired RT60 in seconds
        abs_weights : list, optional
            Absorption coefficient ratios of the walls,
            by default [0.9]*5+[0.5]
        att_diff : float, optional
            Desired attenuation (in dB), by default 15.0 dB
        att_max : float, optional
            Maximum attenuation (in dB), by default 60.0 dB
        mic_pos_center : (3,) array_like, optional
            Position of center of microphone array, 
            The default None, which means the center of room.
        kwargs : dict
            Additional arguments for gpuRIR.simulateRIR

        Returns
        -------
        array_like, shape (num_mic, num_src, length)
            Generated SRIR.
        """        
        
        import gpuRIR

        if mic_pos_center is None:
            mic_pos = self.mic_pos_cart + room_dim / 2
        else:
            mic_pos = self.mic_pos_cart + mic_pos_center
        room_dim = np.array(room_dim)
        src_pos = np.array(src_pos)

        beta = gpuRIR.beta_SabineEstimation(room_dim, rt60, abs_weights=abs_weights) # Reflection coefficients
        Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, rt60) # Time to start the diffuse reverberation model [s]
        Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt60)	 # Time to stop the simulation [s]
        nb_img = gpuRIR.t2n( Tdiff, room_dim )	# Number of image sources in each dimension
        rir = gpuRIR.simulateRIR(
            room_dim, beta, src_pos, mic_pos, nb_img, 
            Tmax, self.fs, Tdiff=Tdiff, **kwargs)
        self.rir = np.transpose(rir, (1, 0, 2))

    # def compute_moving_srir_gpuRIR(self, room_dim, trajectories, rt60, mic_pos_center=None, **kwargs):
    #     """
    #     使用 gpuRIR 批量计算移动轨迹的 RIR
    #     """
    #     import gpuRIR
    #     mic_pos = self.mic_pos_cart + (mic_pos_center if mic_pos_center is not None else room_dim/2)
        
    #     all_points = np.vstack(trajectories)
    #     beta = gpuRIR.beta_SabineEstimation(room_dim, rt60)
    #     Tdiff = gpuRIR.att2t_SabineEstimator(15.0, rt60)
    #     Tmax = gpuRIR.att2t_SabineEstimator(60.0, rt60)
    #     nb_img = gpuRIR.t2n(Tdiff, room_dim)
        
    #     rir_batch = gpuRIR.simulateRIR(
    #         room_dim, beta, all_points, mic_pos, nb_img, 
    #         Tmax, self.fs, Tdiff=Tdiff, **kwargs)
        
    #     # 按声源切分
    #     structured_rir = []
    #     cursor = 0
    #     for traj in trajectories:
    #         num_pts = len(traj)
    #         structured_rir.append(np.transpose(rir_batch[cursor:cursor+num_pts], (1, 0, 2)))
    #         cursor += num_pts
    #     self.moving_rirs = structured_rir
    #     return structured_rir

    def compute_moving_srir_gpuRIR(self, room_dim, trajectories, rt60, mic_pos_center=None, max_points_per_gpu_call=1500, **kwargs):
        """
        分块计算移动轨迹 RIR，防止显存溢出。
        max_points_per_gpu_call: 每次传给 GPU 的最大坐标点数。
        """
        import gpuRIR
        import numpy as np

        mic_pos = self.mic_pos_cart + (mic_pos_center if mic_pos_center is not None else room_dim/2)
        all_points = np.vstack(trajectories)
        total_points = len(all_points)

        # 基础物理参数计算
        beta = gpuRIR.beta_SabineEstimation(room_dim, rt60)
        Tdiff = gpuRIR.att2t_SabineEstimator(15.0, rt60)
        Tmax = gpuRIR.att2t_SabineEstimator(60.0, rt60)
        nb_img = gpuRIR.t2n(Tdiff, room_dim)

        # 分块逻辑
        rir_chunks = []
        for i in range(0, total_points, max_points_per_gpu_call):
            # 截取当前块的坐标点
            point_chunk = all_points[i : i + max_points_per_gpu_call]
            
            # 核心 GPU 调用
            # 返回形状: (chunk_points, num_mic, rir_len)
            chunk_res = gpuRIR.simulateRIR(
                room_dim, beta, point_chunk, mic_pos, nb_img, 
                Tmax, self.fs, Tdiff=Tdiff, **kwargs
            )
            # 立即转存到内存（CPU）中，防止堆积在显存引用里
            rir_chunks.append(chunk_res)

        # 在 CPU 上合并所有块
        rir_batch = np.concatenate(rir_chunks, axis=0)
        
        # 清理中间变量，辅助 GC
        del rir_chunks

        # --- 后续的拆分逻辑保持不变 ---
        structured_rir = []
        cursor = 0
        for traj in trajectories:
            num_pts = len(traj)
            # 转置为 (num_mic, N_i, rir_len)
            structured_rir.append(np.transpose(rir_batch[cursor:cursor+num_pts], (1, 0, 2)))
            cursor += num_pts
            
        return structured_rir

    def simulate_rigid_sph_array(self, src_pos_mic, n_points, order=30):
        """Rigid spherical array response simulation.

        Parameters
        ----------
        src_pos : (num_src, 3) array_like
            Source positions in meters
        n_points : int
            Points of filter.
        order : int, optional
            Order of expension term, by default 30
            The expansion is limited to 30 terms which provide 
                negligible modeling error up to 20 kHz.

        Returns
        -------
        h_mic: (num_src, num_mic, n_points) array_like
            Spherical array response.
        """

        src_pos_mic = np.asarray(src_pos_mic)
        c = self.c

        # Compute the frequency-dependent part of the microphone responses
        f = np.linspace(0, self.fs//2, n_points//2+1)
        kr = 2 * np.pi * f / c * self.radius

        b_n = np.zeros((order+1, len(f)), dtype=np.complex128)
        for n in range(order+1):
            b_n[n] = amb.mode_strength(n=n, kr=kr, sphere_type=self.array_type)
        temp = b_n
        temp[:, -1] = np.real(temp[:, -1])
        temp = np.concatenate((temp, temp[:,-2:0:-1].conj()), axis=1)
        b_nt = np.fft.fftshift(np.fft.ifft(temp, axis=1), axes=1).real

        # Compute angular-dependent part of the microphone responses
        # unit vectors of DOAs and microphones
        N_doa = len(src_pos_mic)
        N_mic = len(self.mic_pos_cart)
        h_mic = np.zeros((n_points, N_mic, N_doa))
        H_mic = np.zeros((n_points//2+1, N_mic, N_doa), dtype=np.complex128)
        for i in range(N_doa):
            cosAngle = np.dot(
                self.mic_pos_cart / self.radius, 
                src_pos_mic[i,:] / np.linalg.norm(src_pos_mic[i,:]))
            P = np.zeros((order+1, N_mic))
            for n in range(order+1):
                Pn = scyspecial.lpmv(0, n, cosAngle)
                P[n, :] = (2*n+1) / (4 * np.pi) * Pn
            
            h_mic[:,:,i] = b_nt.T @ P
            H_mic[:,:,i] = b_n.T @ P

        return h_mic.transpose(2,1,0), H_mic.transpose(2,1,0)


    def simulate(self, src_pos_mic, src_signals, n_points=2048, **kwargs):
        """Simulates the microphone signal at every microphone in the array
        
        """

        assert len(src_pos_mic) == len(src_signals), \
            "Number of source position and signals must be equal."
        assert self.rir is not None, "Room impulse response is not computed."

        num_src = len(src_pos_mic)
        num_mic = len(self.rir)

        max_len_rir = np.array(
            [len(self.rir[i][j]) for i, j in product(range(num_mic), range(num_src))]
        ).max()
        f = lambda i: len(src_signals[i])
        max_sig_len = np.array([f(i) for i in range(num_src)]).max()
        num_points = int(max_len_rir) + int(max_sig_len) - 1
        if num_points % 2 == 1:
            num_points += 1
        # the array that will receive all the signals
        premix_signals = np.zeros((num_src, num_mic, num_points))
        # compute the signal at every microphone in the array
        for m in np.arange(num_mic):
            for s in np.arange(num_src):
                sig = src_signals[s]
                if sig is None:
                    continue
                h = self.rir[m][s]
                premix_signals[s, m, :len(sig) + len(h) - 1] += \
                    scysignal.fftconvolve(h, sig)

        if self.array_type == 'open' or self.tools == 'smir':
            return np.sum(premix_signals, axis=0)
        elif self.array_type == 'rigid':
            h_mic, H_mic = self.simulate_rigid_sph_array(src_pos_mic, n_points=n_points)
            return scysignal.fftconvolve(premix_signals, h_mic, axes=-1).sum(axis=0)
        else:
            raise ValueError('Array type not supported.')
            


    # def simulate_moving(self, src_signals):
    #     """
    #     src_signals: list of signals
    #     moving_rirs: 由 compute_moving_srir_gpuRIR 生成的列表
    #     """
    #     num_mic = len(self.mic_pos_cart)
    #     fs = self.fs
    #     frame_len = int(0.1 * fs) # 100ms 分辨率
        
    #     # 最终混合信号
    #     # 计算总长度 (信号长 + RIR长)
    #     max_sig_len = max([len(s) for s in src_signals])
    #     rir_len = self.moving_rirs[0].shape[2]
    #     out_len = max_sig_len + rir_len
    #     output_mic = np.zeros((num_mic, out_len))

    #     for s_idx, sig in enumerate(src_signals):
    #         rirs = self.moving_rirs[s_idx] # (num_mic, num_frames, rir_length)
    #         num_frames = rirs.shape[1]
            
    #         for f_idx in range(num_frames):
    #             start_sample = f_idx * frame_len
    #             end_sample = min(start_sample + frame_len, len(sig))
    #             if start_sample >= len(sig): break
                
    #             segment = sig[start_sample:end_sample]
                
    #             for m_idx in range(num_mic):
    #                 h = rirs[m_idx, f_idx, :]
    #                 # 卷积并叠加到对应位置
    #                 conv_seg = scysignal.fftconvolve(segment, h)
    #                 output_mic[m_idx, start_sample : start_sample + len(conv_seg)] += conv_seg
                    
    #     return output_mic

    def simulate_moving(self, src_signals, moving_rirs, update_interval, crossfade_len=0.02):
        """
        使用时变RIR和分段交叉淡化模拟移动声源。
        
        参数:
        ----------
        src_signals : List[np.ndarray]
            各个声源的干燥信号列表。
        moving_rirs : List[np.ndarray]
            各个声源的RIR序列，形状为 [num_mic, num_frames, rir_len]。
        update_interval : float
            RIR更新的时间间隔（秒）。
        crossfade_len : float
            淡入淡出的长度（秒），建议设为 update_interval 的 10%-20%。
        """
        import scipy.signal as scysignal

        num_mic = len(self.mic_pos_cart)
        fs = self.fs
        
        # 将秒转换为采样点数
        frame_samples = int(update_interval * fs)
        cf_samples = int(crossfade_len * fs)
        
        # 确保淡入淡出长度不超过间隔的一半
        cf_samples = min(cf_samples, frame_samples // 2)
        
        # 定义线性淡入淡出窗
        fade_in = np.linspace(0, 1, cf_samples)
        fade_out = np.linspace(1, 0, cf_samples)

        # 预先计算总输出长度
        max_sig_len = max([len(s) for s in src_signals])
        rir_len = moving_rirs[0].shape[2]
        total_len = max_sig_len + rir_len
        output_mic = np.zeros((num_mic, total_len))

        for s_idx, sig in enumerate(src_signals):
            # 当前声源的RIR序列 [mic, frames, rir_len]
            rirs = moving_rirs[s_idx]
            num_frames = rirs.shape[1]
            
            # 创建当前声源的临时缓存，用于存储该声源的多通道卷积结果
            source_output = np.zeros((num_mic, total_len))

            for f_idx in range(num_frames):
                # 1. 确定当前音频段的范围
                # 我们需要额外多取 cf_samples 长度的音频用于做下一段的淡入淡出
                start_s = f_idx * frame_samples
                end_s = min(start_s + frame_samples + cf_samples, len(sig))
                
                if start_s >= len(sig):
                    break
                    
                segment = sig[start_s:end_s]
                
                # 2. 对每个麦克风通道进行处理
                for m_idx in range(num_mic):
                    h = rirs[m_idx, f_idx, :]
                    # 进行卷积
                    conv_res = scysignal.fftconvolve(segment, h)
                    
                    # --- 核心 Cross-fade 逻辑 ---
                    
                    # A. 如果不是第一帧，对当前段的头部进行淡入
                    if f_idx > 0:
                        conv_res[:cf_samples] *= fade_in
                    
                    # B. 如果不是最后一帧，对当前段的尾部（overlap部分）进行淡出
                    # 注意：淡出发生在 frame_samples 到 frame_samples + cf_samples 这一段
                    if f_idx < num_frames - 1:
                        # 确定淡出的起始位置
                        tail_start = frame_samples
                        tail_end = frame_samples + cf_samples
                        
                        if len(conv_res) > tail_end:
                            conv_res[tail_start:tail_end] *= fade_out
                            # 超过淡出区间的部分直接置零，因为它将由下一帧负责
                            conv_res[tail_end:] = 0
                        elif len(conv_res) > tail_start:
                            # 如果卷积结果长度不足以覆盖整个淡出区，也要部分淡出
                            valid_cf_len = len(conv_res) - tail_start
                            conv_res[tail_start:] *= fade_out[:valid_cf_len]

                    # 3. 叠加到声源缓存中
                    # 叠加位置是根据当前帧的起始采样点 start_s
                    sig_slice = source_output[m_idx, start_s : start_s + len(conv_res)]
                    sig_slice += conv_res[:len(sig_slice)]

            # 将该声源的结果累加到总混音中
            output_mic += source_output

        return output_mic
    
    def simulate_moving_ltv(self, src_signals, moving_rirs, update_interval, win_size=1024):
        """
        使用方案 B 的 LTV 逻辑进行渲染
        src_signals: 干燥音频列表
        moving_rirs: compute_moving_srir_gpuRIR 返回的列表
        update_interval: RIR 更新间隔 (例如 0.1s)
        """
        import numpy as np
        
        num_mic = len(self.mic_pos_cart)
        # 计算输出的最大可能长度
        max_sig_len = max([len(s) for s in src_signals])
        rir_len = moving_rirs[0].shape[2]
        total_output = np.zeros((max_sig_len + rir_len, num_mic)) # 注意渲染引擎输出通常是 (L, C)

        for s_idx, sig in enumerate(src_signals):
            print(f"{s_idx=}")
            # 1. 准备该声源的 RIR 序列
            # 原始形状: (num_mic, num_pts, rir_len)
            # 转换到 ctf_ltv_direct 要求的: (rir_len, num_mic, num_pts)
            irs = np.transpose(moving_rirs[s_idx], (2, 0, 1))
            
            # 2. 生成对应的时间点 (秒)
            # 假设 RIR 是从该声源开始时刻起，每隔 update_interval 采样的
            num_pts = irs.shape[2]
            ir_times = np.arange(num_pts) * update_interval
            
            # 3. 调用方案 B 的核心引擎
            # 注意：sig 这里应该是声源的有效音频部分
            try:
                rendered_event = utils.ctf_ltv_direct(
                    sig=sig, 
                    irs=irs, 
                    ir_times=ir_times, 
                    fs=self.fs, 
                    win_size=win_size
                )
                
                # 4. 叠加到总输出
                # ctf_ltv_direct 返回的大小通常由音频长度决定
                l_ev = rendered_event.shape[0]
                total_output[:l_ev, :] += rendered_event
                
            except Exception as e:
                print(f"Error rendering source {s_idx}: {e}")
                
        return total_output.T # 转回 (num_mic, Length) 适配你的脚本