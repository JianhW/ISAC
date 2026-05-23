function x_opt = Varshney(c, C_idx, R_idx, varargin)
% OPTIMIZE_WAVEFORM_CVX  通信-雷达一体化 OFDM 波形优化函数（基于 CVX）
%
% 功能：
%   对输入频域码元 c 进行迭代优化，通过求解凸优化问题（CVX + MOSEK/SDPT3），
%   在保证通信子载波频域误差最小的同时，将时域 PAPR 压缩到目标阈值以内。
%
% ─── 调用格式 ───────────────────────────────────────────────────────────
%   x_opt = optimize_waveform_cvx(c, C_idx, R_idx)
%   x_opt = optimize_waveform_cvx(c, C_idx, R_idx, Name, Value, ...)
%
% ─── 必选输入 ────────────────────────────────────────────────────────────
%   c       - 频域码元，(N×1) 复数列向量
%               N = 总子载波数（通信+雷达子载波之和）
%   C_idx   - 通信子载波索引向量，整数，范围 1…N
%               指定 c 中哪些位置承载通信符号（EVM 约束作用的位置）
%   R_idx   - 雷达子载波索引向量，整数，范围 1…N
%               指定 c 中哪些位置用作雷达波形（优化后该频点幅度置零）
%
% ─── 可选 Name-Value 参数 ─────────────────────────────────────────────────
%   'L'       - 过采样因子（正整数），默认 1
%   'Iter'    - 优化迭代次数，默认 2
%   'CR'      - 削峰比（Clipping Ratio）= sqrt(10^(PAPR_target_dB/10))
%               默认 sqrt(10^(5/10)) ≈ 1.778 → 目标 PAPR = 5 dB
%   'UseCVX'  - 是否调用 CVX 凸优化（true/false），默认 true
%               若为 false 或 CVX 不可用，退回到简单投影近似
%   'A'       - 预计算的 IFFT 变换矩阵，(LN×N) 复数矩阵
%               若不提供则函数内部自动生成
%               A(i,k) = (1/√LN) * exp(−j·2π·(i−1)·(k−1)/LN)
%   'Verbose' - 是否打印调试信息（true/false），默认 false
%
% ─── 输出 ─────────────────────────────────────────────────────────────────
%   x_opt   - 优化后的时域波形，(LN×1) 复数列向量，LN = L·N
%
% ─── 算法流程 ─────────────────────────────────────────────────────────────
%   Step 0: x = ifft([c; 0…0], LN) * √LN          (补零过采样 IFFT)
%   For it = 1 : Iter
%     Step 1: T_clip = CR · ‖x‖/√LN               (削峰阈值)
%             x_hat = clip(x, T_clip)               (幅度限幅)
%     Step 2: c_in = fft(x_hat)[1:N] / √LN        (提取带内分量)
%     Step 3: T_peak = CR · ‖x_hat‖/√LN           (时域峰值约束)
%     Step 4: CVX 求最优频域滤波器 H ∈ C^N：
%               min  t                              (相对 EVM)
%               s.t. ‖c(C_idx) − H(C_idx)·c_in(C_idx)‖ ≤ ‖c(C_idx)‖·t
%                    H(R_idx) · c_in(R_idx) = 0     (雷达子载波清零)
%                    |A · (H ⊙ c_in)| ≤ T_peak     (逐点时域峰值)
%     Step 5: C_new = [H ⊙ c_in; 0…0]  →  x = ifft(C_new) · √LN
%     Step 6: x = clip(x, T_peak)                  (二次轻削峰)
%   End
%
% ─── 依赖 ─────────────────────────────────────────────────────────────────
%   CVX     (http://cvxr.com/)          凸优化建模框架
%   MOSEK   (https://www.mosek.com/)    首选 SOCP 求解器（需许可证）
%   SDPT3                               备用求解器（CVX 自带）
%
% ─── 示例 ─────────────────────────────────────────────────────────────────
%   N = 128;  L = 4;
%   % 生成 QPSK 频域码元
%   data = randi([0 3], N, 1);
%   c = pskmod(data, 4, pi/4, 'gray');
%   % 所有子载波用于通信，无雷达子载波
%   x_opt = optimize_waveform_cvx(c, 1:N, [], 'L', L, 'Iter', 2, 'Verbose', true);
%
%   % ISAC 场景：前 100 个通信，后 28 个雷达
%   C_idx = 1:100;   R_idx = 101:128;
%   x_opt = optimize_waveform_cvx(c, C_idx, R_idx, 'L', 4, 'CR', sqrt(10^(5/10)));
%
% 版本：v1.0  (2026-03-30)
% 作者：ISAC Simulation Team

%% ──────────────── 参数检验与解析 ────────────────────────────────────────
if nargin < 3
    error('optimize_waveform_cvx: 至少需要 3 个输入参数 (c, C_idx, R_idx)');
end

% 强制列向量
c     = c(:);
C_idx = C_idx(:)';   % 行向量
R_idx = R_idx(:)';   % 行向量

N = length(c);

% 索引合法性检查
if any(C_idx < 1) || any(C_idx > N)
    error('C_idx 超出范围 [1, N=%d]', N);
end
if ~isempty(R_idx) && (any(R_idx < 1) || any(R_idx > N))
    error('R_idx 超出范围 [1, N=%d]', N);
end
if ~isempty(intersect(C_idx, R_idx))
    error('C_idx 与 R_idx 存在重叠子载波，请检查索引');
end

% 可选参数
p = inputParser;
addParameter(p, 'L',       1,                    @(x) isscalar(x) && x>=1 && mod(x,1)==0);
addParameter(p, 'Iter',    2,                    @(x) isscalar(x) && x>=1 && mod(x,1)==0);
addParameter(p, 'CR',      sqrt(10^(5/10)),      @(x) isscalar(x) && x>0);
addParameter(p, 'UseCVX',  true,                 @islogical);
addParameter(p, 'A',       [],                   @(x) isnumeric(x) || isempty(x));
addParameter(p, 'Verbose', false,                @islogical);
parse(p, varargin{:});

L       = p.Results.L;
iters   = p.Results.Iter;
CR      = p.Results.CR;
use_cvx = p.Results.UseCVX;
A       = p.Results.A;
verbose = p.Results.Verbose;

LN = L * N;

%% ──────────────── 预生成 IFFT 变换矩阵 A ────────────────────────────────
% A(i,k) = (1/√LN) * exp(−j·2π·i·k / LN)，(i,k 均从 0 起)
% 等价于：x ≈ A * c_in  →  ifft([c_in; zeros(…)]) * √LN
if isempty(A)
    n_vec = (0:LN-1).';
    k_vec = 0:(N-1);
    A = (1/sqrt(LN)) * exp(-1j * 2*pi * (n_vec * k_vec) / LN);  % LN×N
end

if size(A,1) ~= LN || size(A,2) ~= N
    error('A 矩阵维度错误：期望 (%d×%d)，实际 (%d×%d)', LN, N, size(A,1), size(A,2));
end

%% ──────────────── CVX 可用性检测 ────────────────────────────────────────
if use_cvx
    if exist('cvx_begin','file') ~= 2
        if exist('D:\cvx','dir')
            addpath(genpath('D:\cvx'));
        end
        try
            cvx_setup;  %#ok<NOEFF>
        catch
            use_cvx = false;
            warning('optimize_waveform_cvx: CVX 初始化失败，回落到近似求解');
        end
    end

    if use_cvx && exist('cvx_begin','file') == 2
        cvx_quiet(true);
        cvx_precision high;
        % 探针测试：优先尝试 MOSEK，失败则退回 SDPT3，两者都不行才禁用 CVX
        solver_ok = false;
        for try_solver = {'mosek', 'sdpt3'}
            try
                cvx_solver(try_solver{1});
                cvx_begin quiet
                    variable z_probe
                    minimize(z_probe)
                    subject to; z_probe >= 0
                cvx_end
                if strcmpi(cvx_status, 'Solved')
                    if verbose
                        fprintf('[CVX] 使用求解器: %s\n', try_solver{1});
                    end
                    solver_ok = true;
                    break;
                end
            catch
                % 当前求解器不可用，继续尝试下一个
            end
        end
        if ~solver_ok
            use_cvx = false;
            warning('optimize_waveform_cvx: CVX 求解器均不可用，回落到近似求解');
        end
    else
        use_cvx = false;
    end
end

%% ──────────────── 时域初始化 ────────────────────────────────────────────
c_padded = [c; zeros(LN - N, 1)];
x = ifft(c_padded) * sqrt(LN);

%% ──────────────── 主迭代 ────────────────────────────────────────────────
stat = struct('cvx_used',0,'cvx_solved',0,'fallback',0,'approx',0);

for it = 1:iters

    %--- Step 1: 削峰 ---
    T_clip = CR * norm(x) / sqrt(LN);
    x_hat  = local_clip(x, T_clip);

    %--- Step 2: 提取带内分量 ---
    C_hat = fft(x_hat) / sqrt(LN);   % 对应 (1/√LN)·FFT
    c_in  = C_hat(1:N);              % N×1

    %--- Step 3: 时域峰值约束 ---
    T_peak = CR * norm(x_hat) / sqrt(LN);

    %--- Step 4: 求最优滤波器 H ---
    if use_cvx && exist('cvx_begin','file') == 2
        try
            stat.cvx_used = stat.cvx_used + 1;

            if isempty(R_idx)
                %--- 纯通信场景：无雷达子载波清零约束 ---
                cvx_begin quiet
                    variable H(N) complex
                    variable t
                    minimize(t)
                    subject to
                        norm(c(C_idx) - H(C_idx) .* c_in(C_idx), 2) ...
                            <= norm(c(C_idx), 2) * t
                        abs(A * (H .* c_in)) <= T_peak
                cvx_end
            else
                %--- ISAC 场景：含雷达子载波清零约束 ---
                % 用选择矩阵提取 R_idx 处的约束，避免在 CVX 块内写 if
                nr = length(R_idx);
                S_r = zeros(N, nr);
                for ki = 1:nr; S_r(R_idx(ki), ki) = 1; end
                % H(R_idx) .* c_in(R_idx) == 0  ←→  diag(c_in(R_idx)) * (S_r' * H) == 0
                D_radar = diag(c_in(R_idx));

                cvx_begin quiet
                    variable H(N) complex
                    variable t
                    minimize(t)
                    subject to
                        norm(c(C_idx) - H(C_idx) .* c_in(C_idx), 2) ...
                            <= norm(c(C_idx), 2) * t
                        D_radar * (S_r' * H) == 0
                        abs(A * (H .* c_in)) <= T_peak
                cvx_end
            end

            if strcmpi(cvx_status, 'Solved')
                stat.cvx_solved = stat.cvx_solved + 1;
                H_opt = H;
            else
                if verbose
                    fprintf('[CVX WARN] it=%d, status=%s → feas_project\n', it, cvx_status);
                end
                stat.fallback = stat.fallback + 1;
                H_opt = local_feas_project(ones(N,1), c_in, LN, N, T_peak);
            end

        catch ME
            if verbose
                fprintf('[CVX ERROR] it=%d: %s\n', it, ME.message);
            end
            use_cvx = false;   % 当前符号后续迭代不再用 CVX
            stat.approx = stat.approx + 1;
            H_opt = local_feas_project(ones(N,1), c_in, LN, N, T_peak);
        end
    else
        stat.approx = stat.approx + 1;
        H_opt = local_feas_project(ones(N,1), c_in, LN, N, T_peak);
    end

    %--- Step 5: 重建时域波形 ---
    C_new        = zeros(LN, 1);
    C_new(1:N)   = H_opt .* c_in;
    % 雷达子载波强制清零（双保险）
    if ~isempty(R_idx)
        C_new(R_idx) = 0;
    end
    x = ifft(C_new) * sqrt(LN);

    %--- Step 6: 二次轻削峰 ---
    x = local_clip(x, T_peak);
end

x_opt = x;

if verbose
    papr_db = 10 * log10(max(abs(x_opt).^2) / mean(abs(x_opt).^2));
    fprintf('[optimize_waveform_cvx] 完成 | PAPR=%.2f dB | ', papr_db);
    fprintf('CVX %d/%d Solved | fallback %d | approx %d\n', ...
        stat.cvx_solved, stat.cvx_used, stat.fallback, stat.approx);
end

end  % ← 主函数结束

%% ══════════════════ 辅助函数（局部函数） ══════════════════════════════

function y = local_clip(x, T)
%LOCAL_CLIP  幅度限幅：|x(i)| > T 时等比缩放到 T
    y   = x;
    mag = abs(x);
    idx = mag > T;
    if any(idx)
        y(idx) = T * x(idx) ./ mag(idx);
    end
end

function H = local_feas_project(H_init, c_in, LN, N, T_peak)
%LOCAL_FEAS_PROJECT  可行化投影（无 CVX 时的回退方案）
%   一次 freq→time→clip→time→freq 投影，使时域峰值 ≤ T_peak
    spec = zeros(LN, 1);
    spec(1:N) = H_init .* c_in;
    xt = ifft(spec) * sqrt(LN);          % 时域
    xt = local_clip(xt, T_peak);         % 削峰
    Cxt = fft(xt) / sqrt(LN);           % 回频域，保持归一化一致
    Cxt(N+1:end) = 0;                   % 只保留带内 N 点
    H = Cxt(1:N) ./ (c_in + 1e-12);
end