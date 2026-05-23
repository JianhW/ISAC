%% OCDM / OFDM BER simulation
% Compare original OCDM and original OFDM under the same ISAC background as
% OCDM_CCDF_main.m. Only embedded 4QAM(QPSK) communication BER with MMSE
% equalization is simulated.

clear; clc; close all;

script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir)
    addpath(fullfile(script_dir, 'Algorithms'));
end

%% Parameters aligned with OCDM_CCDF_main.m
N = 128;                   % Total number of subcarriers
Nc = 64;                   % Number of communication subcarriers
Nr = N - Nc;               % Number of radar subcarriers
modSC = 'QPSK';            % Communication modulation, equivalent to 4QAM
e = 0.2;                   % Communication energy ratio
monte = 1000;                % Monte Carlo trials
rng(1);

%% BER-only channel and equalization settings
channel_taps = 16;         % Multipath channel length
cyclic_prefix_len = 32;    % CP length, no shorter than channel_taps
Equal = 2;                 % Equalization interface: 1=ZF, 2=MMSE
SNR_dB_list = 0:2:30;
resume_simulation = true;  % true: continue from result_file checkpoint

%% 结果保存 / 断点续仿接口
% 断点续仿使用方法：
%   1. 首次运行：
%        设置 resume_simulation = true
%        设置 monte 为当前阶段目标次数，例如 monte = 1000
%        运行本脚本。每完成一种波形仿真后，程序会自动将
%        err_count、bit_count、mc_done 以及随机数状态
%        保存到 result_file 中。
%
%   2. 在之前结果基础上继续运行：
%        保持 result_file、N、Nc、SNR_dB_list、waveform_specs、
%        channel_taps、cyclic_prefix_len、Equal 以及算法参数不变。
%        增大 monte，例如从 1000 改为 2000。
%        保持 resume_simulation = true。
%        程序会自动加载 result_file，并从 mc_done + 1 开始继续仿真。
%
%   3. 开始新的独立仿真：
%        可以设置 resume_simulation = false，
%        或修改 result_file / use_auto_result_file 参数，
%        使用新的检查点文件。
%
%   注意：
%        如果在断点续仿时修改了关键参数，
%        程序会触发 checkpoint mismatch 错误。
%        这样可以避免不同仿真配置的数据被错误混合。

result_dir = fullfile(script_dir, 'Results');
use_auto_result_file = true; % false：严格使用 manual_result_file
manual_result_file = fullfile(result_dir, 'BER_my_experiment.mat');

%% 算法设置
% 所有波形均使用相同的比特流、子载波分配、信道、噪声、
% MMSE 均衡器以及 idx_comm BER 检测方式。
% 当已经保存了部分 OFDM/OCDM 数据后，
% 可以通过修改该列表，仅继续运行部分算法。
waveform_specs = {
    'Mine',     'Proposed Algorithm', 'OCDM';
    'OCDM',     'Original OCDM', 'OCDM';
    'OFDM',     'Original OFDM', 'OFDM';
    'Varshney', 'OFDM Varshney', 'OFDM';
    'Wang',     'OFDM New-ICF',  'OFDM';
};
waveform_keys = waveform_specs(:, 1).';
waveforms = waveform_specs(:, 2).';
waveform_domains = waveform_specs(:, 3).';
active_waveform_mask = true(size(waveform_keys));
% Example: only rerun the slow algorithms while keeping cached OFDM/OCDM
% active_waveform_mask = ismember(waveform_keys, {'Mine', 'Varshney', 'Wang'});
if ~all(active_waveform_mask)
    waveform_specs = waveform_specs(active_waveform_mask, :);
    waveform_keys = waveform_specs(:, 1).';
    waveforms = waveform_specs(:, 2).';
    waveform_domains = waveform_specs(:, 3).';
    active_waveform_mask = true(size(waveform_keys));
end

algorithm_params.mine_lambda = 0.0016;
algorithm_params.mine_l_norm_order = 100;
algorithm_params.mine_iteration = 1000;
algorithm_params.mine_min_iteration = 200;
algorithm_params.mine_os_factor = 4;
algorithm_params.mine_eps_stop = 1e-7;
algorithm_params.mine_lnorm_bound = 'current';
algorithm_params.mine_eig_mode = 'upper';
algorithm_params.mine_update_mode = 'fast';
algorithm_params.mine_mm_period = 0;
algorithm_params.mine_max_backtracking = 14;
algorithm_params.mine_lambda_shrink = 0.5;
algorithm_params.mine_papr_weight = 0.85;

algorithm_params.varshney_iteration = 2;
algorithm_params.varshney_eps_stop = 1e-5;
algorithm_params.varshney_output_mode = 'mapped';

algorithm_params.wang_L = 4;
algorithm_params.wang_iteration = 2;
algorithm_params.wang_use_cvx = false;

waveform_file_tag = strjoin(waveform_keys, '_');
if ~isfolder(result_dir)
    mkdir(result_dir);
end
auto_result_name = sprintf('BER_ISAC_%s_N%d_Nc%d_e%.1f_MC%d_SNR%d_%d_%d.mat', ...
    waveform_file_tag, N, Nc, e, monte, ...
    SNR_dB_list(1), SNR_dB_list(2) - SNR_dB_list(1), SNR_dB_list(end));
if use_auto_result_file
    result_file = fullfile(result_dir, auto_result_name);
else
    result_file = ResolveResultPath(manual_result_file, result_dir);
end

if cyclic_prefix_len < channel_taps - 1
    error('cyclic_prefix_len must be no shorter than channel_taps - 1.');
end
fprintf('BER result checkpoint: %s\n', result_file);

nWave = numel(waveforms);
nSNR = numel(SNR_dB_list);

err_count = zeros(nWave, nSNR);
bit_count = zeros(nWave, nSNR);
mc_done = zeros(nWave, 1);

Psi = DFnT(N);
FH = @(x) ifft(x(:)) * sqrt(N);

%% Resume checkpoint
rng_state = rng;
if resume_simulation && isfile(result_file)
    saved_checkpoint = load(result_file);
    ValidateCheckpoint(saved_checkpoint, SNR_dB_list, waveform_keys, N, Nc, e, channel_taps, cyclic_prefix_len, Equal);
    if isfield(saved_checkpoint, 'err_count')
        err_count = saved_checkpoint.err_count;
    end
    if isfield(saved_checkpoint, 'bit_count')
        bit_count = saved_checkpoint.bit_count;
    end
    if isfield(saved_checkpoint, 'mc_done')
        mc_done = saved_checkpoint.mc_done(:);
    elseif isfield(saved_checkpoint, 'bit_count')
        mc_done = EstimateDoneFromBitCount(saved_checkpoint.bit_count, 2 * Nc);
    end
    if numel(mc_done) ~= nWave
        error('Checkpoint mc_done length does not match the current waveform list.');
    end
    if isfield(saved_checkpoint, 'rng_state_after')
        rng(saved_checkpoint.rng_state_after);
    elseif isfield(saved_checkpoint, 'rng_state')
        rng(saved_checkpoint.rng_state);
    end
    rng_state = rng;
    fprintf('Resuming BER simulation from %s\n', result_file);
    for w = 1:nWave
        fprintf('  %-14s completed %d/%d Monte Carlo trials\n', ...
            waveforms{w}, min(mc_done(w), monte), monte);
    end
end

%% Monte Carlo BER simulation, run separately for each algorithm
progress_step = max(1, floor(monte / 10));
for w = 1:nWave
    if mc_done(w) >= monte
        fprintf('Waveform %d/%d: %s already completed (%d/%d)\n', ...
            w, nWave, waveforms{w}, mc_done(w), monte);
        continue;
    end
    fprintf('Waveform %d/%d: %s\n', w, nWave, waveforms{w});
    for mc = mc_done(w) + 1:monte
        if mod(mc, progress_step) == 0 || mc == 1 || mc == monte
            fprintf('  Monte Carlo: %d/%d\n', mc, monte);
        end

        [trial_err, trial_bits] = RunOneTrial(N, Nc, Nr, modSC, e, ...
            SNR_dB_list, channel_taps, Equal, waveform_keys{w}, ...
            waveform_domains{w}, algorithm_params, Psi, FH);
        err_count(w, :) = err_count(w, :) + trial_err;
        bit_count(w, :) = bit_count(w, :) + trial_bits;
        mc_done(w) = mc;
    end
    ber = err_count ./ max(bit_count, 1);
    SaveCheckpoint(result_file, ber, err_count, bit_count, mc_done, ...
        SNR_dB_list, waveforms, waveform_keys, waveform_domains, ...
        N, Nc, Nr, modSC, e, monte, channel_taps, cyclic_prefix_len, ...
        Equal, algorithm_params, rng_state, rng);
    fprintf('Checkpoint saved to %s\n', result_file);
end

ber = err_count ./ max(bit_count, 1);
SaveCheckpoint(result_file, ber, err_count, bit_count, mc_done, ...
    SNR_dB_list, waveforms, waveform_keys, waveform_domains, ...
    N, Nc, Nr, modSC, e, monte, channel_taps, cyclic_prefix_len, ...
    Equal, algorithm_params, rng_state, rng);
fprintf('Saved BER result data to %s\n', result_file);

fprintf('\nBER summary at highest SNR = %.1f dB:\n', SNR_dB_list(end));
for w = 1:nWave
    fprintf('  %-14s %.3e\n', waveforms{w}, ber(w, end));
end

%% BER plotting, aligned with OCDM_CCDF_main.m style
% ===== Professional Academic Color Palette =====
plot_colors = [
    0.7200, 0.2200, 0.1200;   % Proposed (highlight)
    0.0000, 0.2700, 0.5200;   % Baseline 1
    0.0000, 0.5000, 0.3200;   % Baseline 2
    0.4500, 0.1800, 0.6200;   % Baseline 3
    0.7800, 0.5500, 0.0000;   % Baseline 4
];

% ===== Solid publication-style lines =====
line_styles = {'-', '-', '-', '-', '-'};

% ===== Distinct professional markers =====
marker_styles = { 'h', 'o', 's','^', 'd'};
line_width = 3.0;
marker_size = 10.5;
n_markers = 11;
legend_handles = gobjects(1, nWave);

fig = figure('Color', 'w', 'Position', [120, 120, 760, 560]);
ax = axes(fig);
hold(ax, 'on');
box(ax, 'on');
set(ax, 'FontName', 'Times New Roman', ...
        'FontSize', 13, ...
        'LineWidth', 1.2, ...
        'TickDir', 'in', ...
        'TickLength', [0.018, 0.018], ...
        'YScale', 'log');

for w = 1:nWave
    ber_plot = ber(w, :);
    ber_plot(ber_plot == 0) = NaN;
    marker_idx = unique(round(linspace(1, nSNR, min(n_markers, nSNR))));

    legend_handles(w) = semilogy(ax, SNR_dB_list, ber_plot, ...
        'LineStyle', line_styles{mod(w - 1, numel(line_styles)) + 1}, ...
        'Color', plot_colors(mod(w - 1, size(plot_colors, 1)) + 1, :), ...
        'LineWidth', line_width, ...
        'Marker', marker_styles{mod(w - 1, numel(marker_styles)) + 1}, ...
        'MarkerIndices', marker_idx, ...
        'MarkerSize', marker_size, ...
        'MarkerFaceColor', 'w');
end

grid(ax, 'on');
ax.GridAlpha = 0.18;
ax.MinorGridAlpha = 0.10;
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';
ax.Layer = 'top';
xlim(ax, [SNR_dB_list(1), SNR_dB_list(end)]);
ylim(ax, [1e-6, 1]);

xlabel(ax, 'SNR of receiving signal (dB)', ...
    'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold');
ylabel(ax, 'BER', ...
    'FontName', 'Times New Roman', 'FontSize', 15, 'FontWeight', 'bold');
title(ax, sprintf('BER Performance (N = %d, N_c = %d, e = %.1f, MC = %d)', ...
    N, Nc, e, monte), ...
    'FontName', 'Times New Roman', 'FontSize', 14, 'FontWeight', 'bold');

lgd = legend(ax, legend_handles, waveforms, ...
    'Location', 'southwest', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 11);
lgd.Box = 'off';
hold(ax, 'off');

function [bits, z, idx_comm] = GenerateISACCodeword(N, Nc, Nr, modSC, e)
rand_idx = randperm(N);
idx_comm = sort(rand_idx(1:Nc)).';
mask_comm = false(N, 1);
mask_comm(idx_comm) = true;
idx_radar = find(~mask_comm);

switch upper(modSC)
    case 'QPSK'
        bits = randi([0 1], 2 * Nc, 1);
        zc_unit = (1 / sqrt(2)) * ...
            ((2 * bits(1:2:end) - 1) + 1j * (2 * bits(2:2:end) - 1));
    otherwise
        error('Only QPSK/4QAM is supported in this BER simulation.');
end

zc = sqrt(e) * zc_unit / sqrt(Nc);

Er = 1 - e;
if Er <= 0
    error('Parameter e is too large and leaves no radar energy.');
end

zr = randn(Nr, 1) + 1j * randn(Nr, 1);
zr = zr / norm(zr) * sqrt(Er);

z = zeros(N, 1);
z(idx_comm) = zc;
z(idx_radar) = zr;
end

function Hfreq = GenerateChannel(L, N)
h = (1 / sqrt(2 * L)) * (randn(1, L) + 1i * randn(1, L));
Hfreq = fft([h(:); zeros(N - L, 1)]);
end

function [trial_err, trial_bits] = RunOneTrial(N, Nc, Nr, modSC, e, ...
    SNR_dB_list, channel_taps, Equal, waveform_key, waveform_domain, ...
    algorithm_params, Psi, FH)
nSNR = numel(SNR_dB_list);
[bits, z, idx_comm] = GenerateISACCodeword(N, Nc, Nr, modSC, e);
idx_radar = setdiff((1:N).', idx_comm);

z_waveform_cell = GenerateWaveformCodewords(z, idx_comm, idx_radar, ...
    waveform_key, algorithm_params);
z_waveform = z_waveform_cell{1};

switch upper(waveform_domain)
    case 'OCDM'
        tx_time = Psi' * z_waveform;
        tx_freq = fft(tx_time);
    case 'OFDM'
        tx_time = FH(z_waveform);
        tx_freq = fft(tx_time);
    otherwise
        error('Unknown waveform domain: %s', waveform_domain);
end

Hfreq = GenerateChannel(channel_taps, N);
rx_clean_freq = Hfreq .* tx_freq;
noise_var_vec = (1 / N) ./ (10.^(SNR_dB_list / 10));
noise_freq_all = fft((randn(N, nSNR) + 1i * randn(N, nSNR)) / sqrt(2));

rx_freq_all = rx_clean_freq + sqrt(noise_var_vec) .* noise_freq_all;
if Equal == 1
    eq_freq_all = rx_freq_all ./ Hfreq;
elseif Equal == 2
    eq_freq_all = conj(Hfreq) .* rx_freq_all ./ (abs(Hfreq).^2 + N * noise_var_vec);
else
    error('Unknown equalization mode. Use 1=ZF or 2=MMSE.');
end
eq_time_all = ifft(eq_freq_all, [], 1);

trial_err = zeros(1, nSNR);
trial_bits = numel(bits) * ones(1, nSNR);
for snr_idx = 1:nSNR
    switch upper(waveform_domain)
        case 'OCDM'
            z_hat = Psi * eq_time_all(:, snr_idx);
        case 'OFDM'
            z_hat = fft(eq_time_all(:, snr_idx)) / sqrt(N);
    end

    bits_hat = QPSKDemod(z_hat(idx_comm));
    trial_err(snr_idx) = sum(bits_hat ~= bits);
end
end

function z_waveforms = GenerateWaveformCodewords(z, idx_comm, idx_radar, waveform_keys, params)
if ischar(waveform_keys) || isstring(waveform_keys)
    waveform_keys = cellstr(waveform_keys);
end
z_waveforms = cell(numel(waveform_keys), 1);
for w = 1:numel(waveform_keys)
    switch upper(waveform_keys{w})
        case {'OCDM', 'OFDM'}
            z_current = z;

        case 'MINE'
            z_current = Mine(z, idx_comm, idx_radar, ...
                'lambda', params.mine_lambda, ...
                'adaptiveLambda', false, ...
                'osFactor', params.mine_os_factor, ...
                'maxIt', params.mine_iteration, ...
                'minIt', params.mine_min_iteration, ...
                'lNormOrder', params.mine_l_norm_order, ...
                'epsStop', params.mine_eps_stop, ...
                'lnormBound', params.mine_lnorm_bound, ...
                'etaMode', params.mine_eig_mode, ...
                'UpdateMode', params.mine_update_mode, ...
                'mmPeriod', params.mine_mm_period, ...
                'maxBacktracking', params.mine_max_backtracking, ...
                'backtrackingShrink', params.mine_lambda_shrink, ...
                'PaprWeight', params.mine_papr_weight, ...
                'verbose', false);

        case 'VARSHNEY'
            z_current = Varshney(z, idx_comm, idx_radar, ...
                'OutputMode', params.varshney_output_mode, ...
                'maxIt', params.varshney_iteration, ...
                'epsStop', params.varshney_eps_stop, ...
                'verbose', false);

        case 'WANG'
            z_current = Wang(z, idx_comm, idx_radar, ...
                'L', params.wang_L, ...
                'Iter', params.wang_iteration, ...
                'UseCVX', params.wang_use_cvx, ...
                'Verbose', false);

        otherwise
            error('Unknown waveform key: %s', waveform_keys{w});
    end

    z_current = z_current(:);
    if numel(z_current) ~= numel(z)
        error('Waveform %s returned a codeword with an unexpected length.', waveform_keys{w});
    end

    z_current(idx_comm) = z(idx_comm);
    z_waveforms{w} = z_current;
end
end

function ValidateCheckpoint(saved, SNR_dB_list, waveform_keys, N, Nc, e, channel_taps, cyclic_prefix_len, Equal)
if ~isfield(saved, 'err_count') || ~isfield(saved, 'bit_count')
    error('Checkpoint file does not contain err_count and bit_count.');
end
if ~isfield(saved, 'SNR_dB_list') || ~isequal(saved.SNR_dB_list, SNR_dB_list)
    error('Checkpoint SNR_dB_list does not match the current simulation.');
end
if isfield(saved, 'waveform_keys') && ~isequal(saved.waveform_keys(:).', waveform_keys(:).')
    error('Checkpoint waveform_keys do not match the current simulation.');
end
if isfield(saved, 'N') && saved.N ~= N
    error('Checkpoint N does not match the current simulation.');
end
if isfield(saved, 'Nc') && saved.Nc ~= Nc
    error('Checkpoint Nc does not match the current simulation.');
end
if isfield(saved, 'e') && abs(saved.e - e) > 1e-12
    error('Checkpoint communication energy ratio e does not match the current simulation.');
end
if isfield(saved, 'channel_taps') && saved.channel_taps ~= channel_taps
    error('Checkpoint channel_taps does not match the current simulation.');
end
if isfield(saved, 'cyclic_prefix_len') && saved.cyclic_prefix_len ~= cyclic_prefix_len
    error('Checkpoint cyclic_prefix_len does not match the current simulation.');
end
if isfield(saved, 'Equal') && saved.Equal ~= Equal
    error('Checkpoint Equal mode does not match the current simulation.');
end
end

function SaveCheckpoint(result_file, ber, err_count, bit_count, mc_done, ...
    SNR_dB_list, waveforms, waveform_keys, waveform_domains, ...
    N, Nc, Nr, modSC, e, monte, channel_taps, cyclic_prefix_len, ...
    Equal, algorithm_params, rng_state, rng_state_after)
save(result_file, ...
    'ber', 'err_count', 'bit_count', 'mc_done', ...
    'SNR_dB_list', 'waveforms', 'waveform_keys', 'waveform_domains', ...
    'N', 'Nc', 'Nr', 'modSC', 'e', 'monte', ...
    'channel_taps', 'cyclic_prefix_len', 'Equal', ...
    'algorithm_params', 'rng_state', 'rng_state_after');
end

function result_path = ResolveResultPath(result_path, result_dir)
result_path = char(result_path);
if isempty(result_path)
    error('Result file path cannot be empty.');
end
[folder_part, ~, ~] = fileparts(result_path);
if isempty(folder_part)
    result_path = fullfile(result_dir, result_path);
end
end

function mc_done = EstimateDoneFromBitCount(bit_count, bits_per_trial)
if isempty(bit_count)
    mc_done = zeros(0, 1);
    return;
end
mc_done = min(bit_count, [], 2) / bits_per_trial;
mc_done = floor(mc_done(:));
end

function bits = QPSKDemod(symbols)
symbols = symbols(:);
bits = zeros(2 * numel(symbols), 1);
bits(1:2:end) = real(symbols) >= 0;
bits(2:2:end) = imag(symbols) >= 0;
end

function DFnT_matrix = DFnT(P)
m = (0:P - 1)' * ones(1, P);
n = ones(P, 1) * (0:P - 1);
fresnel_kernel = exp(1j * pi / P * (m - n).^2);
normalization = (1 / sqrt(P)) * exp((pi / 4) * 1j);
DFnT_matrix = normalization * fresnel_kernel;
end
