%% ILC
test_num=31;
FILE_NAME=sprintf("E:\\zynq_test_data\\rfdc_adc\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_pa_ILC_v0.6_test%d.bin",test_num);
FILE_NAME_2="E:\zynq_test_data\rfdc_dac\test_1t1r_1GHz\ILC_test\100mhz\data_38400_v0.77_test1.dat";
IQ_NUM=38400;
SAMPLE_RATE=1000e6;
voltage=0.77;
[~, ~, IQ_RX] = import_iq_binary(FILE_NAME, IQ_NUM); 
[~, ~, IQ_TX] = import_iq_binary(FILE_NAME_2, IQ_NUM);
pa_out1=IQ_RX;
u_k=IQ_TX;
[pa_out1,u_k,~] = align_circular_signal(pa_out1,u_k);
%% 测pa_out
gain_peak = max(abs(pa_out1))/max(abs(u_k));
gain_rms = rms(pa_out1)/rms(u_k);
gain=pa_out1./u_k;
pa_out_plot=pa_out1;
pacoe=gain_peak;
%----------------------update---------------------------------------------
%% 第一次
s1=u_k;
u=1;
e_k=s1*pacoe-pa_out1;
u_k1=u_k+u*e_k./gain;
%%
z0=u_k1;
z0=round(z0./(max(abs(z0)))*voltage*2^15-10);
SAVE_NAME = sprintf("E:\\zynq_test_data\\rfdc_dac\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_dpd_ILC_v0.6_test%d_1.dat",test_num);

fid = fopen(SAVE_NAME, 'wb');

for i = 1:length(z0)
    fwrite(fid, real(z0(i)), 'int16');
    fwrite(fid, imag(z0(i)), 'int16');
end

fclose(fid);
I = real(z0);
Q = imag(z0);

IQ = I + 1i * Q;

pwelch(IQ, [], [], [], 'centered',SAMPLE_RATE);
%% 迭代2次
FILE_NAME_3 = sprintf("E:\\zynq_test_data\\rfdc_adc\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_pa_dpd_ILC_v0.6_test%d_1.bin",test_num);
[~, ~, IQ_RX_DPD] = import_iq_binary(FILE_NAME_3, IQ_NUM); 
u_k=u_k1;
pa_out2=IQ_RX_DPD;
%把u_k输入功放,得到第二次功放输出
[pa_out2,u_k,~] = align_circular_signal(pa_out2,u_k);
%% 测pa_out
gain=pa_out2./u_k;
e_k=s1*pacoe-pa_out2;
u_k1=u_k+u*e_k./gain;
pa_out_2_plot = pa_out2; 
%%
z0=u_k1;
z0=round(z0./(max(abs(z0)))*voltage*2^15-10);
SAVE_NAME = sprintf("E:\\zynq_test_data\\rfdc_dac\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_dpd_ILC_v0.6_test%d_2.dat",test_num);

fid = fopen(SAVE_NAME, 'wb');

for i = 1:length(z0)
    fwrite(fid, real(z0(i)), 'int16');
    fwrite(fid, imag(z0(i)), 'int16');
end

fclose(fid);
I = real(z0);
Q = imag(z0);

IQ = I + 1i * Q;

pwelch(IQ, [], [], [], 'centered',SAMPLE_RATE);
%% 迭代3次
FILE_NAME_3 = sprintf("E:\\zynq_test_data\\rfdc_adc\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_pa_dpd_ILC_v0.6_test%d_2.bin",test_num);
[~, ~, IQ_RX_DPD] = import_iq_binary(FILE_NAME_3, IQ_NUM); 
u_k=u_k1;
pa_out3=IQ_RX_DPD;
[pa_out3,u_k,~] = align_circular_signal(pa_out3,u_k);
plot_signals(u_k,pa_out3,SAMPLE_RATE)
     %% 测pa_out
gain=pa_out3./u_k;
e_k=s1*pacoe-pa_out3;
u_k1=u_k+u*e_k./gain;
pa_out_3_plot = pa_out3; 
%%
z0=u_k1;
z0=round(z0./(max(abs(z0)))*voltage*2^15-10);
SAVE_NAME = sprintf("E:\\zynq_test_data\\rfdc_dac\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_dpd_ILC_v0.6_test%d_3.dat",test_num);

fid = fopen(SAVE_NAME, 'wb');

for i = 1:length(z0)
    fwrite(fid, real(z0(i)), 'int16');
    fwrite(fid, imag(z0(i)), 'int16');
end

fclose(fid);
I = real(z0);
Q = imag(z0);

IQ = I + 1i * Q;

pwelch(IQ, [], [], [], 'centered',SAMPLE_RATE);
%% 迭代4次
FILE_NAME_3 = sprintf("E:\\zynq_test_data\\rfdc_adc\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_pa_dpd_ILC_v0.6_test%d_3.bin",test_num);
[~, ~, IQ_RX_DPD] = import_iq_binary(FILE_NAME_3, IQ_NUM); 
u_k=u_k1;
pa_out4=IQ_RX_DPD;
[pa_out4,u_k,~] = align_circular_signal(pa_out4,u_k);
plot_signals(u_k,pa_out4,SAMPLE_RATE)
%% 测pa_out
gain=pa_out4./u_k;
e_k=s1*pacoe-pa_out4;
u_k1=u_k+u*e_k./gain;
pa_out_4_plot = pa_out4; 
%%
z0=u_k1;
z0=round(z0./(max(abs(z0)))*voltage*2^15-10);
SAVE_NAME = sprintf("E:\\zynq_test_data\\rfdc_dac\\test_1t1r_1GHz\\ILC_test\\100mhz\\data_dpd_ILC_v0.6_test%d_4.dat",test_num);

fid = fopen(SAVE_NAME, 'wb');

for i = 1:length(z0)
    fwrite(fid, real(z0(i)), 'int16');
    fwrite(fid, imag(z0(i)), 'int16');
end

fclose(fid);
I = real(z0);
Q = imag(z0);

IQ = I + 1i * Q;

pwelch(IQ, [], [], [], 'centered',SAMPLE_RATE);
%%
function [I, Q, complex_signal] = import_iq_binary(filename, iq_pairs_count)
% IMPORT_IQ_BINARY 导入二进制格式的交错IQ数据
%   [I, Q, complex_signal] = import_iq_binary(filename, iq_pairs_count)
%
% 输入参数:
%   filename - 二进制文件名
%   iq_pairs_count - 预期的IQ对数
%
% 输出参数:
%   I - I信号数组(int16)
%   Q - Q信号数组(int16)
%   complex_signal - 复数信号(double)

    % 计算预期字节数 (每组IQ=4字节)
    expected_bytes = iq_pairs_count * 4;
    
    % 获取文件信息
    file_info = dir(filename);
    if isempty(file_info)
        error('文件不存在: %s', filename);
    end
    
    % 检查文件大小是否匹配
    if file_info.bytes ~= expected_bytes
        warning('文件大小(%d字节)与预期(%d字节)不匹配', ...
                file_info.bytes, expected_bytes);
        iq_pairs_count = floor(file_info.bytes / 4);
    end
    
    % 打开文件
    fid = fopen(filename, 'rb');
    if fid == -1
        error('无法打开文件: %s', filename);
    end
    
    % 读取所有数据(int16格式)
    raw_data = fread(fid, iq_pairs_count * 2, 'int16');
    fclose(fid);
    
    % 分离I和Q通道 (交错存储: I0,Q0,I1,Q1,...)
    I = raw_data(1:2:end);  % 奇数位是I
    Q = raw_data(2:2:end);  % 偶数位是Q
    
    % 转换为复数信号
    %complex_signal = I + 1i*Q;
    complex_signal = double(I) + 1i*double(Q);
    
    % 验证数据长度
    if length(raw_data) ~= iq_pairs_count * 2
        warning('读取的数据长度(%d)与预期(%d)不符', ...
                length(raw_data), iq_pairs_count * 2);
    end
end
