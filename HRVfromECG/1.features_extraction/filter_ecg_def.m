function [ecg_filt,n] = filter_ecg_def(ecg,fs_ecg,m_u)
fNy = fs_ecg/2;
% Wp = [1 45]/fNy;
% Ws = [0.05 60]/fNy;
Wp = [10 30]/fNy;
Ws = [1 50]/fNy;
Rp = 0.2;
Rs = 20;

[n,Wp] = cheb1ord(Wp,Ws,Rp,Rs);
[b,a] = cheby1(n,Rp,Wp);

ecg_filt = filtfilt(b,a,ecg);

amp = mean(maxk(abs(ecg_filt),fix(length(ecg_filt)/900)));
if strcmpi(m_u,'uV') == 1 || amp<5000 && amp>=5.1
    disp('ECG aplitude in uV --> not converted')
%     u = 'uV';
elseif strcmpi(m_u,'mV') == 1 || amp<5.1 && amp>=5*10^(-3)
    ecg_filt = ecg_filt*10^3;
    disp('ECG aplitude in mV --> converted in uV')
%     u = 'mV';
elseif strcmpi(m_u,'V') == 1 || amp<5*10^(-3) && amp>=5*10^(-6)
    ecg_filt = ecg_filt*10^6;
    disp('ECG aplitude in V --> converted in uV')
%     u = 'V';
elseif strcmpi(m_u,'nV') == 1 || amp<5*10^5 && amp>=5*10^(3)
    ecg_filt = ecg_filt*10^(-3);
    disp('ECG aplitude in nV --> converted in uV')
%     u = 'nV';
else
    disp('ERRORE --> ECG amplitude phisical dimension not found')
%     u = 'err';
end
end