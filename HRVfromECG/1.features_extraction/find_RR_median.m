function ECG_plot = find_RR_median(ecg_filt,hypa_res,fs,ECG_plot,paz)
%% ECG features
Nep=fix(length(ecg_filt)/(30*fs));
errore = [];
ecg_R = [];
locs_R = [];
hyp_R = [];

tm = [1:length(ecg_filt)]/fs;

wt = modwt(ecg_filt,5);
wtrec = zeros(size(wt));
wtrec(4:5,:) = wt(4:5,:);
y = imodwt(wtrec,'sym4');

y = abs(y).^2;
[~,locs] = findpeaks(y,tm,'MinPeakHeight',100,'MinPeakDistance',0.4);
locs_cam = locs*fs;
ecg_peaks = ecg_filt(locs_cam);

peaks_pos = 0;
for p = 1:length(ecg_peaks)
    if ecg_peaks(p)>0
        peaks_pos = peaks_pos+1;
    elseif ecg_peaks(p)<0
        peaks_pos = peaks_pos-1;
    end
end

for i = 1:Nep
    div = 2;
    if i ~= 1 && i ~= Nep 
        ecg_filt_ep = ecg_filt(30*fs*(i-1)+1-2*fs:30*fs*i+2*fs);
        tm = -2+(1/fs):1/fs:32;
    elseif i == 1 
        ecg_filt_ep = ecg_filt(30*fs*(i-1)+1:30*fs*i+2*fs);
        tm = 1/fs:1/fs:32;
    else
        ecg_filt_ep = ecg_filt(30*fs*(i-1)+1-2*fs:30*fs*i);
        tm = -2+(1/fs):1/fs:30;
    end    
    % tm = [1:length(ecg_filt_ep)]/fs;

    wt = modwt(ecg_filt_ep,5);
    wtrec = zeros(size(wt));
    wtrec(4:5,:) = wt(4:5,:);
    y = imodwt(wtrec,'sym4');    
    y = abs(y).^2;

    [testpeaks,~] = findpeaks(y,tm,'MinPeakHeight',100,'MinPeakDistance',0.6);
    while length(testpeaks)<20 && div<1000
        [testpeaks,~] = findpeaks(y,tm,'MinPeakHeight',100/div,'MinPeakDistance',0.6);
        div = div+1;
    end

    div = 2;
    med = median(testpeaks);

    testpeaks_std = testpeaks(testpeaks>=(median(testpeaks)-2*std(testpeaks)) ...
        & testpeaks<=(median(testpeaks)+2*std(testpeaks)));
    med_change = (med-median(testpeaks_std))/med;
    while med_change > 0.1 || med_change < -0.1 || median(testpeaks_std)-std(testpeaks_std) <= 0 ...
            || std(testpeaks_std)/median(testpeaks_std) > 0.8 || length(testpeaks)== length(testpeaks_std) && length(testpeaks_std)>10
        s = 2;
        med = median(testpeaks_std);
        len1 = length(testpeaks_std);
        testpeaks_std = testpeaks_std(testpeaks_std>=(median(testpeaks_std)-s*std(testpeaks_std)) ...
            & testpeaks_std<=(median(testpeaks_std)+s*std(testpeaks_std)));
        while len1 == length(testpeaks_std)
            s = 0.8*s;
            testpeaks_std = testpeaks_std(testpeaks_std>=(median(testpeaks_std)-s*std(testpeaks_std)) ...
                & testpeaks_std<=(median(testpeaks_std)+s*std(testpeaks_std)));
        end
        med_change = (med-median(testpeaks_std))/med;
    end
    PeakHeightTreshold = median(testpeaks_std)-std(testpeaks_std);

    [~,locs] = findpeaks(y,tm,'MinPeakHeight',PeakHeightTreshold/2.5,'MinPeakDistance',0.4);
    while length(locs)<0.8*length(testpeaks_std) && div<100
        div = div+1;
        [~,locs] = findpeaks(y,tm,'MinPeakHeight',PeakHeightTreshold/div,'MinPeakDistance',0.4);
    end
    ecg_filt_ep = ecg_filt(30*fs*(i-1)+1:30*fs*i);
    locs = locs(locs <= 30 & locs > 0);
    locs_cam = locs*fs;
    ecg_peaks = ecg_filt_ep(locs_cam);
    
    y = y(tm <= 30 & tm > 0);
    tm = [1:30*fs]/fs;
    
    if peaks_pos>0 %&& length(ecg_peaks)>5
        for p = 1:length(ecg_peaks)
            if ecg_peaks(p)<0
                if locs_cam(p)-ceil(0.05*fs) <= 0
                    [other_peaks,other_locs] = findpeaks(y(1:locs_cam(p)+ceil(0.05*fs)), ...
                    tm(1:locs_cam(p)+ceil(0.05*fs)),'MinPeakHeight',PeakHeightTreshold/(div+2));
                elseif locs_cam(p)+ceil(0.05*fs) > length(y)
                    [other_peaks,other_locs] = findpeaks(y(locs_cam(p)-ceil(0.05*fs):length(y)), ...
                    tm(locs_cam(p)-ceil(0.05*fs):length(y)),'MinPeakHeight',PeakHeightTreshold/(div+2));
                else
                    [other_peaks,other_locs] = findpeaks(y(locs_cam(p)-ceil(0.05*fs):locs_cam(p)+ceil(0.05*fs)), ...
                    tm(locs_cam(p)-ceil(0.05*fs):locs_cam(p)+ceil(0.05*fs)),'MinPeakHeight',PeakHeightTreshold/(div+2));
                end
    
                other_locs(other_peaks == y(locs_cam(p))) = [];
                other_peaks(other_peaks == y(locs_cam(p))) = [];
    
                if isempty(other_peaks) == 0
                    other_locs_cam = other_locs*fs;
                    ecg_peaks(p) = max(ecg_filt_ep(other_locs_cam));
                    locs_cam(p) = other_locs_cam(ecg_filt_ep(other_locs_cam)==ecg_peaks(p));
                    locs(p) = locs_cam(p)/fs;
                end
            end
        end
        locs_cam(ecg_peaks<0)=[];
        locs(ecg_peaks<0)=[];
        ecg_peaks(ecg_peaks<0)=[];
        
    elseif peaks_pos<=0 %&& length(ecg_peaks)>5
        for p = 1:length(ecg_peaks)
            if ecg_peaks(p)>0
                if locs_cam(p)-ceil(0.05*fs) <= 0
                    [other_peaks,other_locs] = findpeaks(y(1:locs_cam(p)+ceil(0.05*fs)), ...
                    tm(1:locs_cam(p)+ceil(0.05*fs)),'MinPeakHeight',PeakHeightTreshold/(div+2));
                elseif locs_cam(p)+ceil(0.05*fs) > length(y)
                    [other_peaks,other_locs] = findpeaks(y(locs_cam(p)-ceil(0.05*fs):length(y)), ...
                    tm(locs_cam(p)-ceil(0.05*fs):length(y)),'MinPeakHeight',PeakHeightTreshold/(div+2));
                else
                    [other_peaks,other_locs] = findpeaks(y(locs_cam(p)-ceil(0.05*fs):locs_cam(p)+ceil(0.05*fs)), ...
                    tm(locs_cam(p)-ceil(0.05*fs):locs_cam(p)+ceil(0.05*fs)),'MinPeakHeight',PeakHeightTreshold/(div+2));
                end
    
                other_locs(other_peaks == y(locs_cam(p))) = [];
                other_peaks(other_peaks == y(locs_cam(p))) = [];
    
                if isempty(other_peaks) == 0
                    other_locs_cam = other_locs*fs;
                    ecg_peaks(p) = min(ecg_filt_ep(other_locs_cam));
                    locs_cam(p) = other_locs_cam(ecg_filt_ep(other_locs_cam)==ecg_peaks(p));
                    locs(p) = locs_cam(p)/fs;
                end
            end
        end
        locs_cam(ecg_peaks>0)=[];
        locs(ecg_peaks>0)=[];
        ecg_peaks(ecg_peaks>0)=[];
    else
        locs = 0;
        errore = [errore,i];
    end
    
    
    if ~isempty(locs_R) && ~isempty(locs) && (locs(1)+30*(i-1))-locs_R(end) == 0
        ecg_R(end) = [];
        locs_R(end) = [];
        hyp_R(end) = [];
    end

    ecg_R = [ecg_R,ecg_peaks];
    locs_R = [locs_R,locs+30*(i-1)];
    hyp_R = [hyp_R,hypa_res(locs_cam+fs*30*(i-1))];
end
ECG_plot{paz,1} = ecg_filt;
ECG_plot{paz,2} = ecg_R;
ECG_plot{paz,3} = locs_R;
ECG_plot{paz,4} = hypa_res;
ECG_plot{paz,5} = hyp_R;

end