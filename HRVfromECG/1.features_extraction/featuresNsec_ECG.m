function [ECG_feat, err] = featuresNsec_ECG(locs_R,fs,paz,ECG_feat,win_len_s,pos)
threshold = mean(diff(locs_R));
k = 0;
err = 0;

% if mod(pos,2) == 0
%     interval = 600:win_len_s:locs_R(end-1)-600;
% else
%     interval = 0:win_len_s:locs_R(end-1);
% end
interval = 600:win_len_s:locs_R(end)-600-win_len_s;
for i = interval
    RR = diff(locs_R(locs_R<=(i+win_len_s) & locs_R>i));
    if ~isempty(RR(RR>20))
        err = err+1;
    end
    RR(RR>20) = [];
    % if ~isempty(RR(RR==0))
    %     err = err+1;
    % end
    if length(RR)>1%~isempty(RR)
        k = k+1;
        meanHR(k) = 60/mean(RR);
        SDNN(k) = HRV.SDNN(RR,0,1);
        RMSSD(k) = HRV.RMSSD(RR,0,1);
        [pLF(k),pHF(k),LFHFratio(k),VLF(k),LF(k),HF(k)] = HRV.fft_val_fun(RR,fs);
    
        for j = 2:length(RR)
            if abs(RR(j)-RR(j-1)) < 0.02*threshold
                b2(j-1) = 0;
                if abs(RR(j)-RR(j-1)) < 0.01*threshold
                    b1(j-1) = 0;
                    if abs(RR(j)-RR(j-1)) < 0.005*threshold
                        b05(j-1) = 0;
                    else
                        b05(j-1) = 1;
                    end
                else
                    b05(j-1) = 1;
                    b1(j-1) = 1;
                end
            else
                b05(j-1) = 1;
                b1(j-1) = 1;
                b2(j-1) = 1;
            end
        end
        ApEn(k) = HRV.ApEn(RR);
        SampEN(k) = sampen(RR,2,0.2);
        %CD(k) = HRV.CD(RR,10,1:length(RR));
        DFA(:,k) = HRV.DFA(RR,1:fix(length(RR)/4),fix(length(RR)/4)+1:length(RR));
        [SD1(k),SD2(k),SD1SD2ratio(k)] = HRV.returnmap_val(RR);
        s05 = binary_seq_to_string(b05);
        s1 = binary_seq_to_string(b1);
        s2 = binary_seq_to_string(b2);
        KC_05(k) = kolmogorov(s05,i);
        KC_1(k) = kolmogorov(s1,i);
        KC_2(k) = kolmogorov(s2,i);
        LZC_05(k) = calc_lz_complexity(b05,'exhaustive',0);
        LZC_1(k) = calc_lz_complexity(b1,'exhaustive',0);
        LZC_2(k) = calc_lz_complexity(b2,'exhaustive',0);
    end
end

vars = [meanHR;SDNN;RMSSD;pLF;pHF;LFHFratio;VLF;LF;HF;ApEn;SampEN;DFA;SD1;SD2;SD1SD2ratio;KC_05;KC_1;KC_2;LZC_05;LZC_1;LZC_2];
[~,c] = find(isnan(vars));
vars(:,c) = [];
for i = 1:size(vars,1)
    ECG_feat(paz,(i-1)*6*8+pos) = mean(vars(i,:));
    ECG_feat(paz,(((i-1)*6)+1)*8+pos) = std(vars(i,:),0);
    ECG_feat(paz,(((i-1)*6)+2)*8+pos) = prctile(vars(i,:),25);
    ECG_feat(paz,(((i-1)*6)+3)*8+pos) = prctile(vars(i,:),75);
    ECG_feat(paz,(((i-1)*6)+4)*8+pos) = kurtosis(vars(i,:),1);
    ECG_feat(paz,(((i-1)*6)+5)*8+pos) = skewness(vars(i,:),1);
    
    % if pos == 6
    % % figure(size(vars,1)*(pos-1)+i)
    %     figure(i)
    %     subplot(2,1,2)
    %     plot(vars(i,:))
    %     hold on
    % end
end

end