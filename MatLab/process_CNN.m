%%
str_fnms = {
    'NN-EP1'
    'NN-EP10'
    'NN-EP100'
};

nRun = 10;
nClass = 2;
all_BER_sta = cell(length(str_fnms),8,2);
all_mean_BER_sta = cell(length(str_fnms),8,2);
all_TP_FN_sta = cell(length(str_fnms),8,2,4);

%%
for iFile = 1:length(str_fnms)
    all_sta_NN_EP = importdata(str_fnms{iFile});
    for noise_tag = 1:2
        for dup_tag = 1:2
            for ran_tag = 1:2
                tmp_train = zeros(nRun,nClass);
                tmp_test  = zeros(nRun,nClass);
                tmp_sta_  = zeros(2,4,nRun,nClass);
                for iRun = 1:nRun
                    iRow_beg = (noise_tag-1)*2*2*nRun*8+(dup_tag-1)*2*nRun*8+(ran_tag-1)*nRun*8+(iRun-1)*4+1;
                    iRow_end = (noise_tag-1)*2*2*nRun*8+(dup_tag-1)*2*nRun*8+(ran_tag-1)*nRun*8+(iRun-1)*4+4;
                    tmp_sta_(1,1,iRun,:) = all_sta_NN_EP(iRow_beg+0,:);
                    tmp_sta_(1,2,iRun,:) = all_sta_NN_EP(iRow_beg+1,:);
                    tmp_sta_(1,3,iRun,:) = all_sta_NN_EP(iRow_beg+2,:);
                    tmp_sta_(1,4,iRun,:) = all_sta_NN_EP(iRow_beg+3,:);
                    tmp_BER = all_sta_NN_EP(iRow_end,:)./(all_sta_NN_EP(iRow_beg,:)+all_sta_NN_EP(iRow_end,:));
                    tmp_train(iRun,:) = tmp_BER;
                    iRow_beg = iRow_beg + nRun*4;
                    iRow_end = iRow_end + nRun*4;
                    tmp_sta_(2,1,iRun,:) = all_sta_NN_EP(iRow_beg+0,:);
                    tmp_sta_(2,2,iRun,:) = all_sta_NN_EP(iRow_beg+1,:);
                    tmp_sta_(2,3,iRun,:) = all_sta_NN_EP(iRow_beg+2,:);
                    tmp_sta_(2,4,iRun,:) = all_sta_NN_EP(iRow_beg+3,:);
                    tmp_BER = all_sta_NN_EP(iRow_end,:)./(all_sta_NN_EP(iRow_beg,:)+all_sta_NN_EP(iRow_end,:));
                    tmp_test(iRun,:) = tmp_BER;
                end
                theRow = (noise_tag-1)*2*2+(dup_tag-1)*2+ran_tag;
                all_BER_sta{iFile,theRow,1} = tmp_train;
                all_BER_sta{iFile,theRow,2} = tmp_test;
                all_mean_BER_sta{iFile,theRow,1} = mean(tmp_train);
                all_mean_BER_sta{iFile,theRow,2} = mean(tmp_test);
                for iTrTs = 1:2
                    for iSta = 1:4
                        tmp_data  = zeros(nRun,nClass);
                        tmp_data(:,:) = tmp_sta_(iTrTs,iSta,:,:);
                        all_TP_FN_sta{iFile,theRow,iTrTs,iSta} = tmp_data;
                    end
                end
            end
        end
    end
end

%%
all_test_BER = zeros(8,length(str_fnms));

for iExp = 1:8
    for iEP = 1:length(str_fnms)
        all_test_BER(iExp,iEP) = mean(all_mean_BER_sta{iEP,iExp,2});
    end
end

%%
all_test_BER_full = zeros(8*length(str_fnms),2);
ind_row = 1;

for iExp = 1:8
    for iEP = 1:length(str_fnms)
        all_test_BER_full(ind_row,:) = all_mean_BER_sta{iEP,iExp,2};
        ind_row = ind_row + 1;
    end
end
