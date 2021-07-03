%%
folder = 'FUN_statistics_20200804';
nClass = 2;

N_TP_train = zeros(nClass,1);
N_FP_train = zeros(nClass,1);
N_TN_train = zeros(nClass,1);
N_FN_train = zeros(nClass,1);

N_TP_test  = zeros(nClass,1);
N_FP_test  = zeros(nClass,1);
N_TN_test  = zeros(nClass,1);
N_FN_test  = zeros(nClass,1);

%%
tmp_data = all_mean_over_trace(:,1,2,1,1,26);
[tmp_val, iAlg] = min(tmp_data,[],1);

tmp_ind = all_ind_mean_over_trace{iAlg,1,2,1,1,26};

nRun = size(tmp_ind,1);

all_BER = zeros(nRun,2,nClass);

for iRun = 1:nRun
    iPop = tmp_ind(iRun,2);
    tmp_fnm = sprintf('%s/FUN_Classify_CNN_Indus_RUN_%d_IND_%d.tmp',folder,iRun,iPop);
    tmp_sta = importdata(tmp_fnm);
    tmp_count = 1;
    for iClass = 1:nClass
        N_TP_train(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_FP_train(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_TN_train(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_FN_train(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_TP_test(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_FP_test(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_TN_test(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    for iClass = 1:nClass
        N_FN_test(iClass) = tmp_sta(tmp_count);
        tmp_count = tmp_count + 1;
    end
    %
    for iClass = 1:nClass
        all_BER(iRun,1,iClass) = N_FN_train(iClass)/(N_FN_train(iClass)+N_TP_train(iClass));
        all_BER(iRun,2,iClass) = N_FN_test(iClass)/(N_FN_test(iClass)+N_TP_test(iClass));
    end
end
