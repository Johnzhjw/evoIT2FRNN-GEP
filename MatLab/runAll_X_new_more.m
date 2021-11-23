%%
clc;
clear;
close all;

%%
nRun = 10;
NTRACE = 25;

%%
models = {
    'FRNN'
};

algs = {
    'PCMLIA-FRNN'
    'PCMLIA-DFRNN'
    'PCMLIA-FGRNN'
    'PCMLIA-DFGRNN'
    'PCMLIA-BFRNN'
    'PCMLIA-BDFRNN'
    'PCMLIA-BFGRNN'
    'PCMLIA-BDFGRNN'
    'PCMLIA-w-FRNN'
    'PCMLIA-w-DFRNN'
    'PCMLIA-w-FGRNN'
    'PCMLIA-w-DFGRNN'
    'PCMLIA-w-BFRNN'
    'PCMLIA-w-BDFRNN'
    'PCMLIA-w-BFGRNN'
    'PCMLIA-w-BDFGRNN'
    'PCMLIA-Pseu-FRNN'
    'PCMLIA-Pseu-DFRNN'
    'PCMLIA-Pseu-FGRNN'
    'PCMLIA-Pseu-DFGRNN'
    'PCMLIA-Pseu-BFRNN'
    'PCMLIA-Pseu-BDFRNN'
    'PCMLIA-Pseu-BFGRNN'
    'PCMLIA-Pseu-BDFGRNN'
    'PCMLIA-Pseu-F1RNN'
    'PCMLIA-Pseu-DF1RNN'
    'PCMLIA-Pseu-F1GRNN'
    'PCMLIA-Pseu-DF1GRNN'
    'PCMLIA-Pseu-BF1RNN'
    'PCMLIA-Pseu-BDF1RNN'
    'PCMLIA-Pseu-BF1GRNN'
    'PCMLIA-Pseu-BDF1GRNN'
};
for i = 1:length(algs)
    algs{i} = strrep(algs{i}, 'PCMLIA', 'PCMLIA-ADE');
end

fnms = {
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_opt'
    'exp1_w'
    'exp1_w'
    'exp1_w'
    'exp1_w'
    'exp1_w'
    'exp1_w'
    'exp1_w'
    'exp1_w'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_1pseu'
    'exp1_1pseu'
    'exp1_1pseu'
    'exp1_1pseu'
    'exp1_1pseu'
    'exp1_1pseu'
    'exp1_1pseu'
    'exp1_1pseu'
};
for i = 1:length(fnms)
    fnms{i} = strrep(fnms{i}, 'exp1_opt', 'trial07');
    fnms{i} = strrep(fnms{i}, 'exp1_w', 'trial08');
    fnms{i} = strrep(fnms{i}, 'exp1_pseu', 'trial06');
    fnms{i} = strrep(fnms{i}, 'exp1_1pseu', 'trial09');
end

mainfils = cell(1, length(algs));
for i = 1:length(mainfils)
    mainfils{i} = 'exp';
end

filestrs = cell(1, length(algs));
PFfilestrs = cell(1, length(algs));
PFstrs = cell(1, length(algs));
PSfilestrs = cell(1, length(algs));
PSstrs = cell(1, length(algs));
TIMEstrs = cell(1, length(algs));

probs={
    '0941.HK'
    '1288.HK'
    '0005.HK'
    'gnfuv-pi2'  
    'gnfuv-pi3'  
    'gnfuv-pi4'  
    'gnfuv-pi5'  
    'hungaryChickenpox'  
    'SML2010-DATA-1'  
    'SML2010-DATA-2'
    'traffic'
    'Daily-Demand-Forecasting-Orders'
};

probstrs={
    '0941'
    '1288'
    '0005'
    'gnfuv-pi2'  
    'gnfuv-pi3'  
    'gnfuv-pi4'  
    'gnfuv-pi5'  
    'hungaryChickenpox'  
    'SML2010-DATA-1'  
    'SML2010-DATA-2'
    'traffic'
    'Daily_Demand_Forecasting_Orders'
};

mdstrs = {
    'evoFRNN_Predict_'
    'evoDFRNN_Predict_'
    'evoFGRNN_Predict_'
    'evoDFGRNN_Predict_'
    'evoBFRNN_Predict_'
    'evoBDFRNN_Predict_'
    'evoBFGRNN_Predict_'
    'evoBDFGRNN_Predict_'
    'evoFRNN_Predict_'
    'evoDFRNN_Predict_'
    'evoFGRNN_Predict_'
    'evoDFGRNN_Predict_'
    'evoBFRNN_Predict_'
    'evoBDFRNN_Predict_'
    'evoBFGRNN_Predict_'
    'evoBDFGRNN_Predict_'
    'evoFRNN_Predict_'
    'evoDFRNN_Predict_'
    'evoFGRNN_Predict_'
    'evoDFGRNN_Predict_'
    'evoBFRNN_Predict_'
    'evoBDFRNN_Predict_'
    'evoBFGRNN_Predict_'
    'evoBDFGRNN_Predict_'
    'evoFRNN_Predict_'
    'evoDFRNN_Predict_'
    'evoFGRNN_Predict_'
    'evoDFGRNN_Predict_'
    'evoBFRNN_Predict_'
    'evoBDFRNN_Predict_'
    'evoBFGRNN_Predict_'
    'evoBDFGRNN_Predict_'
};

% dims = cell(length(algs), length(probs), nRun);

probName = cell(1, length(algs));

for i = 1:length(probName)
    probName{i} = cell(1, length(probstrs));
    for j = 1:length(probName{i})
        if contains(algs{i}, 'F1')
            probName{i}{j} = sprintf('%s%s_flagT2_0', mdstrs{i}, probstrs{j});
        else
            probName{i}{j} = sprintf('%s%s', mdstrs{i}, probstrs{j});
        end
    end
    for j = 1:3
        probName{i}{j} = strrep(probName{i}{j}, 'Predict_', 'Predict_Stock_');
    end
    for j = 4:length(probName{i})
        probName{i}{j} = strrep(probName{i}{j}, 'Predict_', 'Predict_TimeSeries_');
    end
end

modelName = {
    'FRNN'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
};

% nObj = 2;

nMPI = cell(1, length(algs));

nobjs = cell(1, length(algs));

%%
nPop_all = cell(1, length(algs));
maxFEs=2e3;

%%
for i = 1:length(algs)
    nobjs{i} = 3;
    nPop_all{i} = 12;
    nMPI{i} = 96;
    filestrs{i} = 'DPCCMOLSIA_MP_III';
    PFstrs{i} = 'DPCCMOLSIA_MP_III_FUN';
    PFfilestrs{i} = 'PF';
    PSstrs{i} = 'DPCCMOLSIA_MP_III_VAR';
    PSfilestrs{i} = 'PS';
    TIMEstrs{i} = 'TIME_DPCCMOLSIA_MP_III';
end

tINDs    = 1:length(algs);
algs     = algs(tINDs);
fnms     = fnms(tINDs);
mdstrs   = mdstrs(tINDs);
% dims     = dims(tINDs);
nobjs    = nobjs(tINDs);
nPop_all = nPop_all(tINDs);
nMPI     = nMPI(tINDs);
filestrs = filestrs(tINDs);
PFstrs   = PFstrs(tINDs);
PSstrs   = PSstrs(tINDs);
TIMEstrs = TIMEstrs(tINDs);

tINDs_prob = 1:11;
probs = probs(tINDs_prob);
probstrs = probstrs(tINDs_prob);
for i = 1:length(probName)
%     dims{i} = dims{i}(tINDs_prob);
    probName{i} = probName{i}(tINDs_prob);
end

%%
tarINDs = 1:length(algs);%15;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lenINDs = length(tarINDs);

%%
candidRand = {
     'FRNN'
};

curAlg = [];
ALL_TIMES_NN = cell(length(candidRand),1);
mean_time_NN = 0;

ALL_PFs_4_run_trace_NN = cell(length(candidRand),1);
ALL_min_objs_run_NN = cell(length(candidRand),1);
ALL_minInd_objs_run_NN = cell(length(candidRand),1);
ALL_minInv_objs_run_NN = cell(length(candidRand),1);

%%
for iRnd = 1:length(candidRand)
%     for i = 1:length(fnms)
%         algs{i} = strrep(algs{i}, 'PCMLEA', candidRand{iRnd});
%         fnms{i} = strrep(fnms{i}, 'PCMLEA', candidRand{iRnd});
%     end
    
    curAlg = candidRand{iRnd};
    
    readPF;
    readPF_trace;
    arrangePF;
    arrangePF_trace;
    ALL_PFs_4_run_trace_NN{iRnd} = all_PFs_4_run_trace;
    ALL_min_objs_run_NN{iRnd} = all_min_over_run;
    ALL_minInd_objs_run_NN{iRnd} = all_minInd_over_run;
    ALL_minInv_objs_run_NN{iRnd} = all_minInv_over_run;
    plotBER_trace_comp;
    readTIME;
    ALL_TIMES_NN{iRnd} = all_TIMEs;

%     for i = 1:length(fnms)
%         algs{i} = strrep(algs{i}, candidRand{iRnd}, 'PCMLEA');
%         fnms{i} = strrep(fnms{i}, candidRand{iRnd}, 'PCMLEA');
%     end
end

tmp_sz = 3;
table_NN_all_mean_min_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_mean_min2_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_mean_minInd_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_both_mean_min_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_min_min_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_min_min2_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_min_minInv_ACE_run = cell(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_both_min_min_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_mean_final_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);
table_NN_all_both_mean_final_ACE_run = zeros(length(candidRand)*length(probs)*tmp_sz, lenINDs);

[tmp_str_val tmp_str_i] = sort(candidRand);

for iRnd_r = 1:length(candidRand)
    iRnd = tmp_str_i(iRnd_r);
    for iExpSet = 1 : lenINDs
        for iProb = 1 : length(probs)
            for i = 1 : tmp_sz
                tmp_r = (iRnd_r-1)*length(probs)*tmp_sz+(iProb-1)*tmp_sz+i;
                table_NN_all_mean_min_ACE_run(tmp_r, iExpSet) = mean(ALL_min_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                table_NN_all_mean_minInd_ACE_run(tmp_r, iExpSet) = mean(ALL_minInd_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                [tmp_v, tmp_i] = min(ALL_min_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,:));
                table_NN_all_min_min_ACE_run(tmp_r, iExpSet) = tmp_v;
                table_NN_all_min_minInv_ACE_run{tmp_r, iExpSet} = [tmp_i; ALL_minInv_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,tmp_i)];
                min2_ACE = 0;
                iRun = tmp_i;
                tmp = ALL_minInv_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,tmp_i);
                iTrc = tmp{1}(1);
                iInv = tmp{1}(2);
                if i == 1
                    min2_ACE = ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,2,iProb,iRun,iTrc}(iInv,1);
                elseif i == 2
                    min2_ACE = ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,1,iProb,iRun,iTrc}(iInv,1);
                end
                if i == 1
                    table_NN_all_min_min2_ACE_run(tmp_r+1, iExpSet) = min2_ACE;
                elseif i == 2
                    table_NN_all_min_min2_ACE_run(tmp_r-1, iExpSet) = min2_ACE;
                end
                mean2_ACE = 0;
                for iRun = 1:nRun
                    tmp = ALL_minInv_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,iRun);
                    iTrc = tmp{1}(1);
                    iInv = tmp{1}(2);
                    if i == 1
                        mean2_ACE = mean2_ACE + ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,2,iProb,iRun,iTrc}(iInv,1);
                    elseif i == 2
                        mean2_ACE = mean2_ACE + ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,1,iProb,iRun,iTrc}(iInv,1);
                    end
                end
                if i == 1
                    table_NN_all_mean_min2_ACE_run(tmp_r+1, iExpSet) = mean2_ACE/nPop_all{iExpSet};
                elseif i == 2
                    table_NN_all_mean_min2_ACE_run(tmp_r-1, iExpSet) = mean2_ACE/nPop_all{iExpSet};
                end
                if i == 3
                    iRun = tmp_i;
                    tmp = ALL_minInv_objs_run_NN{iRnd}(iExpSet,1,i,iProb,1,tmp_i);
                    iTrc = tmp{1}(1);
                    iInv = tmp{1}(2);
                    table_NN_all_both_min_min_ACE_run(tmp_r-2, iExpSet) = ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,1,iProb,iRun,iTrc}(iInv,1);
                    table_NN_all_both_min_min_ACE_run(tmp_r-1, iExpSet) = ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,2,iProb,iRun,iTrc}(iInv,1);
                    tmpInds = ALL_minInd_objs_run_NN{iRnd}(iExpSet,1,1,iProb,1,:);
                    mean_train = 0;
                    mean_test  = 0;
                    for iInd = 1:length(tmpInds)
                        iRun = iInd;
                        tmp = ALL_minInv_objs_run_NN{iRnd}(iExpSet,1,1,iProb,1,iInd);
                        iTrc = tmp{1}(1);
                        iInv = tmp{1}(2);
                        mean_train = mean_train + ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,1,iProb,iRun,iTrc}(iInv,1);
                        mean_test  = mean_test  + ALL_PFs_4_run_trace_NN{iRnd}{iExpSet,1,2,iProb,iRun,iTrc}(iInv,1);
                    end
                    table_NN_all_both_mean_min_ACE_run(tmp_r-2, iExpSet) = mean_train/length(tmpInds);
                    table_NN_all_both_mean_min_ACE_run(tmp_r-1, iExpSet) = mean_test/length(tmpInds);
                end
                table_NN_all_mean_final_ACE_run(tmp_r, iExpSet) = all_mean_over_trace(iExpSet,1,i,iProb,1,end);
                table_NN_all_both_mean_final_ACE_run(tmp_r, iExpSet) = all_mean_over_trace2(iExpSet,1,i,iProb,1,end);
            end
        end
    end
end

%%
tmp_inds = table_NN_all_min_minInv_ACE_run(1:3:33,25:32);
tmp_ind_table = zeros(88,3);
for i=1:11
    for j=1:8
        tmp_ind_table(j+(i-1)*8,1)=tmp_inds{i,j}{1};
        tmp_ind_table(j+(i-1)*8,2)=tmp_inds{i,j}{2}(1);
        tmp_ind_table(j+(i-1)*8,3)=tmp_inds{i,j}{2}(2);
    end
end
