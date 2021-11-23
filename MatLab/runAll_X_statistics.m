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
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp1_pseu'
    'exp2_pseu'
    'exp2_pseu'
    'exp2_pseu'
    'exp2_pseu'
    'exp2_pseu'
    'exp2_pseu'
    'exp2_pseu'
    'exp2_pseu'
};
for i = 1:length(fnms)
    fnms{i} = strrep(fnms{i}, 'exp1_pseu', 'obj_trial01');
    fnms{i} = strrep(fnms{i}, 'exp2_pseu', 'obj_trial02');
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
};

% dims = cell(length(algs), length(probs), nRun);

probName = cell(1, length(algs));

for i = 1:length(probName)
    probName{i} = cell(1, length(probstrs));
    for j = 1:length(probName{i})
        probName{i}{j} = sprintf('%s%s', mdstrs{i}, probstrs{j});
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
    curAlg = candidRand{iRnd};
    %
    read_statistics;
    plot_FuzzyRule;
end

%%

