%%
tarINDs = [1 2 3 7 8 9 15 16 17];
lenINDs = length(tarINDs);

%%
%all_nd_prc = cell(length(algs),length(modelName),length(probs));
all_hv_prc = zeros(length(algs),length(modelName),length(probs));

nObj = 2;

for i = 1:length(probs)
    PF = [];
    for j = 1:length(modelName)
        for k = 1:length(algs)
            PF = [PF; all_nd_prc{k,j,i}];
        end
    end
    for j = 1:length(modelName)
        for k = 1:length(algs)
            fprintf('HV_%s_%s_%s\n',...
                algs{k}, modelName{j}, probName{k}{i});
            PopObj = all_nd_prc{k,j,i};
            all_hv_prc(k,j,i) = HV(PopObj(:,1:2), PF(:,1:2));
        end
    end
end

%%
%all_hv_prc_x_algs = zeros(length(modelName)*length(probs),length(algs));

all_hv_prc_x_algs = [];
iRow = 0;
indAlgs = tarINDs;
for i = [1 2 3 4 5] %1:length(modelName)
    for j=1:length(probs)
        iRow = iRow + 1;
        for iiii = 1:lenINDs
            iAlg = indAlgs(iiii);
            all_hv_prc_x_algs(iRow,iiii) = all_hv_prc(iAlg,i,j);
        end
    end
end

%%
%all_hv_prc_x_models = zeros(length(algs)*length(probs),length(modelName));

all_hv_prc_x_models = [];
iRow = 0;
for iAlg = tarINDs
    for j=1:length(probs)
        iRow = iRow + 1;
        for i = [1 2 3 4 5] %1:length(modelName)
            all_hv_prc_x_models(iRow,i) = all_hv_prc(iAlg,i,j);
        end
    end
end
