%%
all_HVs_raw = cell(length(algs),4);
all_meanHVs_raw = cell(length(algs),4);
all_stdHVs_raw = cell(length(algs),4);

for i = 1:length(algs)
    for k = 1:4
        ffile = sprintf('%s/%s/',mainfils{i},fnms{i});
        fname = sprintf('slurm-*');
        fname = dir([ffile fname]);
        fname1= sprintf('%s\\%s',fname(1).folder,fname(1).name);
        tmp_mat_HV = zeros(nRun, NTRACE+1);
        for iRun = 1:nRun
            for iTrace = 0:NTRACE
                if k == 1
                    tarStr = sprintf('%s-OBJ_%d-DIM_%d-Run_%d-TRAIN:[ ]*%d%% --- [0-9.]*', ...
                        probName{i}{1}, nobjs{i}, dims{i}{iRun}, iRun, iTrace*100/NTRACE);
                elseif k == 2
                    tarStr = sprintf('%s-OBJ_%d-DIM_%d-Run_%d-TEST:[ ]*%d%% --- [0-9.]*', ...
                        probName{i}{1}, nobjs{i}, dims{i}{iRun}, iRun, iTrace*100/NTRACE);
                elseif k == 3
                    tarStr = sprintf('%s-OBJ_%d-DIM_%d-Run_%d-FINAL:[ ]*%d%% --- [0-9.]*', ...
                        probName{i}{1}, nobjs{i}, dims{i}{iRun}, iRun, iTrace*100/NTRACE);
                elseif k == 4
                    tarStr = sprintf('%s-OBJ_%d-DIM_%d-Run_%d-VALIDATION:[ ]*%d%% --- [0-9.]*', ...
                        probName{i}{1}, nobjs{i}, dims{i}{iRun}, iRun, iTrace*100/NTRACE);
                end
                s = fileread(fname1);   
                expr = tarStr;
                str = regexp(s,expr,'match');
                if length(str) ~= 1
                    fprintf('%s - the number of strings found is not 1.\n', tarStr);
                end
                regx = '--- [0-9.]+';
                data = regexp(str{end},regx,'match');
                if length(data) ~= 1
                    fprintf('%s - the number of strings found in str{1} is not 1.\n', tarStr);
                end
                tmp  = sscanf(data{1}, '--- %f');
                tmp_mat_HV(iRun,iTrace+1) = tmp;
            end
        end
        all_HVs_raw{i,k} = tmp_mat_HV;
        all_meanHVs_raw{i,k} = mean(tmp_mat_HV);
        all_stdHVs_raw{i,k} = std(tmp_mat_HV);
    end
end

%%
all_meanHVs = cell(length(algs),length(models),length(allprobnames),3);

for iAlg = 1:length(algs)
    for i=1:length(models)
        for j=1:length(allprobnames)
            for k=1:4
                all_meanHVs{iAlg,i,j,k} = ...
                all_meanHVs_raw{iAlg,k}((i-1)*length(allprobnames)+j,:);
            end
        end
    end
end

%%
all_stdHVs = cell(length(algs),length(models),length(allprobnames),3);

for iAlg = 1:length(algs)
    for i=1:length(models)
        for j=1:length(allprobnames)
            for k=1:4
                all_stdHVs{iAlg,i,j,k} = ...
                all_stdHVs_raw{iAlg,k}((i-1)*length(allprobnames)+j,:);
            end
        end
    end
end
